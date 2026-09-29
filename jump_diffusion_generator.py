"""
Multivariate jump-diffusion generator with tunable cross-sectional coupling.

Purpose
-------
Generate synthetic multi-asset price paths where BOTH the diffusive and
jump components have a known, controllable split between a "common factor"
(cross-sectional) contribution and an idiosyncratic (per-asset, temporal-only)
contribution. Because the true mixing ratio is a parameter you set, this is
a clean ground-truth benchmark for testing whether a learned interpolation
scalar (temporal vs. cross-sectional mixing) recovers the true value.

Model
-----
For each asset i, log-price increments over [t, t+dt] are:

    dX_i = mu_i * dt
           + diffusion_i
           + jump_i

Diffusion component (factor-structured Brownian motion):
    diffusion_i = sigma_i * ( sqrt(rho)      * dZ_common
                             + sqrt(1 - rho)  * dW_i )

    where dZ_common ~ N(0, dt) is shared across ALL assets, dW_i ~ N(0, dt)
    is independent per asset, and rho in [0, 1] is the DIFFUSIVE cross-
    sectional coupling. rho=0 -> assets diffuse independently. rho=1 -> all
    assets driven by a single common Brownian motion (perfect correlation).

    Optional regime switch (`rho_2` != None): every experiment through Exp6
    used a FIXED rho for the whole path -- real data (Exp6) directly
    falsified this as an assumption about real markets (rolling correlation
    swung 0.31->0.89 across a 3-year window, not a minor wobble). Setting
    `rho_2` activates a single, one-time switch from `rho` to `rho_2` at
    `regime_switch_frac * n_steps` -- the literal "low rho, then a shift to
    high rho mid-series, mimicking a correlation breakdown" test named in
    the original spec's Experiment 3 list, never run until now. `rho_2=None`
    (default) reproduces the original fixed-rho behavior exactly.

    Optional second common factor (`factor_dispersion` > 0): the single
    dZ_common above makes the common component EXACTLY rank-1 -- every
    asset's common exposure is the same shock scaled only by its own
    sigma_i, so return_i / sigma_i is identical across all assets for any
    diffusive event. This is a strong, testable structural assumption, not
    a realistic claim about real markets (which plausibly have several
    common factors -- e.g. sector + market-wide). Setting
    `factor_dispersion` in (0, 1] activates a SECOND independent common
    factor dZ_common2 and gives each asset i a fixed, heterogeneous loading
    angle theta_i ~ Uniform(0, factor_dispersion * pi/2), combining as
    cos(theta_i) * dZ_common + sin(theta_i) * dZ_common2. theta_i=0 for
    every asset (factor_dispersion=0) recovers the original rank-1 case
    exactly. Larger dispersion makes the common component genuinely rank-2 --
    a stress test of whether results that depend on rank-1 structure
    (e.g. the shape-based jump classifier in experiments/experiment3d/3e/3f)
    degrade gracefully or break down as that assumption is relaxed.

Jump component (two-layer compound Poisson):
    - Common jumps: a single Poisson process N_common(t) with intensity
      lambda_common. When it fires, EVERY asset jumps simultaneously (jump
      sizes can still differ in magnitude per asset, but the jump TIMES are
      shared -> this is the cross-sectional coupling channel for jumps).
    - Idiosyncratic jumps: independent Poisson processes N_i(t) per asset
      with intensity lambda_idio, firing and sized independently.

    jump_coupling = lambda_common / (lambda_common + lambda_idio) is the
    analogous "fraction of jump risk that is systemic" knob.

    Jump sizes are drawn from N(jump_mean, jump_std^2) in log-price space
    (Merton-style), independently per asset even for common jump events.

Together, (rho, jump_coupling) give you two independently tunable
cross-sectional coupling parameters -- one for the continuous part, one for
the tail/jump part -- which is useful if you want to test whether a learned
interpolation scalar picks up on diffusive coupling, jump coupling, or a
blend of both.

Optional temporal autocorrelation (`momentum_phi_low` / `momentum_phi_high` /
`momentum_threshold_mult`, all 0.0 by default -> exactly the original i.i.d.-
over-time behavior, bit-for-bit). Every component above is i.i.d. across
time by construction -- rho and jump_coupling only shape CROSS-SECTIONAL
co-movement within a timestep, never how any asset's own future relates to
its own past. That means the model's TEMPORAL pathway (own-history-only,
`src/model.py`'s causal conv) has never had any real signal to find in any
experiment run against this generator so far. Setting `momentum_phi_high !=
momentum_phi_low` activates a threshold-AR(1) momentum term, independent per
asset (no new cross-sectional coupling introduced):

    momentum_i(t) = phi_high * r_i(t-1)   if |r_i(t-1)| >= threshold_i
                  = phi_low  * r_i(t-1)   otherwise

    threshold_i = momentum_threshold_mult * sigma_i * sqrt(dt)   (the asset's
    own typical per-step diffusive scale), added to the next increment. This
    is deliberately NONLINEAR (a regime switch, not a single global AR
    coefficient) so a plain linear AR(1) fit is a genuinely weaker baseline
    than a threshold-aware one -- the fair classical comparison is a
    threshold-AR / Markov-switching-AR fit, not vanilla AR(1).

Usage
-----
    gen = JumpDiffusionGenerator(n_assets=10, rho=0.4, jump_coupling=0.7)
    paths = gen.simulate(n_steps=2000, dt=1/252, n_paths=1, seed=0)
    df = gen.to_dataframe(paths)
    df.to_parquet("synthetic_prices.parquet")
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class JumpDiffusionGenerator:
    n_assets: int = 10
    s0: float = 100.0
    mu: float = 0.05          # annualized drift, same for all assets (scalar or array)
    sigma: float = 0.20       # annualized diffusive vol (scalar or array)
    rho: float = 0.3          # diffusive cross-sectional coupling, in [0, 1]
    lambda_common: float = 2.0   # annualized intensity of common (systemic) jumps
    lambda_idio: float = 1.0     # annualized intensity of idiosyncratic jumps, per asset
    jump_mean: float = -0.02     # mean log-jump size (negative -> crash-skew, Merton-style)
    jump_std: float = 0.05       # std of log-jump size
    factor_dispersion: float = 0.0   # 0 = single common factor (rank-1); >0 activates a second, see module docstring
    momentum_phi_low: float = 0.0    # threshold-AR(1) coefficient below threshold; 0.0 = no momentum term at all
    momentum_phi_high: float = 0.0   # threshold-AR(1) coefficient at/above threshold
    momentum_threshold_mult: float = 1.0   # threshold as a multiple of an asset's own per-step diffusive scale
    rho_2: float | None = None       # None = fixed rho throughout (original behavior); set to activate a single regime switch
    regime_switch_frac: float = 0.5  # fraction of n_steps at which rho switches from `rho` to `rho_2`

    def __post_init__(self):
        self.mu = np.full(self.n_assets, self.mu) if np.isscalar(self.mu) else np.asarray(self.mu)
        self.sigma = np.full(self.n_assets, self.sigma) if np.isscalar(self.sigma) else np.asarray(self.sigma)
        assert 0.0 <= self.rho <= 1.0, "rho must be in [0, 1]"
        assert self.rho_2 is None or 0.0 <= self.rho_2 <= 1.0, "rho_2 must be in [0, 1]"

    @property
    def jump_coupling(self) -> float:
        """Fraction of total jump intensity that is systemic (common)."""
        total = self.lambda_common + self.lambda_idio
        return 0.0 if total == 0 else self.lambda_common / total

    def simulate(self, n_steps: int, dt: float, n_paths: int = 1, seed: int | None = None,
                 return_diagnostics: bool = False, rho_path: np.ndarray | None = None):
        """
        Returns log-price array of shape (n_paths, n_steps + 1, n_assets).

        If `return_diagnostics=True`, also returns a dict with the per-step
        ground truth `z_common` (n_paths, n_steps) and `common_jump_fired`
        boolean (n_paths, n_steps) -- not observable on real data, only
        meaningful for validating a detection method against known truth on
        synthetic data (see experiments/experiment3d_shape_signal_check.py).

        `rho_path`, if given, is an arbitrary array of length `n_steps`
        specifying rho at every timestep directly -- overrides `rho`/`rho_2`/
        `regime_switch_frac` entirely. Use this to inject a REALISTIC, e.g.
        real-data-derived, continuously-varying coupling trajectory as ground
        truth (see experiments/experiment9_real_shaped_rho.py), rather than
        the idealized single clean step tested in Exp7/8.
        """
        rng = np.random.default_rng(seed)
        n = self.n_assets

        log_paths = np.zeros((n_paths, n_steps + 1, n))
        log_paths[:, 0, :] = np.log(self.s0)

        z_common_hist = np.zeros((n_paths, n_steps))
        common_jump_fired = np.zeros((n_paths, n_steps), dtype=bool)

        # Fixed per-asset loading onto the two common factors (see module
        # docstring). theta=0 for every asset when factor_dispersion=0 -> the
        # second draw below is skipped entirely, so rng call order (and thus
        # results) is byte-for-byte identical to the original single-factor code.
        if self.factor_dispersion > 0:
            theta = rng.uniform(0.0, self.factor_dispersion * np.pi / 2, size=n)
            loading_1, loading_2 = np.cos(theta), np.sin(theta)
        else:
            loading_1, loading_2 = np.ones(n), np.zeros(n)

        drift = (self.mu - 0.5 * self.sigma**2) * dt  # (n,)
        has_momentum = (self.momentum_phi_low != 0.0) or (self.momentum_phi_high != 0.0)
        momentum_threshold = self.momentum_threshold_mult * self.sigma * np.sqrt(dt)  # (n,)

        # Per-step rho schedule: constant (self.rho) unless rho_2 is set, in
        # which case it switches once at regime_switch_frac * n_steps. This is
        # the single-regime-break test named in the original spec ("low rho,
        # then a shift to high rho mid-series, mimicking a correlation
        # breakdown") -- rho_2=None reproduces the original constant-rho
        # behavior exactly, byte-for-byte.
        if rho_path is not None:
            assert len(rho_path) == n_steps, "rho_path must have length n_steps"
            rho_schedule = np.asarray(rho_path)
        elif self.rho_2 is not None:
            switch_step = int(n_steps * self.regime_switch_frac)
            rho_schedule = np.where(np.arange(n_steps) < switch_step, self.rho, self.rho_2)
        else:
            rho_schedule = np.full(n_steps, self.rho)
        rho_hist = np.tile(rho_schedule, (n_paths, 1))  # (n_paths, n_steps), ground truth for diagnostics

        for p in range(n_paths):
            prev_ret = np.zeros(n)
            for t in range(n_steps):
                rho_t = rho_schedule[t]
                # --- diffusion: factor-structured Brownian increment ---
                z_common = rng.normal(0.0, np.sqrt(dt))          # shared scalar shock
                if self.factor_dispersion > 0:
                    z_common2 = rng.normal(0.0, np.sqrt(dt))      # second shared shock
                    common_component = loading_1 * z_common + loading_2 * z_common2
                else:
                    common_component = z_common                  # scalar broadcasts, rank-1
                w_idio = rng.normal(0.0, np.sqrt(dt), size=n)     # per-asset shock
                diffusion = self.sigma * (
                    np.sqrt(rho_t) * common_component + np.sqrt(1 - rho_t) * w_idio
                )
                z_common_hist[p, t] = z_common

                # --- jumps: common (shared timing) + idiosyncratic ---
                jump = np.zeros(n)

                # common jump: one Bernoulli draw decides if ALL assets jump this step
                p_common = 1 - np.exp(-self.lambda_common * dt)
                if rng.uniform() < p_common:
                    jump += rng.normal(self.jump_mean, self.jump_std, size=n)
                    common_jump_fired[p, t] = True

                # idiosyncratic jumps: independent Bernoulli per asset
                p_idio = 1 - np.exp(-self.lambda_idio * dt)
                idio_fires = rng.uniform(size=n) < p_idio
                if idio_fires.any():
                    jump[idio_fires] += rng.normal(
                        self.jump_mean, self.jump_std, size=idio_fires.sum()
                    )

                # --- optional threshold-AR(1) momentum, own-history only (no new cross-sectional coupling) ---
                if has_momentum:
                    phi = np.where(np.abs(prev_ret) >= momentum_threshold,
                                   self.momentum_phi_high, self.momentum_phi_low)
                    momentum = phi * prev_ret
                else:
                    momentum = 0.0

                increment = drift + diffusion + jump + momentum
                log_paths[p, t + 1, :] = log_paths[p, t, :] + increment
                if has_momentum:
                    prev_ret = increment

        if return_diagnostics:
            return log_paths, {
                "z_common": z_common_hist,
                "common_jump_fired": common_jump_fired,
                "loading_1": loading_1,
                "loading_2": loading_2,
                "rho_hist": rho_hist,
            }
        return log_paths

    def to_dataframe(self, log_paths, dt: float, start="2020-01-01", path_index: int = 0) -> pd.DataFrame:
        """Convert one simulated path to a tidy OHLC-ish dataframe (close price only, plus log-return)."""
        n_steps = log_paths.shape[1] - 1
        n = self.n_assets
        prices = np.exp(log_paths[path_index])  # (n_steps+1, n)

        freq_days = max(int(round(dt * 365)), 1)
        dates = pd.date_range(start=start, periods=n_steps + 1, freq=f"{freq_days}D")

        cols = {f"asset_{i}": prices[:, i] for i in range(n)}
        df = pd.DataFrame(cols, index=dates)
        df.index.name = "date"
        return df

    def realized_cross_sectional_corr(self, log_paths, path_index: int = 0) -> np.ndarray:
        """Sanity check: realized correlation matrix of log-returns should track rho (roughly)."""
        rets = np.diff(log_paths[path_index], axis=0)  # (n_steps, n)
        return np.corrcoef(rets, rowvar=False)


if __name__ == "__main__":
    # Quick sanity check: does realized correlation track the rho we set?
    for rho in (0.0, 0.3, 0.7, 1.0):
        gen = JumpDiffusionGenerator(n_assets=8, rho=rho, lambda_common=3.0, lambda_idio=1.0)
        paths = gen.simulate(n_steps=5000, dt=1 / 252, n_paths=1, seed=42)
        corr = gen.realized_cross_sectional_corr(paths)
        off_diag_mean = corr[np.triu_indices_from(corr, k=1)].mean()
        print(f"rho={rho:.1f} -> mean off-diagonal realized corr = {off_diag_mean:.3f} "
              f"(jump_coupling = {gen.jump_coupling:.2f})")

    # Save one example dataset
    gen = JumpDiffusionGenerator(n_assets=10, rho=0.4, lambda_common=2.0, lambda_idio=1.0)
    paths = gen.simulate(n_steps=2000, dt=1 / 252, n_paths=1, seed=0)
    df = gen.to_dataframe(paths, dt=1 / 252)
    df.to_parquet("synthetic_prices_example.parquet")
    print("\nSaved example dataset:", df.shape)
    print(df.head())
