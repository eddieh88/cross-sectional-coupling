"""
Experiment 5a -- classical baseline for the TEMPORAL pathway, before touching
the NN. Every experiment so far (1 through 4b) tested the model's
CROSS-SECTIONAL pathway exclusively -- every generator used has i.i.d.-over-
time returns by construction, so the temporal pathway (own-history-only
causal conv) never had any real signal to find. This is the untested half
of the architecture, flagged explicitly in Experiment 1's findings and
picked as the most promising remaining direction (see results/SUMMARY.md).

`jump_diffusion_generator.py` now supports a threshold-AR(1) momentum term
(`momentum_phi_low` / `momentum_phi_high` / `momentum_threshold_mult`),
independent per asset, deliberately NONLINEAR (a regime switch based on the
size of the previous move, not a single global AR coefficient) so a plain
linear AR(1) is a genuinely weaker baseline than a threshold-aware fit --
the classical toolkit's fair answer to nonlinear temporal dependence, same
spirit as robust/rank correlation for fat tails or GARCH for
heteroskedasticity.

Same discipline as every step so far: before comparing anything to the NN,
(1) confirm the injected structure is real and detectable with standard
diagnostics (ACF, Ljung-Box), then (2) give classical time-series models a
fair shot (linear AR(1) vs. threshold-AR(1)) and measure what's actually
recoverable. This script does both, at the same panel scale used throughout
Experiments 1-4 (keeping rho and jump_coupling active, for realism/
consistency -- momentum is an ADDITIONAL, orthogonal channel, not a
replacement for the cross-sectional structure already tested).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jump_diffusion_generator import JumpDiffusionGenerator

N_ASSETS = 10
N_PATHS = 30
N_STEPS = 300
DT = 1 / 252
RHO = 0.5
LAMBDA_COMMON = 2.0
LAMBDA_IDIO = 1.0
SEED = 0

MOMENTUM_PHI_LOW = 0.0
MOMENTUM_PHI_HIGH = 0.4
MOMENTUM_THRESHOLD_MULT = 1.0


def ljung_box_q(r: np.ndarray, max_lag: int = 5) -> float:
    n = len(r)
    acfs = [np.corrcoef(r[:-k], r[k:])[0, 1] for k in range(1, max_lag + 1)]
    return n * (n + 2) * sum((a ** 2) / (n - k) for k, a in enumerate(acfs, 1))


def fit_ar1(x_prev: np.ndarray, x_next: np.ndarray):
    A = np.vstack([x_prev, np.ones_like(x_prev)]).T
    coef, *_ = np.linalg.lstsq(A, x_next, rcond=None)
    resid = x_next - A @ coef
    r2 = 1 - resid.var() / x_next.var() if x_next.var() > 0 else 0.0
    return coef, r2


def main():
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO, lambda_common=LAMBDA_COMMON,
                                  lambda_idio=LAMBDA_IDIO, momentum_phi_low=MOMENTUM_PHI_LOW,
                                  momentum_phi_high=MOMENTUM_PHI_HIGH,
                                  momentum_threshold_mult=MOMENTUM_THRESHOLD_MULT)
    log_paths = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED)
    rets = np.diff(log_paths, axis=1)  # (P, T, N)

    print(f"=== Panel: {N_PATHS} paths x {N_STEPS} steps x {N_ASSETS} assets, "
          f"rho={RHO}, momentum phi_low={MOMENTUM_PHI_LOW} phi_high={MOMENTUM_PHI_HIGH} ===\n")

    # --- Step 1: confirm the injected autocorrelation is real and detectable ---
    print("--- Diagnostics: is genuine temporal structure present, per asset? ---")
    acf1_all, ljungbox_all = [], []
    for i in range(N_ASSETS):
        r_i = rets[:, :, i].reshape(-1)  # pool across paths (each path independent, fine for a per-asset ACF check)
        r0, r1 = rets[:, :-1, i].reshape(-1), rets[:, 1:, i].reshape(-1)
        acf1 = np.corrcoef(r0, r1)[0, 1]
        acf1_all.append(acf1)
        # Ljung-Box within-path only (autocorrelation across path boundaries is meaningless)
        Q = np.mean([ljung_box_q(rets[p, :, i], max_lag=5) for p in range(N_PATHS)])
        ljungbox_all.append(Q)
    print(f"  mean lag-1 ACF across assets: {np.mean(acf1_all):.4f} "
          f"(range {min(acf1_all):.4f} to {max(acf1_all):.4f})")
    print(f"  mean Ljung-Box Q (5 lags, per-path, critical ~11.07 at 5%): {np.mean(ljungbox_all):.2f}")

    # --- Step 2: classical baselines -- linear AR(1) vs threshold-AR(1), pooled across paths per asset ---
    print("\n--- Classical baselines: linear AR(1) vs threshold-AR(1), per asset ---")
    lin_r2s, tar_r2s = [], []
    for i in range(N_ASSETS):
        r0 = rets[:, :-1, i].reshape(-1)
        r1 = rets[:, 1:, i].reshape(-1)

        _, lin_r2 = fit_ar1(r0, r1)
        lin_r2s.append(lin_r2)

        threshold = MOMENTUM_THRESHOLD_MULT * gen.sigma[i] * np.sqrt(DT)
        low_mask = np.abs(r0) < threshold
        pred = np.zeros_like(r1)
        for mask in [low_mask, ~low_mask]:
            if mask.sum() < 2:
                continue
            coef, _ = fit_ar1(r0[mask], r1[mask])
            A = np.vstack([r0[mask], np.ones(mask.sum())]).T
            pred[mask] = A @ coef
        resid = r1 - pred
        tar_r2 = 1 - resid.var() / r1.var() if r1.var() > 0 else 0.0
        tar_r2s.append(tar_r2)

    print(f"  linear AR(1)      mean R^2 across assets: {np.mean(lin_r2s):.4f}")
    print(f"  threshold-AR(1)   mean R^2 across assets: {np.mean(tar_r2s):.4f}")
    print(f"  relative improvement: {(np.mean(tar_r2s) - np.mean(lin_r2s)) / np.mean(lin_r2s) * 100:.1f}%")


if __name__ == "__main__":
    main()
