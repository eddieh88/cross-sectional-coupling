"""The trivial baseline Experiment 2 compares `alpha` against: realized
cross-sectional correlation, computed directly on data -- no NN involved.
"""

import numpy as np
import torch
from scipy.special import gamma
from scipy.stats import norm


def realized_corr_estimate(x: torch.Tensor) -> float:
    """Pooled realized cross-sectional correlation.

    x: (n_paths, n_steps, n_assets) returns (same tensor the model trains
    on). Pools all (path, timestep) observations into one sample before
    computing the correlation matrix -- valid because paths are independent
    draws of the same cross-sectional structure and returns are i.i.d. over
    time, so this uses the full data budget rather than averaging noisier
    per-path estimates.

    Returns the mean off-diagonal entry of the correlation matrix, on the
    same footing as `JumpDiffusionGenerator.realized_cross_sectional_corr`.
    """
    arr = x.reshape(-1, x.shape[-1]).numpy()  # (n_paths * n_steps, n_assets)
    corr = np.corrcoef(arr, rowvar=False)
    n = corr.shape[0]
    return float(corr[np.triu_indices(n, k=1)].mean())


def _bipower_variation(returns_1d: np.ndarray) -> float:
    """Barndorff-Nielsen & Shephard bipower variation: a per-step diffusive
    variance estimate that is robust to jumps.

    Uses products of ADJACENT returns rather than squares. An isolated jump
    return appears in two such products, each diluted by its (typically
    small, continuous-scale) neighbor -- so a jump barely moves this
    estimate, unlike realized variance (sum of squares) where it dominates.
    This dataset has constant volatility within a run (no time-varying vol
    regime), so one global estimate per series is enough -- no rolling
    window needed.
    """
    r = np.abs(returns_1d)
    return float((np.pi / 2) * np.mean(r[:-1] * r[1:]))


def _naive_threshold_jump_flags(series: np.ndarray, jump_threshold_mult: float = 9.0) -> np.ndarray:
    """Original (naive) jump flag: r_t^2 > jump_threshold_mult * global bipower variation.

    Kept for comparison -- this is the version Experiment 3 (first pass)
    showed conflating jump_corr with rho at high rho. Diagnosis: this
    doesn't fail because the threshold "doesn't scale with volatility" (each
    asset's own diffusive variance is actually invariant to rho by
    construction in this generator -- rho only changes CROSS-asset
    correlation, not any individual asset's marginal variance). It fails
    because a FIXED multiplier isn't a calibrated statistical test at all:
    it has no defined false-positive rate, so nothing guarantees it stays
    rare enough for co-occurring false positives (inevitable when rho makes
    the underlying series correlated) to stay negligible.
    """
    bv = _bipower_variation(series)
    return series**2 > jump_threshold_mult * bv


_MU_4_3 = 2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)  # ~0.8309, BNS tripower quarticity constant
_VARTHETA = (np.pi / 2) ** 2 + np.pi - 5  # ~0.6090, asymptotic variance constant for BV


def _bns_jump_flags(series: np.ndarray, block_size: int = 20, alpha: float = 0.001) -> np.ndarray:
    """Barndorff-Nielsen & Shephard / Huang-Tauchen ratio jump test, applied
    per non-overlapping block, with individual jump identification.

    Fixes the naive test's real flaw: rather than comparing |r_t| to a fixed
    multiple of bipower variation, this normalizes the realized-variance /
    bipower-variation GAP by its own asymptotic standard error (estimated
    via realized tripower quarticity) -- a properly calibrated z-statistic
    under the null of "no jump in this block," with a known, controllable
    false-positive rate, rather than an arbitrary cutoff.

    For each block of `block_size` consecutive returns:
      RV  = sum(r_i^2)                                    -- realized variance
      BV  = (pi/2) * (K/(K-1)) * sum(|r_i| |r_{i-1}|)      -- bipower variation (jump-robust)
      TQ  = K * mu_(4/3)^-3 * (K/(K-2)) * sum(|r_i|^(4/3) |r_{i-1}|^(4/3) |r_{i-2}|^(4/3))
      z   = ((RV-BV)/RV) / sqrt(vartheta/K * max(1, TQ/BV^2))

    z is asymptotically N(0,1) under "no jump in this block." A block is
    flagged only if z exceeds the one-sided critical value for `alpha`
    (default 0.001 -- deliberately conservative since many blocks x many
    assets x many paths are tested here, and an uncontrolled multiple-
    testing false-positive rate is exactly the failure mode being fixed).
    Within a flagged block, only the single most extreme |r_i| is marked as
    the jump, matching standard practice for turning a block-level "a jump
    occurred" test into an individual jump flag.
    """
    z_critical = norm.ppf(1 - alpha)
    n = len(series)
    flags = np.zeros(n, dtype=bool)

    for start in range(0, n - block_size + 1, block_size):
        block = series[start : start + block_size]
        k = len(block)
        if k < 5:  # too short for tripower quarticity (needs >=3 lags)
            continue

        r = np.abs(block)
        rv = float(np.sum(block**2))
        bv = float((np.pi / 2) * (k / (k - 1)) * np.sum(r[:-1] * r[1:]))
        tq = float(
            k * _MU_4_3**-3 * (k / (k - 2))
            * np.sum(r[:-2] ** (4 / 3) * r[1:-1] ** (4 / 3) * r[2:] ** (4 / 3))
        )

        if rv == 0 or bv == 0:
            continue

        rj = (rv - bv) / rv
        denom = np.sqrt(_VARTHETA / k * max(1.0, tq / bv**2))
        z = rj / denom if denom > 0 else 0.0

        if z > z_critical:
            local_idx = np.argmax(r)  # most extreme move in the block = the jump
            flags[start + local_idx] = True

    return flags


def diffusive_and_jump_baselines(x: torch.Tensor, method: str = "bns", **kwargs):
    """Classical disentanglement baseline: jump detection, then two
    SEPARATE correlation signals -- one for diffusive coupling, one for
    jump-timing coupling. No NN, no training.

    `method="bns"` (default): calibrated ratio-statistic test, see
    `_bns_jump_flags`. `method="naive"`: fixed-multiple threshold, see
    `_naive_threshold_jump_flags` -- kept for before/after comparison, since
    it's the version that showed jump_corr conflating with rho.

    - `diffusive_corr`: pooled realized correlation on the jump-scrubbed
      ("continuous-only") series -- should track `rho`, largely unpolluted
      by co-jump timing.
    - `jump_corr`: pooled realized correlation of the binary jump-timing
      indicators across assets -- should track `jump_coupling` (assets that
      tend to jump at the same time will have correlated indicator series),
      largely unpolluted by diffusive `rho`.

    x: (n_paths, n_steps, n_assets). Returns (diffusive_corr, jump_corr).
    """
    flag_fn = {"bns": _bns_jump_flags, "naive": _naive_threshold_jump_flags}[method]

    arr = x.numpy()  # (P, T, N)
    p, t, n = arr.shape

    continuous = arr.copy()
    jump_flags = np.zeros_like(arr, dtype=bool)

    for path in range(p):
        for asset in range(n):
            series = arr[path, :, asset]
            flags = flag_fn(series, **kwargs)
            jump_flags[path, :, asset] = flags
            continuous[path, flags, asset] = 0.0  # scrub jump-flagged returns

    diffusive_corr = realized_corr_estimate(torch.tensor(continuous))

    pooled_flags = jump_flags.reshape(-1, n).astype(float)
    if pooled_flags.std(axis=0).min() == 0.0:
        # a column with zero variance (no jumps ever flagged, or always flagged)
        # makes correlation undefined -- fall back to 0 co-occurrence for that pair
        jump_corr = 0.0
    else:
        jc = np.corrcoef(pooled_flags, rowvar=False)
        jump_corr = float(jc[np.triu_indices(n, k=1)].mean())

    return diffusive_corr, jump_corr
