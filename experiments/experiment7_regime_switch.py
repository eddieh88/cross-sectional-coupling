"""
Experiment 7 -- the literal "time-varying rho" test named in the original
spec's Experiment 3 list and never run until now: "regime-switching
coupling within a single simulated series (e.g. low rho, then a shift to
high rho mid-series, mimicking a correlation breakdown). Does a
rolling-window version of alpha track the shift, with what lag?"

Motivated directly by Experiment 6's real-data finding: rolling 30-day
correlation on real crypto swung 0.31->0.89 across a 3-year window -- not a
minor wobble, a real falsification of the fixed-rho assumption used
throughout Exp1-4b. This tests the cheap classical answer first (rolling
realized correlation) against KNOWN ground truth (the regime switch is
built into the generator, per jump_diffusion_generator.py's `rho_2` /
`regime_switch_frac`), before considering anything adaptive/learned.

Question: across a range of window lengths, what's the bias/lag tradeoff?
Short windows track the switch fast but are noisy within each stable
regime; long windows are smooth but lag the switch. Is there a window
length that does BOTH well, or is this a genuine tradeoff a fixed-window
classical estimator can't escape -- which would be the first well-motivated,
non-speculative case for something adaptive (a model whose "effective
window" itself adapts to whether a regime change seems to be underway).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jump_diffusion_generator import JumpDiffusionGenerator

N_ASSETS = 20
N_PATHS = 30
N_STEPS = 2000
DT = 1 / 252
RHO_LOW = 0.2
RHO_HIGH = 0.85  # roughly matches the real swing magnitude observed in Exp6 (0.31 -> 0.89)
SWITCH_FRAC = 0.5
SEED = 0

WINDOW_GRID = [5, 10, 20, 30, 60, 90]


def rolling_corr(rets: np.ndarray, window: int) -> np.ndarray:
    """rets: (T, N) for one path. Returns (T,) mean off-diagonal rolling correlation, NaN before the first full window."""
    T, N = rets.shape
    out = np.full(T, np.nan)
    iu = np.triu_indices(N, 1)
    for t in range(window - 1, T):
        block = rets[t - window + 1: t + 1]
        c = np.corrcoef(block, rowvar=False)
        out[t] = c[iu].mean()
    return out


def detection_lag(estimate: np.ndarray, switch_step: int, rho_low: float, rho_high: float) -> float:
    """Steps after the true switch until the rolling estimate first crosses the midpoint. NaN if it never does within the window shown."""
    midpoint = (rho_low + rho_high) / 2
    for t in range(switch_step, len(estimate)):
        if not np.isnan(estimate[t]) and estimate[t] >= midpoint:
            return t - switch_step
    return float("nan")


def main():
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO_LOW, rho_2=RHO_HIGH,
                                  regime_switch_frac=SWITCH_FRAC, lambda_common=2.0, lambda_idio=1.0)
    log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED, return_diagnostics=True)
    rets = np.diff(log_paths, axis=1)  # (P, T, N)
    switch_step = int(N_STEPS * SWITCH_FRAC)
    print(f"Regime switch: rho={RHO_LOW} -> rho={RHO_HIGH} at step {switch_step} of {N_STEPS}\n")

    for window in WINDOW_GRID:
        lags, pre_stds, post_stds, pre_biases, post_biases = [], [], [], [], []
        for p in range(N_PATHS):
            est = rolling_corr(rets[p], window)
            lag = detection_lag(est, switch_step, RHO_LOW, RHO_HIGH)
            lags.append(lag)

            # stability/bias within each stable regime, away from the switch transition itself
            pre_region = est[window: switch_step - window]      # well within the low-rho regime
            post_region = est[switch_step + window * 3:]        # well within the high-rho regime, past transition
            pre_region = pre_region[~np.isnan(pre_region)]
            post_region = post_region[~np.isnan(post_region)]
            if len(pre_region):
                pre_stds.append(pre_region.std())
                pre_biases.append(pre_region.mean())  # compare to true realized corr at that rho, not literal rho
            if len(post_region):
                post_stds.append(post_region.std())
                post_biases.append(post_region.mean())

        lags = np.array(lags, dtype=float)
        valid_lags = lags[~np.isnan(lags)]
        print(f"window={window:3d}: detection lag mean={np.nanmean(lags):.1f} steps "
              f"(median={np.nanmedian(lags):.1f}, {len(valid_lags)}/{N_PATHS} paths detected at all)  |  "
              f"within-regime std: pre={np.mean(pre_stds):.4f} post={np.mean(post_stds):.4f}  |  "
              f"within-regime mean corr: pre={np.mean(pre_biases):.3f} post={np.mean(post_biases):.3f}")


def ewma_corr(rets: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA (RiskMetrics-style) rolling correlation. rets: (T, N). Returns (T,) mean off-diagonal EWMA correlation."""
    T, N = rets.shape
    decay = 0.5 ** (1.0 / halflife)
    cov = np.cov(rets[:5], rowvar=False) if T >= 5 else np.eye(N) * 1e-8  # warm start
    out = np.full(T, np.nan)
    iu = np.triu_indices(N, 1)
    for t in range(T):
        x = rets[t]
        cov = decay * cov + (1 - decay) * np.outer(x, x)
        std = np.sqrt(np.diag(cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(std, std)
        out[t] = np.nanmean(corr[iu])
    return out


def main_ewma():
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO_LOW, rho_2=RHO_HIGH,
                                  regime_switch_frac=SWITCH_FRAC, lambda_common=2.0, lambda_idio=1.0)
    log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED, return_diagnostics=True)
    rets = np.diff(log_paths, axis=1)
    switch_step = int(N_STEPS * SWITCH_FRAC)

    print(f"\n=== EWMA correlation (RiskMetrics-style), same ground truth ===")
    for halflife in [5, 10, 20, 30, 60, 90]:
        lags, pre_stds, post_stds, pre_biases, post_biases = [], [], [], [], []
        for p in range(N_PATHS):
            est = ewma_corr(rets[p], halflife)
            lag = detection_lag(est, switch_step, RHO_LOW, RHO_HIGH)
            lags.append(lag)
            pre_region = est[int(halflife * 3): switch_step]
            post_region = est[switch_step + int(halflife * 3):]
            if len(pre_region):
                pre_stds.append(np.std(pre_region)); pre_biases.append(np.mean(pre_region))
            if len(post_region):
                post_stds.append(np.std(post_region)); post_biases.append(np.mean(post_region))
        lags = np.array(lags, dtype=float)
        print(f"halflife={halflife:3d}: detection lag mean={np.nanmean(lags):.1f} steps "
              f"(median={np.nanmedian(lags):.1f})  |  "
              f"within-regime std: pre={np.mean(pre_stds):.4f} post={np.mean(post_stds):.4f}  |  "
              f"within-regime mean corr: pre={np.mean(pre_biases):.3f} post={np.mean(post_biases):.3f}")


if __name__ == "__main__":
    main()
    main_ewma()


def per_step_coupling_stat(rets: np.ndarray, winsor_mult: float = 8.0) -> np.ndarray:
    """Per-timestep instantaneous pairwise-correlation proxy: mean_{i<j} standardized(r_i)*standardized(r_j).
    Very noisy per single step (single-sample correlation estimate) but its MEAN over time shifts
    exactly when rho shifts -- the right kind of series to feed a CUSUM mean-shift detector.

    Winsorized against jump contamination: a single common JUMP event (this generator still
    has active jump processes, per lambda_common/lambda_idio) produces one huge, transient
    co-movement spike that is NOT the persistent diffusive regime shift this statistic is meant
    to track -- the exact jump-vs-diffusive contamination problem Experiment 3 solved, recurring
    here in a new statistic that was not built jump-robust in the first pass. Fixed the same way:
    clip to a robust (median + winsor_mult*MAD) band before it can dominate the CUSUM accumulator."""
    T, N = rets.shape
    sigma_hat = rets.std(axis=0, keepdims=True)
    z = rets / sigma_hat
    iu = np.triu_indices(N, 1)
    prod = z[:, iu[0]] * z[:, iu[1]]  # (T, n_pairs)
    raw = prod.mean(axis=1)  # (T,)
    med = np.median(raw)
    mad = np.median(np.abs(raw - med)) * 1.4826
    return np.clip(raw, med - winsor_mult * mad, med + winsor_mult * mad)


def cusum_detect_and_estimate(rets: np.ndarray, burn_in: int, k_mult: float = 0.5, h_mult: float = 5.0):
    """Standard one-sided CUSUM (Page's test) for an INCREASE in the coupling statistic's mean.
    Baseline (mu0, sigma0) estimated from the first `burn_in` steps. Returns (detect_step, post_break_estimate)
    where post_break_estimate[t] is a rolling correlation using only data from the detected break onward
    (an expanding window, growing lag-free after detection since there's no need to smooth through the break)."""
    s = per_step_coupling_stat(rets)
    mu0, sigma0 = s[:burn_in].mean(), s[:burn_in].std()
    k = k_mult * sigma0
    h = h_mult * sigma0

    cusum = 0.0
    detect_step = None
    for t in range(burn_in, len(s)):
        cusum = max(0.0, cusum + (s[t] - mu0 - k))
        if cusum > h:
            detect_step = t
            break

    T, N = rets.shape
    iu = np.triu_indices(N, 1)
    post_estimate = np.full(T, np.nan)
    if detect_step is not None:
        for t in range(detect_step, T):
            block = rets[detect_step:t + 1]
            if len(block) < 2:
                continue
            c = np.corrcoef(block, rowvar=False)
            post_estimate[t] = c[iu].mean()
    return detect_step, post_estimate


def main_cusum():
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO_LOW, rho_2=RHO_HIGH,
                                  regime_switch_frac=SWITCH_FRAC, lambda_common=2.0, lambda_idio=1.0)
    log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED, return_diagnostics=True)
    rets = np.diff(log_paths, axis=1)
    switch_step = int(N_STEPS * SWITCH_FRAC)
    burn_in = 100  # well within the pre-break regime, enough to estimate a stable baseline

    print(f"\n=== CUSUM change-point detector + post-break expanding window ===")
    lags, noise_20, noise_100, noise_full = [], [], [], []
    n_detected = 0
    for p in range(N_PATHS):
        detect_step, post_est = cusum_detect_and_estimate(rets[p], burn_in)
        if detect_step is None:
            continue
        n_detected += 1
        lags.append(detect_step - switch_step)
        # noise at different amounts of post-detection data accumulated (expanding window)
        for horizon, bucket in [(20, noise_20), (100, noise_100)]:
            idx = detect_step + horizon
            if idx < N_STEPS:
                window_vals = post_est[detect_step + horizon // 2: idx]
                window_vals = window_vals[~np.isnan(window_vals)]
                if len(window_vals):
                    bucket.append(window_vals.std())
        tail = post_est[N_STEPS - 200:]
        tail = tail[~np.isnan(tail)]
        if len(tail):
            noise_full.append(tail.std())

    lags = np.array(lags, dtype=float)
    print(f"detected in {n_detected}/{N_PATHS} paths")
    print(f"detection lag: mean={lags.mean():.1f} steps (median={np.median(lags):.1f}, "
          f"range [{lags.min():.0f}, {lags.max():.0f}])")
    print(f"post-break estimate noise (std): "
          f"~20 steps after detection={np.mean(noise_20):.4f}  "
          f"~100 steps after detection={np.mean(noise_100):.4f}  "
          f"last 200 steps of series (fully expanded)={np.mean(noise_full):.4f}")


if __name__ == "__main__":
    main()
    main_ewma()
    main_cusum()
