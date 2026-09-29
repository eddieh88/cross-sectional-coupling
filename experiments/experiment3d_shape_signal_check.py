"""
Experiment 3d -- pre-registration check for Phase B's core hypothesis, BEFORE
building any architecture.

Phase B's proposed design (results/experiment3_findings.md) rests on a
specific mathematical claim: a shared diffusive draw moves every asset in a
FIXED ratio proportional to its own sigma_i (rank-1: diffusion_i = sigma_i *
sqrt(rho) * z_common + idio), while a common jump draws an INDEPENDENT size
per asset even when timing is shared. If that shape difference isn't
actually visible at realistic sample sizes, Phase B would fail for the same
reason a two-gate architecture might have -- not because the idea is wrong,
but because the signal doesn't survive real event counts (~3-4 jumps per
path, per Experiment 3 Phase A.5).

This test uses the generator's OWN ground truth (which step is a real
common jump, not a detector's guess) so it isolates the shape-signal
question from jump-detection noise entirely -- the cleanest, cheapest
possible test of the claim, no training required.

Method: for each event (a full cross-sectional return vector at one
timestep), regress it against the KNOWN per-asset sigma_i vector. Diffusive
events should fit well (high R^2, proportional to sigma) -- jump events
should not (independent per-asset sizes, no reason to correlate with
sigma_i). Requires heterogeneous sigma_i across assets -- with identical
sigma (used everywhere else in this project so far) the regression is
degenerate (zero variance in the predictor).

"Diffusive" comparison events are the largest |z_common| steps NOT flagged
as a jump, matched in COUNT to the number of real jump events in that same
path -- a fair, realistic-sample-size comparison, not an unlimited pool.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jump_diffusion_generator import JumpDiffusionGenerator

N_ASSETS = 20
SIGMA = np.linspace(0.10, 0.40, N_ASSETS)  # heterogeneous vols -- required for this test to be non-degenerate
N_PATHS = 50
N_STEPS = 300
DT = 1 / 252
LAMBDA_COMMON = 2.0
LAMBDA_IDIO = 1.0
RHO_GRID = [0.3, 0.6, 0.9]
SEED = 7


def r_squared(y: np.ndarray, x: np.ndarray) -> float:
    """R^2 of y ~ a*x + b via OLS."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def main():
    for rho in RHO_GRID:
        gen = JumpDiffusionGenerator(
            n_assets=N_ASSETS, sigma=SIGMA, rho=rho,
            lambda_common=LAMBDA_COMMON, lambda_idio=LAMBDA_IDIO,
        )
        log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED, return_diagnostics=True)
        rets = np.diff(log_paths, axis=1)  # (P, T, N)
        z_common = diag["z_common"]  # (P, T)
        jump_fired = diag["common_jump_fired"]  # (P, T)

        jump_r2 = []
        diffusive_r2 = []
        per_path_example = None

        for p in range(N_PATHS):
            jump_steps = np.where(jump_fired[p])[0]
            if len(jump_steps) == 0:
                continue

            non_jump_steps = np.where(~jump_fired[p])[0]
            # match count to this path's actual jump count -- realistic sample size, not unlimited
            k = len(jump_steps)
            top_diffusive = non_jump_steps[np.argsort(-np.abs(z_common[p, non_jump_steps]))[:k]]

            path_jump_r2 = [r_squared(rets[p, t, :], SIGMA) for t in jump_steps]
            path_diff_r2 = [r_squared(rets[p, t, :], SIGMA) for t in top_diffusive]

            jump_r2.extend(path_jump_r2)
            diffusive_r2.extend(path_diff_r2)

            if per_path_example is None and k >= 2:
                per_path_example = (p, k, path_jump_r2, path_diff_r2)

        jump_r2 = np.array(jump_r2)
        diffusive_r2 = np.array(diffusive_r2)

        print(f"\n=== rho={rho} ===")
        print(f"total events: {len(jump_r2)} jump, {len(diffusive_r2)} matched diffusive "
              f"(from {N_PATHS} paths, ~{len(jump_r2)/N_PATHS:.1f} jumps/path)")
        print(f"jump R^2:      mean={jump_r2.mean():.3f}  median={np.median(jump_r2):.3f}  "
              f"[{jump_r2.min():.3f}, {jump_r2.max():.3f}]")
        print(f"diffusive R^2: mean={diffusive_r2.mean():.3f}  median={np.median(diffusive_r2):.3f}  "
              f"[{diffusive_r2.min():.3f}, {diffusive_r2.max():.3f}]")

        # separation check: what fraction of jump events have R^2 above the median diffusive R^2?
        overlap = (jump_r2 > np.median(diffusive_r2)).mean()
        print(f"fraction of JUMP events with R^2 > median diffusive R^2 (should be low if separable): {overlap:.3f}")

        if per_path_example:
            p, k, pj, pd = per_path_example
            print(f"\nSingle-path illustration (path {p}, {k} jump events -- the realistic sample size a real path gives you):")
            print(f"  jump event R^2s:      {[round(v,3) for v in pj]}")
            print(f"  diffusive event R^2s: {[round(v,3) for v in pd]}")


if __name__ == "__main__":
    main()
