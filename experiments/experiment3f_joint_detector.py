"""
Experiment 3f -- one more cheap classical check before scoping a Phase B
detector: can a JOINT (not per-asset) statistic recover the ~45% of true
jump events that Experiment 3e's per-asset-co-occurrence detector missed?

Diagnosis from 3e: that detector requires several assets to EACH
individually cross their own strict threshold. Since jump sizes are drawn
independently per asset, a real common jump with a couple of small
individual draws slips through entirely -- even though the aggregate
pattern across the whole cross-section may still be visibly abnormal.

This test pools weak individual signals into one joint number per
timestep, no asset-by-asset gate required:

    Z[t, i] = r[t, i] / sigma_hat_i          (per-asset standardized move)
    J[t]    = sum_i Z[t, i]^2                (joint "total abnormality" energy)

A coordinated event -- diffusive or jump -- inflates MANY Z[t, i]
simultaneously, even if none of them individually looks extreme, so J[t]
rises even when no single asset would have tripped a per-asset test. J[t]
is then compared against a ROBUST per-path threshold (median + k * MAD of
J itself over that path), not a theoretical chi-square reference -- honest
about being a practical detector, not a rigorously calibrated asymptotic
test (the exact null distribution of J depends on the cross-sectional
correlation structure, i.e. on rho itself, which isn't known in practice).

If this closes most of the detection gap, that's a fourth negative result
and a complete, fully classical two-stage pipeline (joint detect -> R^2
shape classify). If it doesn't, that's real evidence for scoping Phase B
specifically as a detector.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jump_diffusion_generator import JumpDiffusionGenerator
from src.baseline import _bipower_variation

N_ASSETS = 20
SIGMA = np.linspace(0.10, 0.40, N_ASSETS)
N_PATHS = 50
N_STEPS = 300
DT = 1 / 252
LAMBDA_COMMON = 2.0
LAMBDA_IDIO = 1.0
RHO_GRID = [0.3, 0.6, 0.9]
SEED = 7

K_GRID = [3.0, 5.0, 8.0]  # robust-threshold strictness: median(J) + k * MAD(J)
R2_CLASSIFY_THRESHOLD = 0.5


def r_squared(y: np.ndarray, x: np.ndarray) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def estimate_sigma_per_asset(rets_path: np.ndarray) -> np.ndarray:
    n = rets_path.shape[1]
    return np.array([np.sqrt(_bipower_variation(rets_path[:, i])) for i in range(n)])


def robust_threshold(values: np.ndarray, k: float) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826  # normal-consistency scaling
    return med + k * mad


def main():
    for rho in RHO_GRID:
        gen = JumpDiffusionGenerator(n_assets=N_ASSETS, sigma=SIGMA, rho=rho,
                                      lambda_common=LAMBDA_COMMON, lambda_idio=LAMBDA_IDIO)
        log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED, return_diagnostics=True)
        rets = np.diff(log_paths, axis=1)
        jump_fired = diag["common_jump_fired"]
        total_true_jumps = int(jump_fired.sum())

        print(f"\n=== rho={rho}  (total true jump timesteps in panel: {total_true_jumps}) ===")

        for k in K_GRID:
            pipeline_rows = []  # (true_label, predicted_label)

            for p in range(N_PATHS):
                sigma_hat = estimate_sigma_per_asset(rets[p])
                z = rets[p] / sigma_hat[None, :]        # (T, N) standardized
                j_stat = (z ** 2).sum(axis=1)           # (T,) joint energy per timestep
                thresh = robust_threshold(j_stat, k)
                candidate_steps = np.where(j_stat > thresh)[0]

                for t in candidate_steps:
                    true_label = "jump" if jump_fired[p, t] else "diffusive"
                    r2 = r_squared(rets[p, t, :], sigma_hat)
                    predicted_label = "diffusive" if r2 > R2_CLASSIFY_THRESHOLD else "jump"
                    pipeline_rows.append((true_label, predicted_label))

            n_candidates = len(pipeline_rows)
            if n_candidates == 0:
                print(f"  k={k}: no candidates detected")
                continue

            true_labels = np.array([r[0] for r in pipeline_rows])
            pred_labels = np.array([r[1] for r in pipeline_rows])
            n_true_jump = int((true_labels == "jump").sum())
            n_true_diff = int((true_labels == "diffusive").sum())
            tp = int(((true_labels == "jump") & (pred_labels == "jump")).sum())
            fp = int(((true_labels == "diffusive") & (pred_labels == "jump")).sum())
            fn = int(((true_labels == "jump") & (pred_labels == "diffusive")).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            classifier_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            detector_recall = n_true_jump / total_true_jumps if total_true_jumps > 0 else float("nan")

            print(f"  k={k}: {n_candidates} candidates ({n_true_jump} true jump, {n_true_diff} true diffusive)  "
                  f"detector_recall={detector_recall:.3f}  precision={precision:.3f}  "
                  f"classifier_recall_among_candidates={classifier_recall:.3f}")


if __name__ == "__main__":
    main()
