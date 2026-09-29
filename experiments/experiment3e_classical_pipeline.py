"""
Experiment 3e -- stress-test a fully classical two-stage pipeline (Phase
A.6) before conceding anything to a trained model.

Experiment 3d's shape-signal check (regressing a return vector against
known sigma_i to tell jump events from diffusive events) left two things
open, per review:

1. It used TRUE sigma_i -- best case. Real data requires an ESTIMATED
   sigma. Does the R^2 separation survive using a jump-robust estimate
   (bipower variation) instead of ground truth?
2. It used ground-truth jump timing to select which events to test. A full
   classical pipeline needs its OWN detector too -- Phase A's calibrated
   BNS test can fill that role (flag candidate coordinated-move timesteps
   per asset; a cross-sectional "event" is a timestep where enough assets
   get flagged at once), followed by the R^2-vs-sigma regression as a
   threshold CLASSIFIER (no training) to decide jump vs diffusive for each
   flagged candidate.

If this fully classical pipeline (BNS detect -> R^2-threshold classify,
estimated sigma throughout) works cleanly, that's a fourth negative result
for any NN, and a complete, useful classical answer to Experiment 3's
question in its own right -- independent of whether Phase B ever gets built.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jump_diffusion_generator import JumpDiffusionGenerator
from src.baseline import _bipower_variation, _bns_jump_flags

N_ASSETS = 20
SIGMA = np.linspace(0.10, 0.40, N_ASSETS)
N_PATHS = 50
N_STEPS = 300
DT = 1 / 252
LAMBDA_COMMON = 2.0
LAMBDA_IDIO = 1.0
RHO_GRID = [0.3, 0.6, 0.9]
SEED = 7

CANDIDATE_FRAC_THRESHOLD = 0.1  # >=10% of assets individually BNS-flagged at the same t => candidate event
R2_CLASSIFY_THRESHOLD = 0.5     # predict "diffusive" if R^2 > this, else "jump"


def r_squared(y: np.ndarray, x: np.ndarray) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def estimate_sigma_per_asset(rets_path: np.ndarray) -> np.ndarray:
    """Jump-robust per-asset scale estimate from one path's own return history."""
    n = rets_path.shape[1]
    return np.array([np.sqrt(_bipower_variation(rets_path[:, i])) for i in range(n)])


def main():
    for rho in RHO_GRID:
        gen = JumpDiffusionGenerator(n_assets=N_ASSETS, sigma=SIGMA, rho=rho,
                                      lambda_common=LAMBDA_COMMON, lambda_idio=LAMBDA_IDIO)
        log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED, return_diagnostics=True)
        rets = np.diff(log_paths, axis=1)
        z_common = diag["z_common"]
        jump_fired = diag["common_jump_fired"]

        # --- Part 1: matched-comparison R^2 separation, true sigma vs. estimated sigma ---
        jump_r2_true, jump_r2_est = [], []
        diff_r2_true, diff_r2_est = [], []

        # --- Part 2: fully classical pipeline (BNS-detected candidates, estimated sigma) ---
        pipeline_rows = []  # (true_label, predicted_label)

        for p in range(N_PATHS):
            sigma_hat = estimate_sigma_per_asset(rets[p])

            jump_steps = np.where(jump_fired[p])[0]
            if len(jump_steps) > 0:
                non_jump_steps = np.where(~jump_fired[p])[0]
                k = len(jump_steps)
                top_diffusive = non_jump_steps[np.argsort(-np.abs(z_common[p, non_jump_steps]))[:k]]

                for t in jump_steps:
                    jump_r2_true.append(r_squared(rets[p, t, :], SIGMA))
                    jump_r2_est.append(r_squared(rets[p, t, :], sigma_hat))
                for t in top_diffusive:
                    diff_r2_true.append(r_squared(rets[p, t, :], SIGMA))
                    diff_r2_est.append(r_squared(rets[p, t, :], sigma_hat))

            # Part 2: per-asset BNS flags -> cross-sectional candidate events
            asset_flags = np.stack([_bns_jump_flags(rets[p, :, i]) for i in range(N_ASSETS)], axis=1)  # (T, N)
            candidate_steps = np.where(asset_flags.mean(axis=1) >= CANDIDATE_FRAC_THRESHOLD)[0]

            for t in candidate_steps:
                true_label = "jump" if jump_fired[p, t] else "diffusive"
                r2 = r_squared(rets[p, t, :], sigma_hat)
                predicted_label = "diffusive" if r2 > R2_CLASSIFY_THRESHOLD else "jump"
                pipeline_rows.append((true_label, predicted_label))

        jump_r2_true, jump_r2_est = np.array(jump_r2_true), np.array(jump_r2_est)
        diff_r2_true, diff_r2_est = np.array(diff_r2_true), np.array(diff_r2_est)

        print(f"\n=== rho={rho} ===")
        print("--- Part 1: does R^2 separation survive ESTIMATED (bipower-variation) sigma? ---")
        print(f"jump R^2:      true_sigma mean={jump_r2_true.mean():.3f}   est_sigma mean={jump_r2_est.mean():.3f}")
        print(f"diffusive R^2: true_sigma mean={diff_r2_true.mean():.3f}   est_sigma mean={diff_r2_est.mean():.3f}")
        overlap_true = (jump_r2_true > np.median(diff_r2_true)).mean()
        overlap_est = (jump_r2_est > np.median(diff_r2_est)).mean()
        print(f"overlap (frac jump R^2 > median diffusive R^2): true_sigma={overlap_true:.3f}   est_sigma={overlap_est:.3f}")

        print("--- Part 2: full classical pipeline (BNS detect -> R^2-threshold classify, estimated sigma) ---")
        total_true_jumps_in_panel = int(jump_fired.sum())
        n_candidates = len(pipeline_rows)
        print(f"total true jump timesteps in the whole panel: {total_true_jumps_in_panel}")
        if n_candidates == 0:
            print("no candidate events detected -- detector recall is 0, pipeline result undefined")
            continue
        true_labels = np.array([r[0] for r in pipeline_rows])
        pred_labels = np.array([r[1] for r in pipeline_rows])
        accuracy = (true_labels == pred_labels).mean()
        n_true_jump = int((true_labels == "jump").sum())
        n_true_diff = int((true_labels == "diffusive").sum())
        tp = int(((true_labels == "jump") & (pred_labels == "jump")).sum())
        fp = int(((true_labels == "diffusive") & (pred_labels == "jump")).sum())
        fn = int(((true_labels == "jump") & (pred_labels == "diffusive")).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        classifier_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        detector_recall = n_true_jump / total_true_jumps_in_panel if total_true_jumps_in_panel > 0 else float("nan")
        print(f"{n_candidates} candidate events detected ({n_true_jump} true jump, {n_true_diff} true diffusive)")
        print(f"DETECTOR recall (true jump timesteps that became candidates at all): {detector_recall:.3f}  <-- the real bottleneck")
        print(f"overall accuracy={accuracy:.3f}   jump-class precision={precision:.3f}   "
              f"classifier recall AMONG CANDIDATES={classifier_recall:.3f}")


if __name__ == "__main__":
    main()
