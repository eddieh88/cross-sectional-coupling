"""
Experiment 4a -- stress-test the classical pipeline (Experiments 3d/3e/3f)
against a genuinely multi-factor common-diffusion structure, before
declaring "consolidate, don't build Phase B" a real conclusion.

Everything in 3d/3e/3f leans on one structural assumption: the diffusive
common component is exactly RANK-1 (every asset's common exposure is the
same shock scaled only by its own sigma_i), so return_i / sigma_i is
identical across assets for any diffusive event. That's precisely why
regressing a return vector against the known sigma_i vector cleanly
separates diffusive events (high R^2) from jump events (independent
per-asset sizes, low R^2). A result that only holds because of a
by-construction rank-1 assumption is a narrow correspondence, not a
general property of the method (spec Experiment 4).

`jump_diffusion_generator.py` now supports `factor_dispersion` (0 = the
original rank-1 case; >0 activates a second independent common factor with
heterogeneous, fixed per-asset loadings -- see its module docstring). This
sweeps factor_dispersion x rho and reruns the two load-bearing pieces of
the classical pipeline:

  1. The 3d shape-signal check (ground-truth jump timing, TRUE sigma_i) --
     does R^2 separation between jump and diffusive events survive as the
     common structure moves away from rank-1?
  2. The 3f joint detector (BNS-free joint statistic + robust threshold) --
     does detector recall/precision survive?

Jump structure is left untouched (jumps still fire across ALL assets
simultaneously via lambda_common, unaffected by factor_dispersion) so any
change in results is attributable specifically to the diffusive commonality
becoming multi-factor, not to some other confound.
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
DISPERSION_GRID = [0.0, 0.3, 0.6, 1.0]
SEED = 7

JOINT_K = 10.0  # fixed operating point from Experiment 3f (~99-100% recall, 65-76% precision at rho=0.3-0.6)


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
    mad = np.median(np.abs(values - med)) * 1.4826
    return med + k * mad


def main():
    for rho in RHO_GRID:
        print(f"\n=== rho={rho} ===")
        for disp in DISPERSION_GRID:
            gen = JumpDiffusionGenerator(n_assets=N_ASSETS, sigma=SIGMA, rho=rho,
                                          lambda_common=LAMBDA_COMMON, lambda_idio=LAMBDA_IDIO,
                                          factor_dispersion=disp)
            log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED,
                                            return_diagnostics=True)
            rets = np.diff(log_paths, axis=1)
            z_common = diag["z_common"]
            jump_fired = diag["common_jump_fired"]

            # --- Part 1: 3d-style shape check, TRUE sigma, ground-truth timing ---
            jump_r2, diff_r2 = [], []
            for p in range(N_PATHS):
                jump_steps = np.where(jump_fired[p])[0]
                if len(jump_steps) == 0:
                    continue
                non_jump_steps = np.where(~jump_fired[p])[0]
                k_match = len(jump_steps)
                top_diffusive = non_jump_steps[np.argsort(-np.abs(z_common[p, non_jump_steps]))[:k_match]]
                jump_r2.extend(r_squared(rets[p, t, :], SIGMA) for t in jump_steps)
                diff_r2.extend(r_squared(rets[p, t, :], SIGMA) for t in top_diffusive)
            jump_r2, diff_r2 = np.array(jump_r2), np.array(diff_r2)
            overlap = (jump_r2 > np.median(diff_r2)).mean() if len(diff_r2) else float("nan")

            # --- Part 2: 3f-style joint detector, estimated sigma, real detector ---
            total_true_jumps = int(jump_fired.sum())
            pipeline_rows = []
            for p in range(N_PATHS):
                sigma_hat = estimate_sigma_per_asset(rets[p])
                z = rets[p] / sigma_hat[None, :]
                j_stat = (z ** 2).sum(axis=1)
                thresh = robust_threshold(j_stat, JOINT_K)
                for t in np.where(j_stat > thresh)[0]:
                    true_label = "jump" if jump_fired[p, t] else "diffusive"
                    r2 = r_squared(rets[p, t, :], sigma_hat)
                    predicted_label = "diffusive" if r2 > 0.5 else "jump"
                    pipeline_rows.append((true_label, predicted_label))

            true_labels = np.array([r[0] for r in pipeline_rows])
            pred_labels = np.array([r[1] for r in pipeline_rows])
            n_true_jump = int((true_labels == "jump").sum())
            n_true_diff = int((true_labels == "diffusive").sum())
            tp = int(((true_labels == "jump") & (pred_labels == "jump")).sum())
            fp = int(((true_labels == "diffusive") & (pred_labels == "jump")).sum())
            fn = int(((true_labels == "jump") & (pred_labels == "diffusive")).sum())
            tn = int(((true_labels == "diffusive") & (pred_labels == "diffusive")).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            classifier_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # diffusive-class accuracy -- the metric that actually shows the collapse
            detector_recall = n_true_jump / total_true_jumps if total_true_jumps > 0 else float("nan")

            # R^2 distribution of the ACTUAL flagged population (estimated sigma), not
            # the matched-count comparison from Part 1 -- reconciles whether Part 1's
            # R^2 collapse actually shows up where it matters (the candidates the
            # pipeline has to classify).
            cand_r2 = []
            for p in range(N_PATHS):
                sigma_hat = estimate_sigma_per_asset(rets[p])
                z = rets[p] / sigma_hat[None, :]
                j_stat = (z ** 2).sum(axis=1)
                thresh = robust_threshold(j_stat, JOINT_K)
                for t in np.where(j_stat > thresh)[0]:
                    r2 = r_squared(rets[p, t, :], sigma_hat)
                    cand_r2.append((("jump" if jump_fired[p, t] else "diffusive"), r2))
            cand_jump_r2 = np.array([r for lab, r in cand_r2 if lab == "jump"])
            cand_diff_r2 = np.array([r for lab, r in cand_r2 if lab == "diffusive"])

            print(f"  dispersion={disp:.1f}: shape R^2 (matched, true sigma) jump={jump_r2.mean():.3f} "
                  f"diffusive={diff_r2.mean():.3f} overlap={overlap:.3f}")
            print(f"      candidates: n={len(pipeline_rows)} (true_jump={n_true_jump} true_diff={n_true_diff})  "
                  f"tp={tp} fp={fp} fn={fn} tn={tn}")
            print(f"      candidate R^2 (est sigma, actual flagged pop): jump_mean={cand_jump_r2.mean() if len(cand_jump_r2) else float('nan'):.3f} "
                  f"diffusive_mean={cand_diff_r2.mean() if len(cand_diff_r2) else float('nan'):.3f}")
            print(f"      pipeline(k={JOINT_K:.0f}) detector_recall={detector_recall:.3f} "
                  f"precision={precision:.3f} classifier_recall(jump)={classifier_recall:.3f} "
                  f"specificity(diffusive)={specificity:.3f}")


if __name__ == "__main__":
    main()
