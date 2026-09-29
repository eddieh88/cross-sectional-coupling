"""
Experiment 2b -- Robustness to jump contamination: alpha vs. the trivial
correlation baseline.

Per docs/research_spec.md Sec.4 Experiment 2: "...more robustly (under jump
contamination)...?" This script answers that half. `jump_coupling` (governed
by lambda_common/lambda_idio) is held fixed across conditions -- only
`jump_std` (jump size noise) changes -- so this isolates "does bigger jump
noise degrade rho-recovery" from "does the jump/diffusive mix change."

Both estimators see exactly the same generated data at each (condition, rho)
point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import time
import numpy as np

from src.data import make_panel
from src.train import train_one_run
from src.baseline import realized_corr_estimate

RHO_GRID = [0.0, 0.3, 0.5, 0.7, 1.0]
N_ASSETS = 10
N_PATHS = 30
N_STEPS = 300
EPOCHS = 250
SEED = 0

# (label, jump_std) -- generator default jump_std is 0.05
JUMP_CONDITIONS = [
    ("baseline_jumps", 0.05),
    ("high_jump_noise", 0.15),
]


def main():
    results = []  # (condition_label, jump_std, rho, alpha, baseline_corr)
    grid_t0 = time.time()

    for label, jump_std in JUMP_CONDITIONS:
        print(f"\n=== jump condition '{label}': jump_std={jump_std} ===", flush=True)
        for i, rho in enumerate(RHO_GRID, start=1):
            tag = f"[{label} {i}/{len(RHO_GRID)} rho={rho:.2f}]"
            x = make_panel(
                rho=rho, n_assets=N_ASSETS, n_paths=N_PATHS, n_steps=N_STEPS,
                jump_std=jump_std, seed=SEED,
            )
            baseline = realized_corr_estimate(x)
            out = train_one_run(x, epochs=EPOCHS, seed=SEED, log_every=50, tag=tag)
            alpha = out["final_alpha"]

            print(f"{tag} baseline_corr={baseline:.4f}  alpha={alpha:.4f}  "
                  f"(grid elapsed {time.time() - grid_t0:.1f}s)", flush=True)
            results.append((label, jump_std, rho, alpha, baseline))

    print("\n=== Summary: Pearson corr(estimator, true rho) by jump condition ===")
    summary_rows = []
    rhos_arr = np.array(RHO_GRID)
    for label, jump_std in JUMP_CONDITIONS:
        rows = [r for r in results if r[0] == label]
        alphas = np.array([r[3] for r in rows])
        baselines = np.array([r[4] for r in rows])
        alpha_corr = np.corrcoef(rhos_arr, alphas)[0, 1]
        baseline_corr_r = np.corrcoef(rhos_arr, baselines)[0, 1]
        print(f"{label:16s} (jump_std={jump_std}): "
              f"alpha_r={alpha_corr:.4f}  baseline_r={baseline_corr_r:.4f}  "
              f"(alpha {'beats' if alpha_corr > baseline_corr_r else 'does NOT beat'} baseline)")
        summary_rows.append((label, jump_std, alpha_corr, baseline_corr_r))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "experiment2b_jump_robustness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "jump_std", "rho", "alpha", "baseline_corr"])
        writer.writerows(results)

    with open(out_dir / "experiment2b_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "jump_std", "alpha_pearson_r", "baseline_pearson_r"])
        writer.writerows(summary_rows)

    print(f"\nSaved results to {out_dir}/experiment2b_jump_robustness.csv and experiment2b_summary.csv")


if __name__ == "__main__":
    main()
