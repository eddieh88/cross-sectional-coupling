"""
Experiment 2a -- Sample efficiency: alpha vs. the trivial correlation baseline.

Per docs/research_spec.md Sec.4 Experiment 2: "does alpha recover rho any
better, faster (fewer samples), or more robustly than the two-line
correlation calculation?" This script answers the "fewer samples" half.

Both estimators see EXACTLY the same generated data at each (sample_size,
rho) point -- same tensor, no separate RNG draws -- so any difference is
about the estimator, not the data.

"medium" reuses Experiment 1's already-trained alpha values (same n_paths,
n_steps, seed) instead of retraining, since that grid already exists.
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
EPOCHS = 250
SEED = 0

# (label, n_paths, n_steps, retrain_model)
SAMPLE_SIZES = [
    ("small", 5, 50, True),
    ("medium", 30, 300, False),  # reuse experiment1_recovery.csv, don't retrain
    ("large", 50, 500, True),
]

EXP1_CSV = Path(__file__).resolve().parent.parent / "results" / "experiment1_recovery.csv"


def load_exp1_alphas():
    with open(EXP1_CSV) as f:
        rows = list(csv.DictReader(f))
    return {round(float(r["rho"]), 4): float(r["alpha"]) for r in rows}


def main():
    exp1_alphas = load_exp1_alphas()
    results = []  # (sample_size_label, n_paths, n_steps, rho, alpha_or_none, baseline_corr)

    grid_t0 = time.time()
    for label, n_paths, n_steps, retrain in SAMPLE_SIZES:
        print(f"\n=== sample size '{label}': {n_paths} paths x {n_steps} steps "
              f"(total obs/asset = {n_paths * n_steps}) ===", flush=True)

        for i, rho in enumerate(RHO_GRID, start=1):
            tag = f"[{label} {i}/{len(RHO_GRID)} rho={rho:.2f}]"
            x = make_panel(rho=rho, n_assets=N_ASSETS, n_paths=n_paths, n_steps=n_steps, seed=SEED)
            baseline = realized_corr_estimate(x)

            if retrain:
                out = train_one_run(x, epochs=EPOCHS, seed=SEED, log_every=50, tag=tag)
                alpha = out["final_alpha"]
            else:
                alpha = exp1_alphas[round(rho, 4)]
                print(f"{tag} reusing Experiment 1 alpha={alpha:.4f} (not retrained)", flush=True)

            print(f"{tag} baseline_corr={baseline:.4f}  alpha={alpha:.4f}  "
                  f"(grid elapsed {time.time() - grid_t0:.1f}s)", flush=True)
            results.append((label, n_paths, n_steps, rho, alpha, baseline))

    # --- summary: how well does each estimator track true rho, per sample size ---
    print("\n=== Summary: Pearson corr(estimator, true rho) by sample size ===")
    summary_rows = []
    rhos_arr = np.array(RHO_GRID)
    for label, n_paths, n_steps, _ in SAMPLE_SIZES:
        rows = [r for r in results if r[0] == label]
        alphas = np.array([r[4] for r in rows])
        baselines = np.array([r[5] for r in rows])
        alpha_corr = np.corrcoef(rhos_arr, alphas)[0, 1]
        baseline_corr_r = np.corrcoef(rhos_arr, baselines)[0, 1]
        print(f"{label:8s} (n={n_paths}x{n_steps:4d}): "
              f"alpha_r={alpha_corr:.4f}  baseline_r={baseline_corr_r:.4f}  "
              f"(alpha {'beats' if alpha_corr > baseline_corr_r else 'does NOT beat'} baseline)")
        summary_rows.append((label, n_paths, n_steps, alpha_corr, baseline_corr_r))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "experiment2a_sample_efficiency.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_size", "n_paths", "n_steps", "rho", "alpha", "baseline_corr"])
        writer.writerows(results)

    with open(out_dir / "experiment2a_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_size", "n_paths", "n_steps", "alpha_pearson_r", "baseline_pearson_r"])
        writer.writerows(summary_rows)

    print(f"\nSaved results to {out_dir}/experiment2a_sample_efficiency.csv and experiment2a_summary.csv")


if __name__ == "__main__":
    main()
