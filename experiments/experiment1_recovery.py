"""
Experiment 1 -- Recovery test (sanity check, not the finding).

Train the gated model on synthetic panels at a grid of known `rho` values,
holding `jump_coupling` fixed. Read off sigmoid(w) after training and check
whether it tracks true rho.

Task is contemporaneous leave-one-out reconstruction (see src/model.py
docstring). Under that task the temporal pathway carries no real signal at
any rho (this generator has no serial autocorrelation), while the
cross-sectional pathway's achievable loss strictly improves as rho
increases -- so alpha is expected to rise toward 1 fairly quickly rather
than trace a straight line across [0, 1]. Check for MONOTONICITY, not
linearity.

A pass here is necessary but not interesting on its own -- it confirms the
plumbing isn't broken, not that alpha is a useful signal (that's Experiment
2, comparing against the trivial realized-correlation baseline).

Failure mode to watch for: alpha saturates to 0/1 regardless of rho, or is
flat/noisy -- would indicate the gate isn't learning the intended signal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import time
import numpy as np

from src.data import make_panel
from src.train import train_one_run

RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_ASSETS = 10
N_PATHS = 30
N_STEPS = 300
EPOCHS = 250
SEED = 0


def main():
    print(
        f"Experiment 1: {len(RHO_GRID)} rho values x {EPOCHS} epochs, "
        f"panel = {N_PATHS} paths x {N_STEPS} steps x {N_ASSETS} assets\n",
        flush=True,
    )

    results = []
    grid_t0 = time.time()
    for i, rho in enumerate(RHO_GRID, start=1):
        tag = f"[{i}/{len(RHO_GRID)} rho={rho:.1f}]"
        print(f"{tag} generating synthetic panel...", flush=True)
        x = make_panel(
            rho=rho,
            n_assets=N_ASSETS,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            seed=SEED,
        )
        out = train_one_run(x, epochs=EPOCHS, seed=SEED, log_every=50, tag=tag)
        final_alpha = out["final_alpha"]
        results.append((rho, final_alpha, out["loss_history"][-1]))
        print(f"{tag} recorded alpha={final_alpha:.4f}  "
              f"(grid elapsed {time.time() - grid_t0:.1f}s)\n", flush=True)

    rhos = np.array([r[0] for r in results])
    alphas = np.array([r[1] for r in results])
    corr = np.corrcoef(rhos, alphas)[0, 1]
    print(f"\nPearson corr(rho, alpha) = {corr:.4f}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "experiment1_recovery.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rho", "alpha", "final_loss"])
        writer.writerows(results)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
