"""
Experiment 3c -- retest the classical disentanglement baseline with a
properly calibrated jump test (BNS/Huang-Tauchen ratio statistic,
tripower-quarticity-normalized) instead of the naive fixed-multiple
threshold.

First pass (experiment3_disentanglement.py) found jump_corr contaminated by
rho (r=0.82) using a naive test: flag jump if r_t^2 > 9 * bipower_variation.
Diagnosis: that's not a calibrated statistical test -- it has no defined
false-positive rate, so nothing stops co-occurring false positives
(inevitable once rho makes the underlying series correlated) from
accumulating into a spurious jump_corr-vs-rho relationship.

This script re-runs the same two arms with BOTH methods side by side (same
generated data for both, isolating the jump-detector as the only variable)
to see whether the calibrated test actually fixes the contamination, or
whether it's a structural limit (large diffusive tail comovement being
genuinely hard to distinguish from real co-jumps at high rho, no matter how
well-calibrated the per-asset test is).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import time
import numpy as np

from src.data import make_panel
from src.baseline import diffusive_and_jump_baselines

N_ASSETS = 8
N_PATHS = 50
N_STEPS = 400
TOTAL_LAMBDA = 3.0
SEED = 1

RHO_ARM = [0.0, 0.25, 0.5, 0.75, 1.0]
JC_ARM = [0.2, 0.35, 0.5, 0.65, 0.8]
FIXED_JC = 2.0 / 3.0
FIXED_RHO = 0.5

METHODS = ["naive", "bns"]


def lambdas_for(jump_coupling: float) -> tuple[float, float]:
    return jump_coupling * TOTAL_LAMBDA, (1 - jump_coupling) * TOTAL_LAMBDA


def main():
    rows = []  # (method, arm, rho, jump_coupling, diffusive_corr, jump_corr)
    t0 = time.time()

    for method in METHODS:
        print(f"\n########## method = {method} ##########", flush=True)

        print(f"=== arm 'vary_rho' (jump_coupling fixed ~{FIXED_JC:.3f}) ===", flush=True)
        lc, li = lambdas_for(FIXED_JC)
        for rho in RHO_ARM:
            x = make_panel(rho=rho, n_assets=N_ASSETS, n_paths=N_PATHS, n_steps=N_STEPS,
                            lambda_common=lc, lambda_idio=li, seed=SEED)
            diff_corr, jump_corr = diffusive_and_jump_baselines(x, method=method)
            print(f"  rho={rho:.2f}  diffusive_corr={diff_corr:.4f}  jump_corr={jump_corr:.4f}  "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)
            rows.append((method, "vary_rho", rho, FIXED_JC, diff_corr, jump_corr))

        print(f"\n=== arm 'vary_jump_coupling' (rho fixed = {FIXED_RHO}) ===", flush=True)
        for jc in JC_ARM:
            lc, li = lambdas_for(jc)
            x = make_panel(rho=FIXED_RHO, n_assets=N_ASSETS, n_paths=N_PATHS, n_steps=N_STEPS,
                            lambda_common=lc, lambda_idio=li, seed=SEED)
            diff_corr, jump_corr = diffusive_and_jump_baselines(x, method=method)
            print(f"  jump_coupling={jc:.2f}  diffusive_corr={diff_corr:.4f}  jump_corr={jump_corr:.4f}  "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)
            rows.append((method, "vary_jump_coupling", FIXED_RHO, jc, diff_corr, jump_corr))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "experiment3c_bns_retest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "arm", "rho", "jump_coupling", "diffusive_corr", "jump_corr"])
        writer.writerows(rows)

    print("\n=== Summary: naive vs. bns, cross-contamination of jump_corr by rho ===")
    for method in METHODS:
        vary_rho = [r for r in rows if r[0] == method and r[1] == "vary_rho"]
        vary_jc = [r for r in rows if r[0] == method and r[1] == "vary_jump_coupling"]

        rho_vals = np.array([r[2] for r in vary_rho])
        diff_over_rho = np.array([r[4] for r in vary_rho])
        jump_over_rho = np.array([r[5] for r in vary_rho])
        jc_vals = np.array([r[3] for r in vary_jc])
        diff_over_jc = np.array([r[4] for r in vary_jc])
        jump_over_jc = np.array([r[5] for r in vary_jc])

        print(f"\n[{method}]")
        print(f"  diffusive_corr: r(rho)={np.corrcoef(rho_vals, diff_over_rho)[0,1]:.4f}  "
              f"| r(jump_coupling)={np.corrcoef(jc_vals, diff_over_jc)[0,1]:.4f}")
        print(f"  jump_corr:      r(jump_coupling)={np.corrcoef(jc_vals, jump_over_jc)[0,1]:.4f}  "
              f"| r(rho)={np.corrcoef(rho_vals, jump_over_rho)[0,1]:.4f}  <-- contamination metric")

    print(f"\nSaved to {out_dir}/experiment3c_bns_retest.csv")


if __name__ == "__main__":
    main()
