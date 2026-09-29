"""
Experiment 3 (disentanglement phase) -- Phase A + A.5.

Before building a two-gate NN architecture meant to separate diffusive
coupling (rho) from jump-timing coupling (jump_coupling), check two cheap
things first:

Phase A.5 -- does the EXISTING single-gate model conflate the two? A single
scalar alpha reading off one gated mixture has no structural reason to
distinguish "assets move together because of shared diffusion" from
"assets move together because they jump at the same time" -- both just look
like "cross-sectional pathway helps." Test: hold rho fixed, vary
jump_coupling, see if alpha moves. (Experiment 1 already showed alpha
tracks rho at fixed jump_coupling -- this fills in the other arm.)

Phase A -- can a classical, non-NN combination disentangle the two instead?
Bipower variation (robust to jumps by construction) estimates each asset's
diffusive variance; individual returns are flagged as jumps where they
exceed a threshold multiple of that. Two SEPARATE correlation signals:
diffusive_corr (on the jump-scrubbed series) and jump_corr (co-occurrence
of jump flags across assets). If this classical combination already
disentangles cleanly, the NN's potential edge shrinks further before Phase
B is even built.

Both phases use a "cross" design: vary rho at fixed jump_coupling, and vary
jump_coupling at fixed rho, rather than a full grid -- cheaper, and directly
answers "does changing X move the signal meant for Y."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import time
import numpy as np

from src.data import make_panel
from src.train import train_one_run
from src.baseline import diffusive_and_jump_baselines

N_ASSETS = 8
N_PATHS = 50
N_STEPS = 400
TOTAL_LAMBDA = 3.0  # lambda_common + lambda_idio, held fixed throughout
EPOCHS = 250
SEED = 1

RHO_ARM = [0.0, 0.25, 0.5, 0.75, 1.0]
JC_ARM = [0.2, 0.35, 0.5, 0.65, 0.8]
FIXED_JC = 2.0 / 3.0  # matches lambda_common=2.0, lambda_idio=1.0 used in Experiments 1-2
FIXED_RHO = 0.5


def lambdas_for(jump_coupling: float) -> tuple[float, float]:
    return jump_coupling * TOTAL_LAMBDA, (1 - jump_coupling) * TOTAL_LAMBDA


def main():
    classical_rows = []  # (arm, rho, jump_coupling, diffusive_corr, jump_corr)
    conflation_rows = []  # (rho, jump_coupling, alpha)

    t0 = time.time()

    # --- Phase A, arm 1: vary rho at fixed jump_coupling ---
    print(f"=== Phase A, arm 'vary_rho' (jump_coupling fixed ~{FIXED_JC:.3f}) ===", flush=True)
    lc, li = lambdas_for(FIXED_JC)
    for rho in RHO_ARM:
        x = make_panel(rho=rho, n_assets=N_ASSETS, n_paths=N_PATHS, n_steps=N_STEPS,
                        lambda_common=lc, lambda_idio=li, seed=SEED)
        diff_corr, jump_corr = diffusive_and_jump_baselines(x)
        print(f"  rho={rho:.2f}  diffusive_corr={diff_corr:.4f}  jump_corr={jump_corr:.4f}  "
              f"({time.time()-t0:.1f}s elapsed)", flush=True)
        classical_rows.append(("vary_rho", rho, FIXED_JC, diff_corr, jump_corr))

    # --- Phase A, arm 2: vary jump_coupling at fixed rho ---
    print(f"\n=== Phase A, arm 'vary_jump_coupling' (rho fixed = {FIXED_RHO}) ===", flush=True)
    for jc in JC_ARM:
        lc, li = lambdas_for(jc)
        x = make_panel(rho=FIXED_RHO, n_assets=N_ASSETS, n_paths=N_PATHS, n_steps=N_STEPS,
                        lambda_common=lc, lambda_idio=li, seed=SEED)
        diff_corr, jump_corr = diffusive_and_jump_baselines(x)
        print(f"  jump_coupling={jc:.2f}  diffusive_corr={diff_corr:.4f}  jump_corr={jump_corr:.4f}  "
              f"({time.time()-t0:.1f}s elapsed)", flush=True)
        classical_rows.append(("vary_jump_coupling", FIXED_RHO, jc, diff_corr, jump_corr))

    # --- Phase A.5: does the existing single-gate NN model conflate the two? ---
    # (rho arm already answered by Experiment 1: alpha tracks rho at fixed jump_coupling.
    # This fills in the other arm: does alpha ALSO move when only jump_coupling changes?)
    print(f"\n=== Phase A.5: existing single-gate model, vary jump_coupling at fixed rho={FIXED_RHO} ===", flush=True)
    for i, jc in enumerate(JC_ARM, start=1):
        lc, li = lambdas_for(jc)
        tag = f"[{i}/{len(JC_ARM)} jump_coupling={jc:.2f}]"
        x = make_panel(rho=FIXED_RHO, n_assets=N_ASSETS, n_paths=30, n_steps=300,
                        lambda_common=lc, lambda_idio=li, seed=SEED)
        out = train_one_run(x, epochs=EPOCHS, seed=SEED, log_every=50, tag=tag)
        alpha = out["final_alpha"]
        print(f"{tag} alpha={alpha:.4f}  ({time.time()-t0:.1f}s elapsed)", flush=True)
        conflation_rows.append((FIXED_RHO, jc, alpha))

    # --- save + summarize ---
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "experiment3_classical_disentanglement.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "rho", "jump_coupling", "diffusive_corr", "jump_corr"])
        writer.writerows(classical_rows)

    with open(out_dir / "experiment3_conflation_diagnostic.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rho", "jump_coupling", "alpha"])
        writer.writerows(conflation_rows)

    print("\n=== Summary ===")
    vary_rho = [r for r in classical_rows if r[0] == "vary_rho"]
    vary_jc = [r for r in classical_rows if r[0] == "vary_jump_coupling"]

    rho_vals = np.array([r[1] for r in vary_rho])
    diff_over_rho = np.array([r[3] for r in vary_rho])
    jump_over_rho = np.array([r[4] for r in vary_rho])
    jc_vals = np.array([r[2] for r in vary_jc])
    diff_over_jc = np.array([r[3] for r in vary_jc])
    jump_over_jc = np.array([r[4] for r in vary_jc])

    print(f"Classical diffusive_corr: r(rho)={np.corrcoef(rho_vals, diff_over_rho)[0,1]:.4f}  "
          f"(should be high) | r(jump_coupling)={np.corrcoef(jc_vals, diff_over_jc)[0,1]:.4f} (should be low)")
    print(f"Classical jump_corr:      r(jump_coupling)={np.corrcoef(jc_vals, jump_over_jc)[0,1]:.4f}  "
          f"(should be high) | r(rho)={np.corrcoef(rho_vals, jump_over_rho)[0,1]:.4f} (should be low)")

    alphas = np.array([r[2] for r in conflation_rows])
    print(f"\nExisting single-gate model: r(alpha, jump_coupling) at fixed rho={FIXED_RHO} "
          f"= {np.corrcoef(jc_vals, alphas)[0,1]:.4f}  (high value = conflation confirmed)")
    print(f"alpha values across jump_coupling grid: {alphas.tolist()}")

    print(f"\nSaved to {out_dir}/experiment3_classical_disentanglement.csv and experiment3_conflation_diagnostic.csv")


if __name__ == "__main__":
    main()
