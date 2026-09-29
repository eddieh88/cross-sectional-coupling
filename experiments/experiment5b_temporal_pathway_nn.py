"""
Experiment 5b -- train the model's TEMPORAL PATHWAY (in isolation, no gate,
no cross-sectional pathway) against the classical AR(1) / threshold-AR(1)
baselines established in 5a, with two guardrails against a misleading
result -- flagged explicitly before this ran, given this is the first
experiment in the whole project where the outcome is genuinely uncertain:

1. APPLES-TO-APPLES EVAL: same held-out train/test SPLIT (by path index,
   never touched during fitting/training), same R^2 definition
   (1 - resid_var/true_var over the exact same held-out reconstruction
   targets), for both the classical baselines and the NN. No training-set
   numbers reported as if they were held-out.

2. PRE-REGISTERED WIN/LOSS CRITERION, given how small the classical gap is
   (linear AR(1) R^2=0.119 vs threshold-AR(1) R^2=0.139 in 5a -- a modest
   effect at this panel scale). Decided BEFORE seeing the NN's result:
     - GENUINE WIN: NN mean held-out R^2 (across training seeds) exceeds
       threshold-AR's R^2 by more than threshold-AR's own natural
       data-sampling noise band (its std across 5 independently generated
       datasets), AND the NN's own across-seed std is small relative to
       that gap (not one lucky init).
     - GENUINE LOSS: NN mean held-out R^2 <= linear AR(1)'s R^2 (fails to
       even match the WEAKER, mis-specified classical baseline).
     - INCONCLUSIVE: anything between -- the gap is within noise, no claim
       either way.

Isolation rationale: training the FULL gated model (with the cross-sectional
pathway active) on this task would let the much stronger rho=0.5
cross-sectional signal dominate the aggregate MSE gradient and drown out the
much weaker momentum signal -- the same dilution failure mode diagnosed in
Experiment 3 Phase A.5 (rare jump signal diluted by aggregate loss). Testing
the temporal pathway standalone against a same-shape target (contemporaneous
reconstruction of x[t,i] from x[<t,i] only) isolates exactly the capability
in question and avoids repeating that known pitfall.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from jump_diffusion_generator import JumpDiffusionGenerator
from src.model import TemporalPathway

N_ASSETS = 10
N_STEPS = 300
TRAIN_PATHS = 20
TEST_PATHS = 10
DT = 1 / 252
RHO = 0.5
LAMBDA_COMMON = 2.0
LAMBDA_IDIO = 1.0
MOMENTUM_PHI_LOW = 0.0
MOMENTUM_PHI_HIGH = 0.4
MOMENTUM_THRESHOLD_MULT = 1.0

PRIMARY_DATA_SEED = 0
NOISE_BAND_DATA_SEEDS = [0, 1, 2, 3, 4]
NN_TRAIN_SEEDS = [0, 1, 2, 3, 4]
SECONDARY_DATA_SEEDS = [1, 2]  # extra dataset-generalization check, one NN run each

EPOCHS = 300
LR = 1e-2
HIDDEN = 16
KERNEL_SIZE = 5
N_CONV_LAYERS = 2


def make_panel(data_seed):
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO, lambda_common=LAMBDA_COMMON,
                                  lambda_idio=LAMBDA_IDIO, momentum_phi_low=MOMENTUM_PHI_LOW,
                                  momentum_phi_high=MOMENTUM_PHI_HIGH,
                                  momentum_threshold_mult=MOMENTUM_THRESHOLD_MULT)
    log_paths = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=TRAIN_PATHS + TEST_PATHS, seed=data_seed)
    rets = np.diff(log_paths, axis=1)  # (P, T, N)
    return gen, rets[:TRAIN_PATHS], rets[TRAIN_PATHS:]


def fit_ar1(x_prev, x_next):
    A = np.vstack([x_prev, np.ones_like(x_prev)]).T
    coef, *_ = np.linalg.lstsq(A, x_next, rcond=None)
    return coef


def classical_baselines_held_out(gen, train_rets, test_rets):
    """Fit on train paths, evaluate R^2 on held-out test paths. Returns (linear_r2, tar_r2)."""
    lin_resid_sq, lin_total_sq = 0.0, 0.0
    tar_resid_sq, tar_total_sq = 0.0, 0.0

    for i in range(N_ASSETS):
        r0_tr, r1_tr = train_rets[:, :-1, i].reshape(-1), train_rets[:, 1:, i].reshape(-1)
        r0_te, r1_te = test_rets[:, :-1, i].reshape(-1), test_rets[:, 1:, i].reshape(-1)

        # linear AR(1): fit once on train, apply to test
        coef_lin = fit_ar1(r0_tr, r1_tr)
        pred_lin = coef_lin[0] * r0_te + coef_lin[1]
        lin_resid_sq += ((r1_te - pred_lin) ** 2).sum()
        lin_total_sq += ((r1_te - r1_te.mean()) ** 2).sum()

        # threshold-AR(1): fit two regimes on train, apply same threshold split to test
        threshold = MOMENTUM_THRESHOLD_MULT * gen.sigma[i] * np.sqrt(DT)
        low_tr = np.abs(r0_tr) < threshold
        coefs = {}
        for regime, mask in [("low", low_tr), ("high", ~low_tr)]:
            coefs[regime] = fit_ar1(r0_tr[mask], r1_tr[mask]) if mask.sum() >= 2 else np.array([0.0, 0.0])

        low_te = np.abs(r0_te) < threshold
        pred_tar = np.where(low_te, coefs["low"][0] * r0_te + coefs["low"][1],
                             coefs["high"][0] * r0_te + coefs["high"][1])
        tar_resid_sq += ((r1_te - pred_tar) ** 2).sum()
        tar_total_sq += ((r1_te - r1_te.mean()) ** 2).sum()

    linear_r2 = 1 - lin_resid_sq / lin_total_sq
    tar_r2 = 1 - tar_resid_sq / tar_total_sq
    return linear_r2, tar_r2


class TemporalOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = TemporalPathway(HIDDEN, KERNEL_SIZE, N_CONV_LAYERS)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.temporal(x)
        return self.head(h).squeeze(-1)


def train_and_eval_nn(train_rets, test_rets, train_seed):
    torch.manual_seed(train_seed)
    model = TemporalOnlyModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    x_train = torch.tensor(train_rets, dtype=torch.float32)
    x_test = torch.tensor(test_rets, dtype=torch.float32)

    model.train()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, x_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test).numpy()
    true_test = test_rets
    resid_sq = ((true_test - pred_test) ** 2).sum()
    total_sq = ((true_test - true_test.mean()) ** 2).sum()
    return 1 - resid_sq / total_sq


def main():
    print("=== Step 1: classical baseline noise band across independent datasets ===")
    lin_r2s, tar_r2s = [], []
    for seed in NOISE_BAND_DATA_SEEDS:
        gen, train_rets, test_rets = make_panel(seed)
        lin_r2, tar_r2 = classical_baselines_held_out(gen, train_rets, test_rets)
        lin_r2s.append(lin_r2)
        tar_r2s.append(tar_r2)
        print(f"  data_seed={seed}: linear AR(1) R^2={lin_r2:.4f}  threshold-AR(1) R^2={tar_r2:.4f}")
    lin_r2s, tar_r2s = np.array(lin_r2s), np.array(tar_r2s)
    print(f"  linear AR(1):    mean={lin_r2s.mean():.4f} std={lin_r2s.std():.4f}")
    print(f"  threshold-AR(1): mean={tar_r2s.mean():.4f} std={tar_r2s.std():.4f}")

    print(f"\n=== Step 2: temporal-pathway-only NN, {len(NN_TRAIN_SEEDS)} training seeds, "
          f"fixed primary dataset (seed={PRIMARY_DATA_SEED}) ===")
    gen0, train0, test0 = make_panel(PRIMARY_DATA_SEED)
    lin_r2_primary, tar_r2_primary = classical_baselines_held_out(gen0, train0, test0)
    print(f"  classical on primary dataset: linear AR(1)={lin_r2_primary:.4f}  threshold-AR(1)={tar_r2_primary:.4f}")

    nn_r2s = []
    for seed in NN_TRAIN_SEEDS:
        r2 = train_and_eval_nn(train0, test0, seed)
        nn_r2s.append(r2)
        print(f"  NN train_seed={seed}: held-out R^2={r2:.4f}")
    nn_r2s = np.array(nn_r2s)
    print(f"  NN across seeds: mean={nn_r2s.mean():.4f} std={nn_r2s.std():.4f} "
          f"min={nn_r2s.min():.4f} max={nn_r2s.max():.4f}")

    print(f"\n=== Step 3: secondary dataset-generalization check (1 NN run each) ===")
    for seed in SECONDARY_DATA_SEEDS:
        gen_s, train_s, test_s = make_panel(seed)
        lin_r2_s, tar_r2_s = classical_baselines_held_out(gen_s, train_s, test_s)
        nn_r2_s = train_and_eval_nn(train_s, test_s, train_seed=0)
        print(f"  data_seed={seed}: linear={lin_r2_s:.4f} threshold-AR={tar_r2_s:.4f} NN={nn_r2_s:.4f}")

    print("\n=== Verdict (pre-registered criterion) ===")
    gap_needed = tar_r2s.std()
    real_gap = nn_r2s.mean() - tar_r2_primary
    print(f"  threshold-AR noise band (std across 5 datasets): {gap_needed:.4f}")
    print(f"  NN mean R^2 - threshold-AR R^2 (primary dataset): {real_gap:.4f}")
    print(f"  NN own std across training seeds: {nn_r2s.std():.4f}")
    if nn_r2s.mean() <= lin_r2_primary:
        print("  -> GENUINE LOSS: NN doesn't even match the weaker linear AR(1) baseline.")
    elif real_gap > gap_needed and nn_r2s.std() < real_gap:
        print("  -> GENUINE WIN: NN exceeds threshold-AR beyond its own data-sampling noise band, "
              "and consistently so across training seeds.")
    else:
        print("  -> INCONCLUSIVE: gap is within the classical baseline's own noise band, "
              "or too unstable across NN training seeds to call.")


if __name__ == "__main__":
    main()
