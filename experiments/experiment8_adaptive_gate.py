"""
Experiment 8 -- the final round. Tests whether a small, genuinely adaptive
gate (a GRU whose update gate can learn to change its own effective memory
based on recent conditions) beats the classical ceiling established in
Experiment 7 (EWMA correlation) at tracking a regime switch in rho, on the
identical task, identical input, and identical evaluation protocol.

Per results/experiment8_prereg.md (written before this file), three
outcomes are possible and pre-defined: WIN (dominates EWMA's frontier at a
majority of matched operating points), REDISCOVERY (learns to approximate
a fixed exponential decay -- i.e., just reinvents EWMA), or LOSS (dominated
by EWMA everywhere). This is the last round of the investigation regardless
of which of the three obtains.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from jump_diffusion_generator import JumpDiffusionGenerator
from experiment7_regime_switch import per_step_coupling_stat, detection_lag, ewma_corr

N_ASSETS = 20
N_STEPS = 2000
DT = 1 / 252
RHO_LOW = 0.2
RHO_HIGH = 0.85
SWITCH_FRAC = 0.5
SWITCH_STEP = int(N_STEPS * SWITCH_FRAC)

N_TRAIN_PATHS = 40
N_TEST_PATHS = 20
TRAIN_SEEDS = list(range(100, 100 + N_TRAIN_PATHS))
TEST_SEEDS = list(range(200, 200 + N_TEST_PATHS))

HIDDEN = 12
EPOCHS = 400
LR = 5e-3

# EWMA frontier established in Experiment 7, for direct comparison
EWMA_FRONTIER = [  # (halflife, lag, within_regime_std_post)
    (5, 6.3, 0.1143), (10, 14.3, 0.0993), (20, 27.6, 0.0825),
    (30, 40.7, 0.0720), (60, 82.2, 0.0542), (90, 127.9, 0.0427),
]


def build_dataset(seeds):
    """Returns (inputs, targets): inputs (n_paths, T) coupling stat, targets (n_paths, T) true rho."""
    inputs, targets = [], []
    for seed in seeds:
        gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO_LOW, rho_2=RHO_HIGH,
                                      regime_switch_frac=SWITCH_FRAC, lambda_common=2.0, lambda_idio=1.0)
        log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=1, seed=seed, return_diagnostics=True)
        rets = np.diff(log_paths, axis=1)[0]
        s = per_step_coupling_stat(rets)
        inputs.append(s)
        targets.append(diag["rho_hist"][0])
    return np.array(inputs), np.array(targets)


class AdaptiveGate(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, 1)
        h, _ = self.gru(x)
        return torch.sigmoid(self.head(h)).squeeze(-1)  # (B, T), in [0,1] like rho


def main():
    print("Building train/test datasets (ground-truth rho regression, same input as EWMA/CUSUM)...")
    x_train, y_train = build_dataset(TRAIN_SEEDS)
    x_test, y_test = build_dataset(TEST_SEEDS)

    # standardize input using train statistics only
    mu, sd = x_train.mean(), x_train.std()
    x_train_n = (x_train - mu) / sd
    x_test_n = (x_test - mu) / sd

    Xtr = torch.tensor(x_train_n, dtype=torch.float32).unsqueeze(-1)
    Ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(x_test_n, dtype=torch.float32).unsqueeze(-1)

    model = AdaptiveGate()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Training GRU gate ({EPOCHS} epochs, {N_TRAIN_PATHS} train paths)...")
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch}: train MSE={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        pred_test = model(Xte).numpy()  # (n_test, T)

    print("\n=== Held-out evaluation: detection lag + within-regime noise ===")
    lags, pre_stds, post_stds = [], [], []
    for p in range(N_TEST_PATHS):
        est = pred_test[p]
        lag = detection_lag(est, SWITCH_STEP, RHO_LOW, RHO_HIGH)
        lags.append(lag)
        pre_region = est[50: SWITCH_STEP - 20]
        post_region = est[SWITCH_STEP + 60:]
        pre_stds.append(pre_region.std())
        post_stds.append(post_region.std())
    lags = np.array(lags, dtype=float)
    print(f"adaptive gate: detection lag mean={np.nanmean(lags):.1f} (median={np.nanmedian(lags):.1f})  |  "
          f"within-regime std: pre={np.mean(pre_stds):.4f} post={np.mean(post_stds):.4f}")

    print("\nEWMA frontier for comparison (halflife, lag, post-switch std):")
    for hl, lag, std in EWMA_FRONTIER:
        print(f"  halflife={hl}: lag={lag}, std={std}")

    # --- Rediscovery check: does the gate's effective impulse response look like a fixed exponential decay? ---
    print("\n=== Rediscovery check: impulse-response shape, stable regime vs. just-after-switch ===")

    def impulse_response(t_eval, seed, k_max=60):
        gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO_LOW, rho_2=RHO_HIGH,
                                      regime_switch_frac=SWITCH_FRAC, lambda_common=2.0, lambda_idio=1.0)
        log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=1, seed=seed, return_diagnostics=True)
        rets = np.diff(log_paths, axis=1)[0]
        s = per_step_coupling_stat(rets)
        s_n = (s - mu) / sd
        x = torch.tensor(s_n[:t_eval + 1], dtype=torch.float32).view(1, -1, 1)
        x.requires_grad_(True)
        h, _ = model.gru(x)
        out = torch.sigmoid(model.head(h)).squeeze(-1)
        out[0, -1].backward()
        grad = x.grad[0, :, 0].numpy()
        return grad[-k_max:][::-1]  # most recent lag first: [t, t-1, t-2, ...]

    def fit_exp_decay(weights):
        k = np.arange(len(weights))
        w = np.abs(weights) + 1e-12
        logw = np.log(w)
        A = np.vstack([k, np.ones_like(k)]).T
        coef, *_ = np.linalg.lstsq(A, logw, rcond=None)
        pred = A @ coef
        ss_res = ((logw - pred) ** 2).sum()
        ss_tot = ((logw - logw.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        decay_rate = np.exp(coef[0])
        return decay_rate, r2

    stable_resp = impulse_response(t_eval=500, seed=300)
    postbreak_resp = impulse_response(t_eval=SWITCH_STEP + 10, seed=300)

    decay_stable, r2_stable = fit_exp_decay(stable_resp)
    decay_post, r2_post = fit_exp_decay(postbreak_resp)
    print(f"stable regime (t=500):        implied decay={decay_stable:.4f}  exp-fit R^2={r2_stable:.3f}")
    print(f"just after switch (t={SWITCH_STEP+10}): implied decay={decay_post:.4f}  exp-fit R^2={r2_post:.3f}")
    print(f"decay rates {'MATCH (consistent with rediscovering a fixed EWMA)' if abs(decay_stable-decay_post) < 0.02 else 'DIFFER (consistent with genuine input-dependent adaptivity)'}")


if __name__ == "__main__":
    main()
