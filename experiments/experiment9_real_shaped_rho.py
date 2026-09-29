"""
Experiment 9 -- does the Exp8 result hold up against a REALISTIC,
continuously-varying rho(t), instead of the idealized clean single-switch
step function tested there? Ground truth here is the actual rolling
realized correlation curve computed from real Coinbase data (Exp6),
lightly smoothed, injected into the synthetic generator via `rho_path` so
we still get exact ground truth to check against -- but the truth now
wanders continuously like real markets do, with no single clean moment to
detect.

Same models, same input signal, same training philosophy as Exp8: EWMA,
supervised-optimal AR(1), LightGBM, and the GRU, all trained/evaluated
against this harder target. No detection-lag metric here (there's no
single discrete event) -- just held-out R^2 against the true curve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb  # must import before torch -- importing torch first causes a
                         # segfault from duplicate OpenMP runtimes on this platform
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

from jump_diffusion_generator import JumpDiffusionGenerator
from experiment7_regime_switch import per_step_coupling_stat, ewma_corr
from experiment8_adaptive_gate import AdaptiveGate, HIDDEN, LR
from experiment8b_lgb_control import make_features

N_ASSETS = 20
DT = 1 / 252
TRAIN_SEEDS = list(range(100, 140))
TEST_SEEDS = list(range(200, 220))
EPOCHS = 400

RHO_PATH = np.load(Path(__file__).resolve().parent.parent / "data_real" / "real_rho_curve.npy")
N_STEPS = len(RHO_PATH)


def gen_returns(seed):
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=0.5, lambda_common=2.0, lambda_idio=1.0)
    log_paths = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=1, seed=seed, rho_path=RHO_PATH)
    return np.diff(log_paths, axis=1)[0]


def r2(true, pred):
    mask = ~np.isnan(pred)
    t, p = true[mask], pred[mask]
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    print(f"Real-shaped rho curve: {N_STEPS} steps, range [{RHO_PATH.min():.3f}, {RHO_PATH.max():.3f}]\n")

    print("Generating train/test data...")
    train_s, train_y = [], []
    for seed in TRAIN_SEEDS:
        rets = gen_returns(seed)
        train_s.append(per_step_coupling_stat(rets))
        train_y.append(RHO_PATH)
    test_s, test_rets = [], []
    for seed in TEST_SEEDS:
        rets = gen_returns(seed)
        test_s.append(per_step_coupling_stat(rets))
        test_rets.append(rets)

    train_s, train_y = np.array(train_s), np.array(train_y)
    test_s = np.array(test_s)

    results = {}

    # --- baseline: raw input signal, no smoothing at all ---
    raw_r2 = np.mean([r2(RHO_PATH, test_s[i]) for i in range(len(TEST_SEEDS))])
    results["raw signal (no smoothing)"] = raw_r2

    # --- EWMA, best of a small halflife grid (still unsupervised/heuristic) ---
    best_ewma_r2, best_hl = -np.inf, None
    for hl in [5, 10, 20, 30, 60, 90]:
        decay = 0.5 ** (1.0 / hl)
        r2s = []
        for i in range(len(TEST_SEEDS)):
            est = np.zeros(N_STEPS)
            est[0] = test_s[i][0]
            for t in range(1, N_STEPS):
                est[t] = decay * est[t - 1] + (1 - decay) * test_s[i][t]
            r2s.append(r2(RHO_PATH, est))
        m = np.mean(r2s)
        if m > best_ewma_r2:
            best_ewma_r2, best_hl = m, hl
    results[f"EWMA (best heuristic halflife={best_hl})"] = best_ewma_r2

    # --- supervised-optimal single-decay AR(1), fit against ground truth ---
    def ar1_filter(x, decay):
        out = np.zeros_like(x)
        out[0] = x[0]
        for t in range(1, len(x)):
            out[t] = decay * out[t - 1] + (1 - decay) * x[t]
        return out

    def train_mse(decay):
        total = 0.0
        for i in range(len(TRAIN_SEEDS)):
            est = ar1_filter(train_s[i], decay)
            total += ((est - train_y[i]) ** 2).mean()
        return total / len(TRAIN_SEEDS)

    res = minimize_scalar(train_mse, bounds=(0.5, 0.999), method="bounded")
    best_decay = res.x
    ar1_r2 = np.mean([r2(RHO_PATH, ar1_filter(test_s[i], best_decay)) for i in range(len(TEST_SEEDS))])
    results[f"supervised-optimal AR(1) (decay={best_decay:.4f})"] = ar1_r2

    # --- LightGBM on lag/rolling features ---
    rows = []
    for i, seed in enumerate(TRAIN_SEEDS):
        feats = make_features(train_s[i])
        feats["rho_true"] = train_y[i]
        rows.append(feats)
    train_df = pd.concat(rows, ignore_index=True).dropna()
    feature_cols = [c for c in train_df.columns if c != "rho_true"]
    lgb_model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                   min_child_samples=50, verbosity=-1)
    lgb_model.fit(train_df[feature_cols], train_df["rho_true"])

    lgb_r2s = []
    for i in range(len(TEST_SEEDS)):
        feats = make_features(test_s[i])
        valid = feats.dropna()
        pred = np.full(N_STEPS, np.nan)
        pred[valid.index] = lgb_model.predict(valid[feature_cols])
        lgb_r2s.append(r2(RHO_PATH, pred))
    results["LightGBM"] = np.mean(lgb_r2s)

    # --- GRU (adaptive gate), same architecture as Exp8 ---
    torch.set_num_threads(1)
    mu, sd = train_s.mean(), train_s.std()
    Xtr = torch.tensor((train_s - mu) / sd, dtype=torch.float32).unsqueeze(-1)
    Ytr = torch.tensor(train_y, dtype=torch.float32)
    model = AdaptiveGate(hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"  GRU epoch {epoch}: train MSE={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        Xte = torch.tensor((test_s - mu) / sd, dtype=torch.float32).unsqueeze(-1)
        pred_test = model(Xte).numpy()
    gru_r2 = np.mean([r2(RHO_PATH, pred_test[i]) for i in range(len(TEST_SEEDS))])
    results["GRU (adaptive gate)"] = gru_r2

    print("\n=== Held-out R^2 against the real-shaped rho curve ===")
    for name, val in results.items():
        print(f"  {name}: R^2={val:.4f}")


if __name__ == "__main__":
    main()
