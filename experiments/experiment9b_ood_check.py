"""
OOD check for Experiment 9: train on the real-shaped rho curve as before,
then evaluate on a TIME-REVERSED version of that same curve with fresh
noise seeds. If the model memorized "day t -> value" by position, this
should collapse. If it genuinely reads the input signal, performance
should hold up reasonably (not necessarily identically, since the reversed
curve has different local dynamics, but not catastrophically worse).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb  # import before torch -- see experiment9_real_shaped_rho.py
import torch
import torch.nn as nn

from jump_diffusion_generator import JumpDiffusionGenerator
from experiment7_regime_switch import per_step_coupling_stat
from experiment8_adaptive_gate import AdaptiveGate, HIDDEN, LR
from experiment8b_lgb_control import make_features
from experiment9_real_shaped_rho import RHO_PATH, N_STEPS, N_ASSETS, DT, TRAIN_SEEDS, TEST_SEEDS, r2, gen_returns, EPOCHS

REVERSED_PATH = RHO_PATH[::-1].copy()
OOD_SEEDS = list(range(400, 420))


def gen_returns_reversed(seed):
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=0.5, lambda_common=2.0, lambda_idio=1.0)
    log_paths = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=1, seed=seed, rho_path=REVERSED_PATH)
    return np.diff(log_paths, axis=1)[0]


def main():
    print("Retraining on forward curve, evaluating on time-reversed curve (fresh seeds)...")
    train_s, train_y = [], []
    for seed in TRAIN_SEEDS:
        rets = gen_returns(seed)
        train_s.append(per_step_coupling_stat(rets))
        train_y.append(RHO_PATH)
    train_s, train_y = np.array(train_s), np.array(train_y)

    ood_s = []
    for seed in OOD_SEEDS:
        rets = gen_returns_reversed(seed)
        ood_s.append(per_step_coupling_stat(rets))
    ood_s = np.array(ood_s)

    # LightGBM
    rows = []
    for i in range(len(TRAIN_SEEDS)):
        feats = make_features(train_s[i])
        feats["rho_true"] = train_y[i]
        rows.append(feats)
    train_df = pd.concat(rows, ignore_index=True).dropna()
    feature_cols = [c for c in train_df.columns if c != "rho_true"]
    lgb_model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                   min_child_samples=50, verbosity=-1)
    lgb_model.fit(train_df[feature_cols], train_df["rho_true"])

    lgb_r2s = []
    for i in range(len(OOD_SEEDS)):
        feats = make_features(ood_s[i])
        valid = feats.dropna()
        pred = np.full(N_STEPS, np.nan)
        pred[valid.index] = lgb_model.predict(valid[feature_cols])
        lgb_r2s.append(r2(REVERSED_PATH, pred))
    print(f"LightGBM on reversed-curve OOD data: R^2={np.mean(lgb_r2s):.4f}  "
          f"(forward-curve held-out was 0.3283)")

    # GRU
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
        Xood = torch.tensor((ood_s - mu) / sd, dtype=torch.float32).unsqueeze(-1)
        pred_ood = model(Xood).numpy()
    gru_r2 = np.mean([r2(REVERSED_PATH, pred_ood[i]) for i in range(len(OOD_SEEDS))])
    print(f"GRU on reversed-curve OOD data: R^2={gru_r2:.4f}  (forward-curve held-out was 0.5031)")


if __name__ == "__main__":
    main()
