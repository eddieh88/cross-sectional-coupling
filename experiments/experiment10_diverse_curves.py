"""
Experiment 10 -- fixes the methodological gap Exp9 exposed: there, every
training AND test path shared the identical rho(t) curve at identical
positions, so a recurrent model could partly win by memorizing "day t ->
value" rather than genuinely reading the input signal (confirmed via the
reversed-curve check, where the GRU collapsed and LightGBM didn't).

Here, EVERY training and test path gets its OWN distinct, randomly
generated curve (different shape family, different random parameters).
No curve repeats, so "day t -> value" is not even a coherent shortcut to
learn -- day 500 means a different true rho in every single example. Any
model that does well here MUST be reading the actual input signal.

Also evaluates specifically on the real Exp6 curve (never used in
training) to check whether training on diverse synthetic shapes produces
something that transfers to the one real pattern that motivated all of
this.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb  # import before torch -- avoids a duplicate-OpenMP segfault on this platform
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

from jump_diffusion_generator import JumpDiffusionGenerator
from experiment7_regime_switch import per_step_coupling_stat
from experiment8_adaptive_gate import AdaptiveGate, HIDDEN, LR
from experiment8b_lgb_control import make_features

N_ASSETS = 20
DT = 1 / 252
N_STEPS = 1066  # matches the real curve length, for comparability with Exp9
N_TRAIN = 40
N_TEST = 20
EPOCHS = 400
LO, HI = 0.2, 0.9

REAL_RHO_PATH = np.load(Path(__file__).resolve().parent.parent / "data_real" / "real_rho_curve.npy")
assert len(REAL_RHO_PATH) == N_STEPS


def random_smoothed_walk(rng, step_std=0.03, smooth_window=15):
    raw = np.cumsum(rng.normal(0, step_std, N_STEPS))
    raw = raw - raw.min()
    raw = raw / (raw.max() + 1e-9) * (HI - LO) + LO
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(raw, kernel, mode="same")
    return np.clip(smoothed, LO, HI)


def random_sinusoid(rng):
    period = rng.uniform(N_STEPS * 0.2, N_STEPS * 1.5)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(N_STEPS)
    raw = np.sin(2 * np.pi * t / period + phase)
    return (raw + 1) / 2 * (HI - LO) + LO


def random_curve(rng):
    return random_smoothed_walk(rng) if rng.random() < 0.5 else random_sinusoid(rng)


def gen_returns(seed, rho_path):
    rng = np.random.default_rng(seed)
    gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=0.5, lambda_common=2.0, lambda_idio=1.0)
    log_paths = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=1, seed=seed, rho_path=rho_path)
    return np.diff(log_paths, axis=1)[0]


def r2(true, pred):
    mask = ~np.isnan(pred)
    t, p = true[mask], pred[mask]
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    curve_rng = np.random.default_rng(42)
    print("Generating training set (each path its own random curve)...")
    train_s, train_y = [], []
    for i in range(N_TRAIN):
        curve = random_curve(curve_rng)
        rets = gen_returns(1000 + i, curve)
        train_s.append(per_step_coupling_stat(rets))
        train_y.append(curve)
    train_s, train_y = np.array(train_s), np.array(train_y)

    print("Generating held-out test set (fresh curves, never seen)...")
    test_s, test_y = [], []
    for i in range(N_TEST):
        curve = random_curve(curve_rng)
        rets = gen_returns(2000 + i, curve)
        test_s.append(per_step_coupling_stat(rets))
        test_y.append(curve)
    test_s, test_y = np.array(test_s), np.array(test_y)

    print("Generating real-curve evaluation set (the actual Exp6 curve, fresh noise)...")
    real_s = []
    for i in range(N_TEST):
        rets = gen_returns(3000 + i, REAL_RHO_PATH)
        real_s.append(per_step_coupling_stat(rets))
    real_s = np.array(real_s)

    results = {"held_out_diverse": {}, "real_curve": {}}

    # --- EWMA, best heuristic halflife ---
    def ewma_apply(x, decay):
        out = np.zeros_like(x)
        out[0] = x[0]
        for t in range(1, len(x)):
            out[t] = decay * out[t - 1] + (1 - decay) * x[t]
        return out

    best_ewma_diverse, best_hl = -np.inf, None
    for hl in [5, 10, 20, 30, 60, 90]:
        decay = 0.5 ** (1.0 / hl)
        r2s = [r2(test_y[i], ewma_apply(test_s[i], decay)) for i in range(N_TEST)]
        if np.mean(r2s) > best_ewma_diverse:
            best_ewma_diverse, best_hl = np.mean(r2s), hl
    results["held_out_diverse"][f"EWMA (halflife={best_hl})"] = best_ewma_diverse
    decay_best = 0.5 ** (1.0 / best_hl)
    results["real_curve"][f"EWMA (halflife={best_hl})"] = np.mean(
        [r2(REAL_RHO_PATH, ewma_apply(real_s[i], decay_best)) for i in range(N_TEST)])

    # --- supervised-optimal AR(1) ---
    def train_mse(decay):
        return np.mean([((ewma_apply(train_s[i], decay) - train_y[i]) ** 2).mean() for i in range(N_TRAIN)])

    res = minimize_scalar(train_mse, bounds=(0.5, 0.999), method="bounded")
    results["held_out_diverse"][f"supervised AR(1) (decay={res.x:.3f})"] = np.mean(
        [r2(test_y[i], ewma_apply(test_s[i], res.x)) for i in range(N_TEST)])
    results["real_curve"][f"supervised AR(1) (decay={res.x:.3f})"] = np.mean(
        [r2(REAL_RHO_PATH, ewma_apply(real_s[i], res.x)) for i in range(N_TEST)])

    # --- LightGBM ---
    rows = []
    for i in range(N_TRAIN):
        feats = make_features(train_s[i])
        feats["rho_true"] = train_y[i]
        rows.append(feats)
    train_df = pd.concat(rows, ignore_index=True).dropna()
    feature_cols = [c for c in train_df.columns if c != "rho_true"]
    lgb_model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                   min_child_samples=50, verbosity=-1)
    lgb_model.fit(train_df[feature_cols], train_df["rho_true"])

    def lgb_predict_series(s):
        feats = make_features(s)
        valid = feats.dropna()
        pred = np.full(N_STEPS, np.nan)
        pred[valid.index] = lgb_model.predict(valid[feature_cols])
        return pred

    results["held_out_diverse"]["LightGBM"] = np.mean([r2(test_y[i], lgb_predict_series(test_s[i])) for i in range(N_TEST)])
    results["real_curve"]["LightGBM"] = np.mean([r2(REAL_RHO_PATH, lgb_predict_series(real_s[i])) for i in range(N_TEST)])

    # --- GRU ---
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
        Xreal = torch.tensor((real_s - mu) / sd, dtype=torch.float32).unsqueeze(-1)
        pred_real = model(Xreal).numpy()
    results["held_out_diverse"]["GRU"] = np.mean([r2(test_y[i], pred_test[i]) for i in range(N_TEST)])
    results["real_curve"]["GRU"] = np.mean([r2(REAL_RHO_PATH, pred_real[i]) for i in range(N_TEST)])

    print("\n=== Held-out R^2 on DIVERSE, never-seen curves (no memorization possible) ===")
    for name, val in results["held_out_diverse"].items():
        print(f"  {name}: R^2={val:.4f}")

    print("\n=== R^2 specifically on the REAL Exp6 curve (never trained on, different from training curve family) ===")
    for name, val in results["real_curve"].items():
        print(f"  {name}: R^2={val:.4f}")


if __name__ == "__main__":
    main()
