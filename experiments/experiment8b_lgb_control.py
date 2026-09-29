"""
Quick control for Experiment 8: does a plain gradient-boosted tree model
(LightGBM) on simple lag/rolling features match the GRU's win, or does the
GRU's recurrent memory specifically matter? Same task, same input signal,
same training/eval protocol as Exp8 -- only the model changes.

Not a full new experiment axis -- a cheap fairness/precision control on an
already-closed result, same spirit as the AR(1) supervised-decay check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from jump_diffusion_generator import JumpDiffusionGenerator
from experiment7_regime_switch import per_step_coupling_stat, detection_lag
from experiment8_adaptive_gate import (
    N_ASSETS, N_STEPS, DT, RHO_LOW, RHO_HIGH, SWITCH_FRAC, SWITCH_STEP,
    TRAIN_SEEDS, N_TRAIN_PATHS,
)

TEST_SEEDS = list(range(200, 220))
OOD_SEEDS = list(range(300, 310))

LAGS = [1, 2, 3, 5, 10, 20]
ROLL_WINDOWS = [5, 10, 20, 40]


def make_features(s: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"s": s})
    for lag in LAGS:
        df[f"lag_{lag}"] = df["s"].shift(lag)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = df["s"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["s"].rolling(w).std()
    df["recent_vs_long"] = df["roll_mean_5"] - df["roll_mean_40"]
    return df


def build_table(seeds, switch_frac=SWITCH_FRAC):
    rows, targets, path_ids = [], [], []
    for pid, seed in enumerate(seeds):
        gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO_LOW, rho_2=RHO_HIGH,
                                      regime_switch_frac=switch_frac, lambda_common=2.0, lambda_idio=1.0)
        log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=1, seed=seed, return_diagnostics=True)
        rets = np.diff(log_paths, axis=1)[0]
        s = per_step_coupling_stat(rets)
        feats = make_features(s)
        feats["rho_true"] = diag["rho_hist"][0]
        feats["path_id"] = pid
        rows.append(feats)
    full = pd.concat(rows, ignore_index=True)
    return full.dropna()


def evaluate(model, feature_cols, seeds, switch_frac):
    switch_step = int(N_STEPS * switch_frac)
    lags, post_stds = [], []
    for seed in seeds:
        gen = JumpDiffusionGenerator(n_assets=N_ASSETS, rho=RHO_LOW, rho_2=RHO_HIGH,
                                      regime_switch_frac=switch_frac, lambda_common=2.0, lambda_idio=1.0)
        log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=1, seed=seed, return_diagnostics=True)
        rets = np.diff(log_paths, axis=1)[0]
        s = per_step_coupling_stat(rets)
        feats = make_features(s)
        valid = feats.dropna()
        pred = model.predict(valid[feature_cols])
        est = np.full(N_STEPS, np.nan)
        est[valid.index] = pred
        lag = detection_lag(est, switch_step, RHO_LOW, RHO_HIGH)
        lags.append(lag)
        post_region = est[switch_step + 60:]
        post_stds.append(np.nanstd(post_region))
    print(f"  switch_frac={switch_frac}: lag mean={np.nanmean(lags):.1f} median={np.nanmedian(lags):.1f}  "
          f"post_std={np.nanmean(post_stds):.4f}")


def main():
    print("Building training table...")
    train_df = build_table(TRAIN_SEEDS)
    feature_cols = [c for c in train_df.columns if c not in ("rho_true", "path_id")]

    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                               min_child_samples=50, verbosity=-1)
    model.fit(train_df[feature_cols], train_df["rho_true"])
    print(f"train R^2: {model.score(train_df[feature_cols], train_df['rho_true']):.4f}")

    print("\nHeld-out evaluation (same switch location as training):")
    evaluate(model, feature_cols, TEST_SEEDS, SWITCH_FRAC)

    print("\nOut-of-distribution switch locations (never seen in training):")
    evaluate(model, feature_cols, OOD_SEEDS, 0.3)
    evaluate(model, feature_cols, OOD_SEEDS, 0.7)

    print("\nFeature importances (top 8):")
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(imp.head(8))


if __name__ == "__main__":
    main()
