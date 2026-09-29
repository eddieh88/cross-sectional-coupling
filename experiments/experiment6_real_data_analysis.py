"""
Experiment 6 -- run the classical toolkit (correlation/PCA, joint detection
statistic, BNS jump flags, threshold-AR) against the real Coinbase panel,
per the pre-registered checks in results/experiment6_prereg.md. Written
and committed BEFORE looking at any of these numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.baseline import _bipower_variation, _bns_jump_flags

PANEL_PATH = Path(__file__).resolve().parent.parent / "data_real" / "coinbase_daily_panel_3y.parquet"
JOINT_K = 10.0
ROLLING_WINDOW = 30
TRAIN_FRAC = 2 / 3  # ~2 years train, ~1 year held out, walk-forward per pre-reg


def robust_threshold(values: np.ndarray, k: float) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826
    return med + k * mad


def r_squared(y: np.ndarray, x: np.ndarray) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def ljung_box_q(r: np.ndarray, max_lag: int = 10) -> tuple:
    n = len(r)
    acfs = [np.corrcoef(r[:-k], r[k:])[0, 1] for k in range(1, max_lag + 1)]
    Q = n * (n + 2) * sum((a ** 2) / (n - k) for k, a in enumerate(acfs, 1))
    return Q, acfs


def main():
    panel = pd.read_parquet(PANEL_PATH)
    rets = np.log(panel).diff().dropna()
    assets = rets.columns.tolist()
    n_assets = len(assets)
    print(f"Panel: {rets.shape[0]} days x {n_assets} assets, "
          f"{rets.index.min().date()} -> {rets.index.max().date()}\n")

    # ============ CHECK 1: correlation / rank structure / joint statistic ============
    print("=" * 70)
    print("CHECK 1: correlation structure")
    print("=" * 70)
    corr = rets.corr().values
    off_diag = corr[np.triu_indices(n_assets, 1)]
    print(f"mean pairwise correlation: {off_diag.mean():.3f}")

    eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    total = eigvals.sum()
    print(f"eigenvalues (top 5): {eigvals[:5].round(2)}")
    print(f"top eigenvalue share of total variance: {eigvals[0]/total:.3f}")
    print(f"top-2 eigenvalues share: {eigvals[:2].sum()/total:.3f}")

    rolling_corr = rets.rolling(ROLLING_WINDOW).corr()
    rolling_mean_corr = []
    dates = rets.index[ROLLING_WINDOW - 1:]
    for d in dates:
        block = rolling_corr.loc[d].values
        rolling_mean_corr.append(block[np.triu_indices(n_assets, 1)].mean())
    rolling_mean_corr = pd.Series(rolling_mean_corr, index=dates)
    print(f"\nrolling {ROLLING_WINDOW}d mean correlation: "
          f"mean={rolling_mean_corr.mean():.3f} std={rolling_mean_corr.std():.3f} "
          f"min={rolling_mean_corr.min():.3f} max={rolling_mean_corr.max():.3f}")
    print(f"5 highest-correlation windows (end date): "
          f"{rolling_mean_corr.nlargest(5).index.date.tolist()}")
    print(f"5 lowest-correlation windows (end date): "
          f"{rolling_mean_corr.nsmallest(5).index.date.tolist()}")

    # joint statistic: sigma_hat per asset via bipower variation, then J[t] = sum_i Z[t,i]^2
    sigma_hat = np.array([np.sqrt(_bipower_variation(rets[a].values)) for a in assets])
    z = rets.values / sigma_hat[None, :]
    j_stat = (z ** 2).sum(axis=1)
    thresh = robust_threshold(j_stat, JOINT_K)
    candidate_mask = j_stat > thresh
    print(f"\njoint statistic (k={JOINT_K}): {candidate_mask.sum()} / {len(j_stat)} days flagged "
          f"({candidate_mask.mean()*100:.1f}%)")
    candidate_dates = rets.index[candidate_mask]

    # ============ CHECK 2: BNS jump flags + top candidate dates for news lookup ============
    print("\n" + "=" * 70)
    print("CHECK 2: BNS jump flags, top candidate dates for real-event lookup")
    print("=" * 70)
    asset_flags = np.stack([_bns_jump_flags(rets[a].values) for a in assets], axis=1)
    n_flagged_per_day = asset_flags.sum(axis=1)
    bns_candidate_mask = n_flagged_per_day >= 2  # at least 2 assets flagged same day
    print(f"BNS: {bns_candidate_mask.sum()} days with >=2 assets flagged simultaneously")

    combined_score = j_stat / j_stat.max() + n_flagged_per_day / max(n_flagged_per_day.max(), 1)
    top_idx = np.argsort(-combined_score)[:15]
    top_idx = sorted(top_idx)
    print("\nTop 15 candidate dates (joint stat + BNS breadth combined score):")
    for i in top_idx:
        d = rets.index[i]
        movers = rets.iloc[i].abs().sort_values(ascending=False).head(3)
        print(f"  {d.date()}  J={j_stat[i]:.1f}  n_bns_flagged={n_flagged_per_day[i]}  "
              f"biggest movers: {', '.join(f'{a}={rets.iloc[i][a]*100:+.1f}%' for a in movers.index)}")

    # ============ CHECK 3: temporal structure, walk-forward ============
    print("\n" + "=" * 70)
    print("CHECK 3: temporal structure (walk-forward, train->test)")
    print("=" * 70)
    n_train = int(len(rets) * TRAIN_FRAC)
    train_rets, test_rets = rets.iloc[:n_train], rets.iloc[n_train:]
    print(f"train: {train_rets.index.min().date()} -> {train_rets.index.max().date()} "
          f"({len(train_rets)} days)")
    print(f"test:  {test_rets.index.min().date()} -> {test_rets.index.max().date()} "
          f"({len(test_rets)} days)\n")

    lin_r2s, tar_r2s, lb_qs = [], [], []
    for a in assets:
        r_test = test_rets[a].values
        Q, _ = ljung_box_q(r_test, max_lag=10)
        lb_qs.append(Q)

        r0_tr, r1_tr = train_rets[a].values[:-1], train_rets[a].values[1:]
        r0_te, r1_te = test_rets[a].values[:-1], test_rets[a].values[1:]

        A = np.vstack([r0_tr, np.ones_like(r0_tr)]).T
        coef, *_ = np.linalg.lstsq(A, r1_tr, rcond=None)
        pred_lin = coef[0] * r0_te + coef[1]
        lin_r2 = 1 - ((r1_te - pred_lin) ** 2).sum() / ((r1_te - r1_te.mean()) ** 2).sum()
        lin_r2s.append(lin_r2)

        threshold = 1.0 * np.std(r0_tr)
        low_tr = np.abs(r0_tr) < threshold
        coefs = {}
        for regime, mask in [("low", low_tr), ("high", ~low_tr)]:
            coefs[regime] = np.linalg.lstsq(
                np.vstack([r0_tr[mask], np.ones(mask.sum())]).T, r1_tr[mask], rcond=None
            )[0] if mask.sum() >= 2 else np.array([0.0, 0.0])
        low_te = np.abs(r0_te) < threshold
        pred_tar = np.where(low_te, coefs["low"][0] * r0_te + coefs["low"][1],
                             coefs["high"][0] * r0_te + coefs["high"][1])
        tar_r2 = 1 - ((r1_te - pred_tar) ** 2).sum() / ((r1_te - r1_te.mean()) ** 2).sum()
        tar_r2s.append(tar_r2)

    lb_qs = np.array(lb_qs)
    lin_r2s, tar_r2s = np.array(lin_r2s), np.array(tar_r2s)
    critical_5pct = 18.31  # chi2(10) critical value at 5%
    n_significant = (lb_qs > critical_5pct).sum()
    print(f"Ljung-Box Q (10 lags, held-out returns), critical value at 5% = {critical_5pct}")
    print(f"  assets with significant autocorrelation: {n_significant} / {n_assets}")
    print(f"  Q stats: mean={lb_qs.mean():.2f} median={np.median(lb_qs):.2f} max={lb_qs.max():.2f}")
    print(f"\nheld-out R^2: linear AR(1) mean={lin_r2s.mean():.4f}  threshold-AR(1) mean={tar_r2s.mean():.4f}")
    print(f"  assets with linear AR(1) R^2 > 0.02: {(lin_r2s > 0.02).sum()} / {n_assets}")
    print(f"  assets with threshold-AR(1) R^2 > 0.02: {(tar_r2s > 0.02).sum()} / {n_assets}")


if __name__ == "__main__":
    main()
