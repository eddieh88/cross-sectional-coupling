"""
Experiment 4b -- does adding a MAGNITUDE feature fix the classification gap
Experiment 4a found (R^2-vs-sigma shape alone is close to non-functional at
low rho for the population a real detector actually flags), and does the
fix survive the multi-factor stress from 4a too?

Motivation: R^2 tests SHAPE (is this event's cross-sectional pattern
proportional to sigma_i?) but throws away SCALE. Jump sizes are drawn from
N(jump_mean, jump_std^2) independent of rho -- a fixed, rho-invariant
absolute scale. A genuinely common diffusive event's typical size scales
with sqrt(rho) * sigma_i, so at low rho even a candidate that clears the
joint detector's threshold is usually much smaller in absolute terms than a
real jump. That's a scale cue the shape-only classifier from 3d/3e/3f/4a
never used.

Two classifiers on [R^2, mean(|return|)] per candidate, both cheap and
classical (no NN):
  - supervised: linear fit on a held-out split (needs SOME labeled
    examples -- realistic if a handful of past events are known/labeled,
    e.g. from other markets or manual review)
  - unsupervised: 2-cluster k-means on the standardized 2D feature space,
    labels used only AFTER fitting to name which cluster is "jump" --
    realistic when no labeled examples exist at all, closer to the spirit
    of the rest of this project's classical toolkit.

Swept across the same rho x factor_dispersion grid as 4a to check both
questions in one pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jump_diffusion_generator import JumpDiffusionGenerator
from src.baseline import _bipower_variation

N_ASSETS = 20
SIGMA = np.linspace(0.10, 0.40, N_ASSETS)
N_PATHS = 50
N_STEPS = 300
DT = 1 / 252
LAMBDA_COMMON = 2.0
LAMBDA_IDIO = 1.0
RHO_GRID = [0.3, 0.6, 0.9]
DISPERSION_GRID = [0.0, 0.3, 0.6, 1.0]
SEED = 7
JOINT_K = 10.0

SPLIT_SEED = 0
KMEANS_SEED = 1


def r_squared(y: np.ndarray, x: np.ndarray) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def estimate_sigma_per_asset(rets_path: np.ndarray) -> np.ndarray:
    n = rets_path.shape[1]
    return np.array([np.sqrt(_bipower_variation(rets_path[:, i])) for i in range(n)])


def robust_threshold(values: np.ndarray, k: float) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826
    return med + k * mad


def kmeans_2(X, seed):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(100):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        new_centers = np.array([
            X[assign == k].mean(0) if (assign == k).any() else centers[k]
            for k in range(2)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return assign


def main():
    for rho in RHO_GRID:
        print(f"\n=== rho={rho} ===")
        for disp in DISPERSION_GRID:
            gen = JumpDiffusionGenerator(n_assets=N_ASSETS, sigma=SIGMA, rho=rho,
                                          lambda_common=LAMBDA_COMMON, lambda_idio=LAMBDA_IDIO,
                                          factor_dispersion=disp)
            log_paths, diag = gen.simulate(n_steps=N_STEPS, dt=DT, n_paths=N_PATHS, seed=SEED,
                                            return_diagnostics=True)
            rets = np.diff(log_paths, axis=1)
            jump_fired = diag["common_jump_fired"]

            feats, labels = [], []
            for p in range(N_PATHS):
                sigma_hat = estimate_sigma_per_asset(rets[p])
                z = rets[p] / sigma_hat[None, :]
                j_stat = (z ** 2).sum(axis=1)
                thresh = robust_threshold(j_stat, JOINT_K)
                for t in np.where(j_stat > thresh)[0]:
                    mag = np.mean(np.abs(rets[p, t, :]))
                    r2 = r_squared(rets[p, t, :], sigma_hat)
                    feats.append([r2, mag])
                    labels.append(1 if jump_fired[p, t] else 0)
            feats, labels = np.array(feats), np.array(labels)
            n = len(feats)
            if n < 10:
                print(f"  dispersion={disp:.1f}: too few candidates (n={n}), skipping")
                continue

            # supervised: fit on a random half, evaluate on the other half
            split_rng = np.random.default_rng(SPLIT_SEED)
            idx = split_rng.permutation(n)
            half = n // 2
            train_idx, test_idx = idx[:half], idx[half:]
            mu, sd = feats[train_idx].mean(0), feats[train_idx].std(0)
            X_train = (feats[train_idx] - mu) / sd
            X_test = (feats[test_idx] - mu) / sd
            A_train = np.hstack([X_train, np.ones((len(X_train), 1))])
            coef, *_ = np.linalg.lstsq(A_train, labels[train_idx], rcond=None)
            A_test = np.hstack([X_test, np.ones((len(X_test), 1))])
            pred = (A_test @ coef) > 0.5
            supervised_acc = (pred == labels[test_idx].astype(bool)).mean()

            # unsupervised: 2-means on standardized features, labels used only to name clusters after
            X_all = (feats - feats.mean(0)) / feats.std(0)
            assign = kmeans_2(X_all, KMEANS_SEED)
            unsupervised_acc = max((assign == labels).mean(), (assign != labels).mean())

            # R^2-only baseline (Experiment 4a's method) for direct comparison
            r2_only_best = max(
                ((feats[:, 0] < thr) == labels.astype(bool)).mean()
                for thr in np.linspace(0, 1, 101)
            )

            print(f"  dispersion={disp:.1f}: n={n:4d}  "
                  f"R2-only best-threshold acc={r2_only_best:.3f}  |  "
                  f"[R2+mag] supervised held-out acc={supervised_acc:.3f}  "
                  f"unsupervised (no labels) acc={unsupervised_acc:.3f}")


if __name__ == "__main__":
    main()
