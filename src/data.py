"""Helpers to turn JumpDiffusionGenerator output into training tensors."""

import numpy as np
import torch

from jump_diffusion_generator import JumpDiffusionGenerator


def make_panel(
    rho: float,
    n_assets: int = 10,
    n_paths: int = 50,
    n_steps: int = 500,
    dt: float = 1 / 252,
    lambda_common: float = 2.0,
    lambda_idio: float = 1.0,
    jump_std: float = 0.05,
    jump_mean: float = -0.02,
    seed: int | None = None,
) -> torch.Tensor:
    """Simulate `n_paths` independent panels at a fixed rho and return log-returns.

    `jump_std`/`jump_mean` are exposed (generator defaults otherwise) so
    Experiment 2's jump-contamination robustness test can crank up jump
    noise without touching `jump_coupling` (still governed by lambda_common
    / lambda_idio, unchanged).

    Returns a tensor of shape (n_paths, n_steps, n_assets).
    """
    gen = JumpDiffusionGenerator(
        n_assets=n_assets,
        rho=rho,
        lambda_common=lambda_common,
        lambda_idio=lambda_idio,
        jump_std=jump_std,
        jump_mean=jump_mean,
    )
    log_paths = gen.simulate(n_steps=n_steps, dt=dt, n_paths=n_paths, seed=seed)  # (P, T+1, N)
    rets = np.diff(log_paths, axis=1)  # (P, T, N) log-returns
    # standardize per-asset so all rho settings are on a comparable scale
    rets = (rets - rets.mean(axis=(0, 1), keepdims=True)) / (rets.std(axis=(0, 1), keepdims=True) + 1e-8)
    return torch.tensor(rets, dtype=torch.float32)
