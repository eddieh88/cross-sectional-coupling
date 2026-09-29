"""Training loop for GatedInterpolationModel: contemporaneous leave-one-out reconstruction."""

import time

import torch
import torch.nn as nn

from src.model import GatedInterpolationModel


def train_one_run(
    x: torch.Tensor,
    hidden: int = 16,
    kernel_size: int = 5,
    n_conv_layers: int = 2,
    n_heads: int = 2,
    epochs: int = 300,
    lr: float = 1e-2,
    alpha_lr: float = 2.0,
    weight_decay: float = 0.0,
    seed: int | None = None,
    log_every: int = 25,
    tag: str = "",
) -> dict:
    """Train a fresh model on panel `x` (n_paths, n_steps, n_assets).

    Task is contemporaneous leave-one-out reconstruction: at each (path, t,
    asset), predict x[t, asset] from cross-sectional info at t (other assets)
    and temporal info before t (own past) -- see src/model.py docstring for
    why this replaced next-step forecasting.

    Prints progress every `log_every` epochs (set to 0/None to silence) so you
    can see loss/alpha moving in real time instead of waiting on a blank
    screen. `tag` is prepended to log lines, e.g. "[rho=0.30]".

    `w` (the scalar behind alpha) is optimized with plain SGD (lr=`alpha_lr`),
    separately from the rest of the network (Adam, lr=`lr`). This is not a
    stylistic choice -- it fixes a real failure mode: at true rho=0 (no
    cross-sectional signal to find), the raw gradient on `w` is tiny
    (~1e-4, measured) but non-zero and roughly consistent in sign epoch to
    epoch. Adam normalizes updates by each parameter's own recent gradient
    magnitude, so it was amplifying that tiny, meaningless drift into
    near-full steps regardless of true signal strength -- alpha drifted from
    ~0.1 up to ~0.54-0.70 over 250 epochs at rho=0 purely from this
    artifact, with the loss barely moving throughout (classic
    overfitting-the-noise signature, not a real fit). Plain SGD's step size
    scales directly with the actual gradient, so genuinely negligible signal
    stays negligible while a real loss-reducing gradient (higher rho) still
    moves alpha.

    Returns dict with final alpha, alpha trajectory, and loss trajectory.
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_assets = x.shape[-1]
    model = GatedInterpolationModel(
        n_assets=n_assets, hidden=hidden, kernel_size=kernel_size, n_conv_layers=n_conv_layers, n_heads=n_heads
    )
    other_params = [p for name, p in model.named_parameters() if name != "w"]
    opt_main = torch.optim.Adam(other_params, lr=lr, weight_decay=weight_decay)
    opt_alpha = torch.optim.SGD([model.w], lr=alpha_lr)
    loss_fn = nn.MSELoss()

    alpha_history = []
    loss_history = []
    prefix = f"{tag} " if tag else ""

    print(f"{prefix}starting training: {epochs} epochs, panel shape {tuple(x.shape)}", flush=True)
    t0 = time.time()

    for epoch in range(epochs):
        opt_main.zero_grad()
        opt_alpha.zero_grad()
        pred = model(x)  # (P, T, N), pred[:, t, :] reconstructs x[:, t, :] itself
        loss = loss_fn(pred, x)
        loss.backward()
        opt_main.step()
        opt_alpha.step()

        alpha_history.append(model.alpha)
        loss_history.append(loss.item())

        if not torch.isfinite(loss):
            print(f"{prefix}!! loss became non-finite ({loss.item()}) at epoch {epoch} -- stopping early", flush=True)
            break

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            elapsed = time.time() - t0
            print(
                f"{prefix}epoch {epoch:4d}/{epochs}  loss {loss.item():.4f}  "
                f"alpha {model.alpha:.4f}  ({elapsed:.1f}s elapsed)",
                flush=True,
            )

    total_time = time.time() - t0
    print(f"{prefix}done in {total_time:.1f}s -- final alpha {model.alpha:.4f}, final loss {loss_history[-1]:.4f}", flush=True)

    return {
        "model": model,
        "final_alpha": model.alpha,
        "alpha_history": alpha_history,
        "loss_history": loss_history,
    }
