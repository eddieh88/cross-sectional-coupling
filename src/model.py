"""
Gated interpolation model: a single learnable scalar `alpha` interpolates
between a temporal-only pathway and a cross-sectional-only pathway.

    h     = alpha * cross_sectional(x) + (1 - alpha) * temporal(x)
    alpha = sigmoid(w)   # w: one trainable scalar

Task: CONTEMPORANEOUS LEAVE-ONE-OUT RECONSTRUCTION, not forecasting.
For each asset i at time t, predict x[t, i] from:

- TemporalPathway: asset i's own STRICTLY PAST history x[<t, i]. No
  cross-asset mixing, ever.
- CrossSectionalPathway: all OTHER assets' values at the SAME timestep,
  x[t, j != i]. No temporal history beyond the current step, and asset i's
  own value is structurally excluded (see leak-proofing note below).

Why reconstruction, not forecasting (this was a real bug, not a style
choice): `jump_diffusion_generator.py` draws each timestep's shock i.i.d.
over time -- there is no serial autocorrelation in log-returns by
construction. `rho` only governs how assets co-move WITHIN a timestep. A
"predict x[t+1] from history through t" objective therefore has zero
exploitable signal tied to rho, at any training horizon -- an earlier
version of this model targeted next-step forecasting and alpha drifted
aimlessly with the loss barely leaving its initial-noise level, which is
exactly what you'd expect when the task has no learnable structure at all.
Contemporaneous reconstruction is the objective that actually depends on
rho: the cross-sectional pathway's achievable loss strictly improves as rho
increases, while the temporal pathway's achievable loss stays at "predict
the unconditional mean" regardless of rho (same i.i.d.-over-time issue,
just now correctly reflecting a genuinely uninformative pathway rather than
masquerading as a broken forecasting task).

Leak-proofing note: the cross-sectional pathway's attention QUERY for asset
i is drawn from a fixed, learnable per-asset identity embedding -- NOT from
x[t, i] itself. Combined with masking asset i out of its own key/value set,
this makes the pathway's output a function of {x[t, j] : j != i} ONLY, by
construction. (A design where the query is built from x[t, i] would let the
attention WEIGHTS depend on the target even if the aggregated VALUES don't
-- a subtle leak. Using an identity embedding for the query closes that off
entirely rather than relying on masking alone.)

Expected shape of the recovery curve: because the temporal pathway carries
no real signal at all (not just "less" signal than cross-sectional), alpha
is expected to rise toward 1 fairly quickly as rho increases off 0, rather
than tracing a clean straight line across the full range. Check the
recovered curve for MONOTONICITY, not linearity.
"""

import torch
import torch.nn as nn


class TemporalPathway(nn.Module):
    """Causal conv over time, shifted so features at position t depend only on x[<t]."""

    def __init__(self, hidden: int, kernel_size: int = 5, n_layers: int = 2):
        super().__init__()
        layers = []
        in_ch = 1
        for _ in range(n_layers):
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size, padding=kernel_size - 1))
            layers.append(nn.ReLU())
            in_ch = hidden
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        # x: (B, T, N) -> treat each (batch, asset) as an independent 1-channel series
        b, t, n = x.shape
        h = x.permute(0, 2, 1).reshape(b * n, 1, t)  # (B*N, 1, T)
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Conv1d):
                h = h[..., :t]  # trim right padding -> h[..., i] depends on x[0..i] (causal)
        h = h.reshape(b, n, -1, t).permute(0, 3, 1, 2)  # (B, T, N, hidden), h[:, i] depends on x[<=i]

        # shift right by one so the feature USED to predict x[:, i] depends only on x[<i]:
        # drop the last (most-current) step and prepend a learned "no history yet" state (zeros).
        pad = torch.zeros_like(h[:, :1])
        h = torch.cat([pad, h[:, :-1]], dim=1)
        return h  # (B, T, N, hidden), h[:, t] depends only on x[<t]


class CrossSectionalPathway(nn.Module):
    """Leave-one-out attention over assets at a single timestep.

    Query for asset i comes from a fixed per-asset identity embedding (not
    from x[t, i]), and asset i is masked out of its own key/value set -- so
    the output for asset i is provably a function of {x[t, j] : j != i}
    only. See module docstring for why this matters.
    """

    def __init__(self, hidden: int, n_assets: int, n_heads: int = 2):
        super().__init__()
        self.n_assets = n_assets
        self.identity_embed = nn.Embedding(n_assets, hidden)  # query source, data-independent
        self.value_embed = nn.Linear(1, hidden)  # key/value source, per-asset realized value
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

        # leave-one-out mask: (N, N), row i has -inf at column i (asset i can't attend to itself)
        mask = torch.zeros(n_assets, n_assets)
        mask.fill_diagonal_(float("-inf"))
        self.register_buffer("loo_mask", mask)

    def forward(self, x):
        b, t, n = x.shape
        assert n == self.n_assets, f"model built for {self.n_assets} assets, got {n}"

        kv = self.value_embed(x.reshape(b * t, n, 1))  # (B*T, N, hidden) -- keys/values from data
        q = self.identity_embed.weight.unsqueeze(0).expand(b * t, -1, -1)  # (B*T, N, hidden) -- data-independent

        attn_out, _ = self.attn(q, kv, kv, attn_mask=self.loo_mask)
        h = self.norm(attn_out)  # no residual to `q` or `kv` on purpose: keeps output leak-proof
        h = h.reshape(b, t, n, -1)  # (B, T, N, hidden)
        return h


class GatedInterpolationModel(nn.Module):
    def __init__(
        self,
        n_assets: int,
        hidden: int = 16,
        kernel_size: int = 5,
        n_conv_layers: int = 2,
        n_heads: int = 2,
    ):
        super().__init__()
        self.temporal = TemporalPathway(hidden, kernel_size, n_conv_layers)
        self.cross_sectional = CrossSectionalPathway(hidden, n_assets, n_heads)
        self.w = nn.Parameter(torch.zeros(1))  # alpha = sigmoid(w), init alpha=0.5
        self.head = nn.Linear(hidden, 1)

    @property
    def alpha(self) -> float:
        return torch.sigmoid(self.w).item()

    def forward(self, x):
        # x: (B, T, N) log-returns. Reconstructs x[:, t, i] for every (t, i).
        temp = self.temporal(x)          # (B, T, N, hidden), depends only on x[<t, i]
        cross = self.cross_sectional(x)  # (B, T, N, hidden), depends only on x[t, j != i]
        a = torch.sigmoid(self.w)
        h = a * cross + (1 - a) * temp
        pred = self.head(h).squeeze(-1)  # (B, T, N)
        return pred
