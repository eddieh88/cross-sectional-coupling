# Learned Coupling Gate vs. Classical Estimators

Can a neural network with a single learnable gate estimate **how strongly a panel of assets
moves together** (cross-sectional coupling) better than classical statistics can?

This repo builds that model, generates synthetic data where the true coupling is known,
and tests the model against classical estimators across 10 experiment rounds, then
checks the assumptions against 3 years of real crypto data.

**Short answer: no.** A classical, untrained estimator matched or beat the learned gate
on every task. The one neural-network win (a separate GRU, Exp 8–10) relied on supervised
training against ground-truth coupling, which real markets don't provide. It also failed
to generalize to the real correlation curve. Negative results are reported as-is, along
with the confounds found and corrected along the way.

## The model

```
h     = alpha * cross_sectional(x) + (1 - alpha) * temporal(x)
alpha = sigmoid(w)          # w: one trainable scalar
```

- **Cross-sectional pathway:** leave-one-out multi-head attention over assets at a single
  timestep (an asset can't attend to itself, so there's no target leakage).
- **Temporal pathway:** causal 1-D convolution over each asset's own history, shifted so
  the output at time `t` only sees `x[<t]`.
- **Hypothesis:** after training, `alpha` measures how much of the data's structure is
  cross-sectional vs. temporal.

See [`src/model.py`](src/model.py) and the original plan in
[`docs/research_spec.md`](docs/research_spec.md).

## Synthetic data

[`jump_diffusion_generator.py`](jump_diffusion_generator.py) is a multivariate
jump-diffusion with two coupling knobs you can set independently:

| knob | meaning |
|---|---|
| `rho` ∈ [0, 1] | diffusive coupling (weight on a common Brownian shock) |
| `jump_coupling` ∈ [0, 1] | fraction of jump intensity that is systemic vs. idiosyncratic |

It also supports multi-factor structure (`factor_dispersion`), injected nonlinear lag
structure, and a time-varying `rho_path` for the regime experiments.

## Results

| # | question | outcome |
|---|---|---|
| 1 | Does `alpha` recover the true `rho`? | Yes: Pearson r = 0.995, monotonic. This is a plumbing check, and it is close to guaranteed by construction. |
| 2 | Does it beat plain realized correlation? | **No.** The baseline reaches r ≈ 1.000 with 250 obs/asset; `alpha` gets 0.866 at the same size. Both are robust to jump noise. |
| 3 | Can it separate jump coupling from diffusive coupling? | **No.** The single gate mixes the two up. Per-asset jump tests (BNS / Huang-Tauchen) can't tell them apart at high `rho` either, because of a structural identifiability limit. |
| 3d–3f | Can a classical pipeline do it? | A pooled cross-sectional jump statistic reaches >90% detector recall. Classifying each flagged event as jump or diffusive uses a shape (R²) feature. |
| 4a | Does that survive multi-factor structure? | Detection does. Classification didn't: specificity on flagged candidates was 5% at `rho`=0.3, which the earlier aggregate metrics hid. |
| 4b | Fix | Adding a magnitude feature to a linear classifier gives 93.5–98% held-out accuracy. It needs a small amount of labeled calibration data. |
| 5 | Can the temporal pathway learn nonlinear lag structure? | It learns real structure (R² 0.131) but doesn't match threshold-AR(1) (0.145), even with more capacity. |
| 6 | Real data: 30 Coinbase assets, 3 years daily | Close to rank-1 (top eigenvalue = 66% of variance). Coupling varies a lot over time (rolling corr 0.31–0.89). Flagged jump days match real market events. No daily AR structure. |
| 7 | How well do classical tools track time-varying `rho`? | EWMA Pareto-dominates rolling windows. A CUSUM change-point detector didn't calibrate cleanly. |
| 8 | Adaptive GRU vs. EWMA (supervised on true `rho(t)`) | GRU noise is 15× lower at matched lag. Caveat: it had privileged access to ground truth. |
| 9 | Same test on a real-shaped `rho(t)` curve | The apparent GRU win came from memorizing the curve: it collapsed on a time-reversed curve (R² 0.50 → 0.04). |
| 10 | Diverse random curves (no memorization possible) | GRU R² 0.67 vs. LightGBM 0.63 vs. EWMA 0.03. **Neither flexible model generalizes to the real curve (GRU −0.23).** |

The full write-up, including every bug and confound caught before results were reported,
is in [`results/SUMMARY.md`](results/SUMMARY.md). Per-experiment findings are in
`results/experiment*_findings.md`. The pre-registration docs (`*_prereg.md`) were written
before any code for that experiment was run.

## Repository layout

```
├── jump_diffusion_generator.py   # synthetic multi-asset jump-diffusion (ground truth)
├── src/
│   ├── model.py                  # GatedInterpolationModel, TemporalPathway, CrossSectionalPathway
│   ├── train.py                  # training loop; reads out alpha
│   ├── data.py                   # panel construction / windowing
│   └── baseline.py               # realized correlation, bipower variation, BNS jump tests
├── experiments/                  # one script per experiment (1 → 10), runnable standalone
├── results/                      # findings write-ups, pre-registrations, CSV outputs
├── notebooks/                    # model walkthrough + Exp 2/3 notebooks (and generators)
├── data_real/
│   ├── fetch_coinbase_panel.py   # pulls the real panel from Coinbase's public API (no key)
│   ├── coinbase_daily_panel_3y.parquet   # 30 assets, 2023-08-28 → 2026-08-27, daily closes
│   └── real_rho_curve.npy        # smoothed 30-day rolling mean correlation from the panel
└── docs/research_spec.md         # original research plan and kill criteria
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/experiment1_recovery.py          # any experiment runs standalone
python experiments/experiment6_real_data_analysis.py
python data_real/fetch_coinbase_panel.py            # optional: refresh the real panel
```

Developed with Python 3.12 and PyTorch 2.2 on CPU. Most experiments finish in minutes.
The training-heavy ones (2a, 5c, 8–10) take longer.

**Notes**
- On macOS, the experiments that use both LightGBM and PyTorch import `lightgbm` first
  on purpose. Importing torch first loads two OpenMP runtimes and segfaults.
- `fetch_coinbase_panel.py` pulls the most recent 3 years, so re-running it gives a
  different window than the committed panel.
- `real_rho_curve.npy` is committed as a data artifact. It is the Exp 6 rolling mean
  correlation, lightly smoothed. The smoothing step isn't scripted in this repo.

## Limitations

- The synthetic generators are built from standard stochastic building blocks that
  classical statistics already has near-optimal estimators for. The negative results
  are strongest for that family of generators.
- The real-data work uses **daily** bars. Microstructure effects that a temporal
  pathway could exploit would more likely show up at intraday frequency.
- The adaptive-tracking results (Exp 8–10) depend on supervised ground truth. Building
  a training distribution that matches how real coupling moves is still an open problem.
