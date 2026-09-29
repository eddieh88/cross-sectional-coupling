# Experiment 1 — Recovery Test: Findings

Per `docs/research_spec.md` section 4, Experiment 1 trains the gated model on synthetic
panels at a grid of known `rho` values and checks whether `alpha = sigmoid(w)` tracks
true `rho` after training. This is a plumbing check, not the interesting result — the
architecture has a literal common-factor pathway matching the generator's literal
common-factor term, so it's expected to pass. The point of running it carefully is to
catch a broken setup before spending compute on Experiment 2+.

It did catch two broken things, both fixed before the result below is trustworthy.

## Bug 1 — task design had no signal to recover

The first version trained the model to forecast `x[t+1]` from history through `t`
(temporal pathway: own past; cross-sectional pathway: contemporaneous cross-section at
`t`). Under `jump_diffusion_generator.py`, log-return increments are i.i.d. over time by
construction — `rho` only shapes how assets co-move *within* a timestep, not how any
asset's future relates to its own or others' past. So next-step forecasting had zero
exploitable structure tied to `rho`, at any rho value. Symptom: loss barely left its
initial-noise level (1.02 → 1.00 over 300 epochs) and alpha drifted with no relationship
to true rho.

**Fix:** switched to contemporaneous leave-one-out reconstruction — predict `x[t, i]`
from (a) all *other* assets' values at the same `t` (cross-sectional pathway), vs (b)
asset `i`'s own strictly-past history (temporal pathway). This is the task that actually
depends on `rho`. To keep it leak-proof (asset `i` can't see its own value through the
cross-sectional pathway by any path, not just via masking), the attention query for each
asset comes from a fixed learned identity embedding rather than from the asset's own
value — see `src/model.py` docstring for the full reasoning.

## Bug 2 — Adam amplified near-zero gradients on the scalar gate into spurious drift

Even with the corrected task, `alpha` moved far more slowly than the loss converged
(loss plateaued by ~epoch 60, alpha was still crawling at epoch 500). Giving `w` (the
parameter behind `alpha`) its own higher learning rate under Adam did make it move
faster — but it also made it move at true `rho=0`, where there is no real signal to
find: alpha drifted from ~0.10 up to ~0.54–0.70 over 250 epochs while the loss barely
changed (0.999 → 0.985), the classic signature of fitting noise, not signal.

Diagnosis: raw gradient on `w` at rho=0 was consistently ~1e-4 in magnitude, same sign,
epoch after epoch. Adam normalizes updates by each parameter's own recent gradient
magnitude, so a small-but-persistent gradient gets treated the same as a large, confident
one — it doesn't distinguish "tiny and meaningless" from "tiny and real."

**Fix:** `w` is now optimized with plain SGD, separately from the rest of the network
(Adam). SGD's step size scales directly with the actual gradient, so negligible signal
stays negligible while real loss-reducing gradient (higher rho) still moves alpha.
Full reasoning in `src/train.py` docstring.

## Result (after both fixes)

Config: 10 assets, 30 independent paths, 300 timesteps/path, 250 epochs, `jump_coupling`
held fixed (`lambda_common=2.0`, `lambda_idio=1.0`), seed=0.

| rho | alpha  | final loss |
|-----|--------|-----------|
| 0.0 | 0.4732 | 0.9825 |
| 0.1 | 0.5364 | 0.9477 |
| 0.2 | 0.5481 | 0.8769 |
| 0.3 | 0.5929 | 0.7956 |
| 0.4 | 0.6422 | 0.7103 |
| 0.5 | 0.6913 | 0.6232 |
| 0.6 | 0.7375 | 0.5341 |
| 0.7 | 0.7749 | 0.4473 |
| 0.8 | 0.8045 | 0.3539 |
| 0.9 | 0.8320 | 0.2635 |
| 1.0 | 0.8552 | 0.1733 |

**Pearson corr(rho, alpha) = 0.995. Strictly monotonic.**

Raw data: `results/experiment1_recovery.csv`.

## Interpretation

- **Pass.** Alpha tracks true rho monotonically and with a strong linear correlation —
  the plumbing works, per the spec's success criterion #1.
- **Not linear, and not expected to be.** The curve is compressed toward the middle
  (0.47 → 0.86, not 0 → 1) rather than spanning the full range. This follows from the
  reconstruction task, not a shortcoming of the fit: the temporal pathway carries no real
  signal at *any* rho (this generator has no serial autocorrelation, by design), so the
  model is choosing between "one informative pathway" and "no signal at all," not
  trading off two genuinely competing signals of different strength. There was never a
  reason for alpha to fully saturate to 0 or 1 under this setup. If a cleaner 0→1 sweep
  matters later, the generator would need real temporal autocorrelation added so the
  temporal pathway has something genuine to contend with — noted in the spec's
  Experiment 3 as a robustness item worth revisiting, not required to call Experiment 1
  passed.
- Per the spec (§6, criterion #1), this is necessary but not informative on its own — it
  confirms the gate can recover a coupling parameter it was architecturally built to find.
  The real test is Experiment 2: does `alpha` beat a two-line rolling correlation
  baseline at the same task, or is this an expensive way to recompute what correlation
  already gives you.

## Next step

Per `docs/research_spec.md` §4, Experiment 2 (baseline comparison against realized
cross-sectional correlation) should run before any further architecture work.
