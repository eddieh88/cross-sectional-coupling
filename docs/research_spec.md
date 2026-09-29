# Research Spec: Learned Interpolation Scalar as a Cross-Sectional Coupling Estimator

## 1. Core Idea

Build a neural architecture with a **learnable scalar gate** `alpha` that interpolates
between two mixing pathways over a multivariate time series panel:

- **Temporal pathway**: mixes each asset's own history over time (no cross-asset information).
- **Cross-sectional pathway**: mixes across assets at a fixed point in time (no temporal
  information beyond the current step).

```
h = alpha * cross_sectional(x) + (1 - alpha) * temporal(x)
alpha = sigmoid(w)   # w: a single trainable scalar parameter
```

The hypothesis: once trained on a forecasting (or reconstruction) objective, `alpha`
becomes a data-driven estimate of **how much of the data's structure is cross-sectionally
coupled vs. purely temporal/idiosyncratic** — a signal not directly observable in real
markets.

This connects to existing work: prior Deribit jump-risk research (jump-risk decomposition via
bipower variation, diffusive vs. jump components) currently has no cross-sectional
dimension. If `alpha` (or an analogous jump-coupling estimate) proves reliable, it's a
natural extension — e.g. detecting when jump risk is systemic vs. idiosyncratic across
multiple instruments, not just decomposing variance for a single one.

## 2. Why This Might Matter (if validated)

- **A nonlinear, learned alternative to realized correlation.** Rolling correlation is
  linear and pairwise; `alpha` comes from whatever the cross-sectional pathway finds
  predictive, potentially capturing nonlinear/higher-order coupling correlation misses.
- **Regime / systemic-risk signal.** Classic stylized fact: correlations spike in crises.
  A rolling `alpha` could serve as a real-time coupling gauge, potentially reacting
  faster than a rolling-window correlation estimate.
- **Joint signal with jump risk.** Tracking `alpha` alongside the existing jump-share
  decomposition could reveal whether shocks are transmitting cross-sectionally, which
  neither measure alone currently shows.
- **Portfolio/hedging input.** A real-time coupling estimate is directly relevant to
  diversification and sizing decisions.
- **Cross-market comparison.** Compare `alpha` across crypto pairs/exchanges or vs.
  equities as a market-integration/fragmentation measure.

**Important caveat, stated up front:** none of this is validated yet. The entire plan
below exists to find out whether `alpha` is a real signal or an elaborate way of
recomputing correlation. Treat every claim above as a hypothesis to be killed, not a
result.

## 3. Synthetic Data Generator (already built)

A multivariate jump-diffusion generator with two **independently tunable** cross-sectional
coupling knobs, so ground truth is known and controllable:

- `rho` in [0,1]: diffusive coupling. Log-return increments are
  `sigma * (sqrt(rho) * common_shock + sqrt(1-rho) * idiosyncratic_shock)`.
  rho=0 → independent random walks. rho=1 → single common Brownian motion.
- `jump_coupling = lambda_common / (lambda_common + lambda_idio)`: fraction of jump
  intensity that is systemic (fires across all assets simultaneously) vs. idiosyncratic
  (independent per asset).

Already implemented and sanity-checked: realized correlation tracks `rho` monotonically
(0.02 → 0.80 realized corr as rho sweeps 0.0 → 1.0), and `jump_coupling` behaves
independently of `rho` as designed. Script: `jump_diffusion_generator.py`.

**Known limitation to keep in mind:** this generator has a single common factor and
Gaussian jump sizes. It is a stylized proxy for real coupling structure, not a claim
about how real markets actually work. Results here validate the *mechanism*, not the
*real-data numbers*.

## 4. Experiment Plan

### Experiment 1 — Recovery test (sanity check, not the finding)

Train the gated model on synthetic panels generated at a grid of known `rho` values
(e.g. 0.0, 0.1, ..., 1.0), holding `jump_coupling` fixed. After training, read off
`sigmoid(w)` and check whether it tracks true `rho` — ideally monotonically, ideally
close to linearly.

- **Expectation**: it should work, close to by construction (the architecture has a
  literal common-factor pathway matching the generator's literal common-factor term).
  A pass here is necessary but not interesting on its own — it confirms the plumbing
  isn't broken.
- **Failure mode to watch for**: `alpha` saturates to 0 or 1 regardless of `rho`, or is
  flat/noisy — would indicate the gate isn't learning the intended signal at all.

### Experiment 2 — Baseline comparison (the actual first real test)

Compare the learned `alpha` against a **trivial baseline**: rolling/realized
cross-sectional correlation computed directly on the same synthetic data, no NN involved.

- **Question**: does `alpha` recover `rho` any better, faster (fewer samples), or more
  robustly (under jump contamination) than the two-line correlation calculation?
- **If the baseline matches `alpha`'s performance**: the NN machinery isn't earning its
  complexity for this question — that's a real, useful negative result, not a failure of
  the exercise.
- **If `alpha` beats the baseline**: identify specifically where/why (fewer samples
  needed? more robust to jump noise? disentangles diffusive vs. jump coupling that
  correlation conflates?) — that's the actual finding worth pursuing.

This should run before any further architecture work. It's the fastest way to find out
if there's a real result here.

### Experiment 3 — Robustness / stress tests

Test conditions where the model's assumptions and the generator's assumptions diverge,
since Experiment 1 only tests the case where they match:

- **Time-varying `rho`**: regime-switching coupling within a single simulated series
  (e.g. low rho, then a shift to high rho mid-series, mimicking a correlation
  breakdown). Does a rolling-window version of `alpha` track the shift, with what lag?
- **Asymmetric coupling**: assets pull together in drawdowns but not in rallies (more
  realistic than the symmetric single-factor structure currently implemented).
- **More than one common factor**: does a single scalar `alpha` degrade gracefully or
  become meaningless when the true structure is multi-factor?
- **Disentanglement check**: does `alpha` (or a jump-specific analog) separately track
  `rho` and `jump_coupling`, or does it conflate "coupled in general" without
  distinguishing diffusive vs. jump-driven coupling? Relevant given the jump-risk
  research angle.
- **Sample efficiency / robustness**: performance vs. panel size (fewer assets), window
  length (shorter series), and noise level.
- **Architecture choice**: does this hold across the literature arc (TCN → DLinear →
  PatchTST → iTransformer → ModernTCN → KDA), or is recoverability specific to one
  architecture's mixing mechanism? Which is most sample-efficient at it?

### Experiment 4 — Generalization across generators

Validate against a structurally different synthetic generator (e.g. copula-based or a
simple agent-based market simulation), not just the jump-diffusion generator used in
Experiment 1–3. A result that only holds for one generator is a narrow correspondence,
not a general property of the method.

### Experiment 5 — Real market application (only after 2–4 hold up)

Move to real data once the mechanism is validated on synthetic ground truth. Since real
data has no ground truth, validation here is necessarily **indirect**:

- Estimate rolling `alpha` on real OHLC/microstructure panels (start with free daily/
  hourly bars across a small crypto universe; move to order-book-snapshot data later if
  useful).
- Check whether `alpha` spikes around known crisis/high-correlation periods (indirect
  proxy validation — e.g. does it track known market-stress windows, or move with
  DVOL/skew-type measures already in use).
- Cross-reference against the two identified crash events in the existing jump-risk
  work: does `alpha`'s behavior around those events add information beyond what
  DVOL/skew already show? This is the most meaningful real-data test available, since
  it's checked against a known, already-studied episode rather than a vague prior.
- Compare `alpha` to realized correlation and to the bipower-variation jump/diffusive
  split on the same real windows — does it add anything, or just re-derive what's
  already computed?

**Framing shift required at this stage**: on synthetic data, `alpha` is validated
against a literal generating parameter. On real data, there's no single stationary DGP,
so the claim necessarily weakens to "a data-driven estimate of how much predictive power
comes from cross-sectional structure vs. own-history," not a literal parameter estimate.
Treat real-data `alpha` as an **ordinal/relative signal** (more vs. less coupled periods)
until proven otherwise — don't trust absolute levels by analogy to synthetic `rho`.

**Known confounds on real data** (synthetic validation does not protect against these):
shared exposure to a common tradable factor (e.g. BTC dominance), liquidity commonality,
overfitting to a few dominant assets in the panel. Any real-data result needs to rule
these out before being trusted as "coupling" in the intended sense.

## 5. Compute / Infrastructure

- Experiments 1–3 (single-gate shallow model, small synthetic panels: ~10 assets,
  thousands of timesteps): trivial compute, seconds-to-minutes per run. Run locally on
  CPU (2019 16" MacBook Pro, 8-core i9, 32GB RAM — no usable GPU on this Intel machine;
  MPS is Apple-Silicon-only and the AMD GPU isn't usable for PyTorch on macOS).
- Experiment 3's full architecture sweep (6 architectures × multiple rho/jump_coupling
  values × multiple seeds) and Experiment 5 (real architectures like iTransformer/
  ModernTCN/KDA at realistic depth on longer real histories) is where a cloud GPU
  becomes worth it — compute adds up from the sweep size, not from any single run being
  heavy.
- Recommendation: develop and run Experiments 1–2 fully locally first (fast iteration,
  no cost). Move to cloud GPU only once scaling into Experiment 3's full sweep or
  Experiment 5's real architectures.

## 6. Success Criteria (what would make this worth continuing)

In rough order of what should be checked, cheapest/most-informative first:

1. Experiment 1 passes (necessary, not sufficient — expected to pass, low information
   value on its own).
2. Experiment 2 shows `alpha` outperforms the trivial correlation baseline in some
   identifiable way (sample efficiency, robustness to jump noise, or disentanglement of
   diffusive vs. jump coupling). **If this fails, seriously reconsider continuing** —
   without this, the project is a more expensive way to compute something correlation
   already gives you.
3. Experiment 3 shows graceful (not catastrophic) degradation under model-misspecified
   conditions, and some disentanglement between diffusive and jump coupling.
4. Experiment 4 shows the result isn't an artifact of one specific generator.
5. Experiment 5 shows `alpha` correlates sensibly with known proxies/events on real
   data, ideally adding information beyond what DVOL/skew/bipower-variation already
   provide for the jump-risk research specifically.

## 7. Open Questions to Resolve While Building

- Global scalar `alpha` vs. per-layer vs. per-channel: start global (one interpretable
  number per trained model) for Experiments 1–2; revisit if deeper architectures in
  Experiment 3 make a single global gate too restrictive.
- For time-varying real-world use (Experiment 5), should `alpha` be re-estimated on
  rolling windows, or made a function of recent conditions (e.g. output by a small
  side-network conditioned on recent volatility/volume) rather than a single free
  parameter trained once? A single global `alpha` trained on a long real window gives
  only a time-averaged number, which is likely too weak a signal to be useful as-is.
- What's the right sub-mixer implementation for the temporal and cross-sectional
  pathways — literature-arc architectures (TCN-style causal conv for temporal,
  iTransformer-style attention-over-assets for cross-sectional) are the natural
  starting point, reusing existing familiarity with that arc.

## 8. Explicit Non-Goals (for now)

- Not trying to beat SOTA forecasting benchmarks (ETT/Electricity/Traffic/Weather) —
  those are useful only as an implementation-correctness check for the base
  architectures, not the object of study here.
- Not committing to tick-level real data yet — start with OHLC/snapshot-interval data;
  full tick/LOB capture is a later step if snapshot-level signal proves insufficient.
- Not claiming any causal interpretation of `alpha` on real data without the indirect
  validation in Experiment 5 — correlation with known events is the bar, not a priori
  assumption of validity.
