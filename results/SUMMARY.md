# Project Summary — Learned Coupling Gate vs. Classical Alternatives

Full record of Experiments 1–10, run against the plan in `docs/research_spec.md`. Kept as
a standing reference: what was tested, in what order, what broke, what was corrected,
and where the investigation currently stands.

## The question

Can a learnable scalar gate (`alpha = sigmoid(w)`, interpolating a cross-sectional
attention pathway against a temporal causal-conv pathway — `src/model.py`) serve as a
useful, data-driven estimator of cross-sectional coupling in multi-asset time series —
diffusive coupling (`rho`) and/or jump-timing coupling (`jump_coupling`) — better than
classical statistics already give you for free? Tested entirely on synthetic
jump-diffusion data (`jump_diffusion_generator.py`) with known ground truth, per the
spec's explicit instruction to validate the mechanism before any real-market work, and
to treat every claim as "a hypothesis to be killed, not a result."

## Executive summary

**No.** Across six rounds of testing — coupling recovery, baseline comparison, jump/
diffusive disentanglement, shape-signal classification, joint detection, and a
multi-factor stress test — a classical, non-trained pipeline matched or beat the NN
gate at every task it was asked to do, including two tasks (jump disentanglement, joint
detection) that looked at first like they might require it. The one real requirement
the classical pipeline picked up along the way is **modest labeled calibration data**
for its final classification step — a data problem, not an architecture problem.

Two of the six rounds directly targeted and corrected an overstated earlier finding
(caught internally, before being reported as settled) — evidence this was a genuine
stress test, not a search for a predetermined conclusion.

## Experiment 1 — Recovery test (sanity check)

**Setup:** train the gated model at a grid of known `rho` (jump_coupling held fixed);
check whether `alpha` tracks `rho`.

**Two bugs found and fixed before the result was trustworthy:**
1. The original forecasting-task design (`x[t+1]` from history through `t`) had *zero*
   exploitable signal — this generator's returns are i.i.d. over time by construction,
   so `rho` (a purely cross-sectional, same-timestep effect) has no relationship to
   next-step prediction. Fixed by switching to contemporaneous leave-one-out
   reconstruction (predict `x[t,i]` from other assets at `t`, vs. asset `i`'s own past).
2. Adam amplified a tiny, consistent, but meaningless gradient on the scalar gate into
   spurious drift at `rho=0` (alpha drifted 0.10→0.54-0.70 while loss barely moved — the
   overfitting-noise signature). Fixed with a separate plain-SGD optimizer for the gate
   parameter, since SGD's step size scales with the actual gradient magnitude.

**Result:** Pearson corr(rho, alpha) = **0.995**, strictly monotonic (alpha ranges
0.47→0.86 across rho 0.0→1.0 — compressed, not full 0→1, because the temporal pathway
carries no real signal at any rho in this generator, so the gate is choosing between
"one informative pathway" and "nothing," never fully saturating).

**Verdict:** pass, as expected — confirms the plumbing works. Not informative on its
own; the spec is explicit that Experiment 2 is the real test.

## Experiment 2 — Baseline comparison ("the actual first real test")

**Setup:** compare `alpha` against `src/baseline.py::realized_corr_estimate` — pooled
realized cross-sectional correlation, two lines of code, no training — on the *same*
generated data, across sample sizes (2a) and jump-noise levels (2b).

**2a — sample efficiency:** baseline hits Pearson r ≈ 1.000 with only 250 observations
per asset (smallest panel tested); alpha needs far more data to even approach that
(0.866 at the same panel size → 0.994 at 30×300) and never clearly surpasses it.

**2b — jump-noise robustness:** both estimators essentially unaffected by 3x jump-noise
in rank-correlation terms; baseline still marginally ahead in both conditions.

**Verdict — first negative result:** the trivial baseline matches or beats `alpha` on
both axes tested, with far less compute. Per the spec's own criterion #2, this is the
result that should trigger "seriously reconsider continuing" — the recommendation was to
test disentanglement (a dimension correlation cannot address at all by construction)
before abandoning the idea, since that's the cheapest remaining place for a real edge.

## Experiment 3 — Disentanglement (jump coupling vs. diffusive coupling)

**Phase A.5 — does the existing single-gate model conflate the two mechanisms?**
Held `rho=0.5`, swept `jump_coupling` 0.2→0.8. **r(alpha, jump_coupling) = -0.041** —
alpha is essentially blind to jump structure, not disentangling it. Diagnosed cause:
jump events are rare (~3-4 per path) against thousands of ordinary timesteps; under an
aggregate MSE loss, jump-timing coupling gets diluted into irrelevance.

**Phase A — classical disentanglement, naive jump test.** Bipower-variation-scrubbed
`diffusive_corr` cleanly recovers `rho` (r=1.000) and is nearly independent of
`jump_coupling` (r=0.070). But `jump_corr` (co-occurrence of naive jump flags) is badly
contaminated by `rho` (r=0.818) — not clean disentanglement.

**Phase A retest — properly calibrated BNS/Huang-Tauchen ratio test**, to rule out "the
naive threshold just has an uncontrolled false-positive rate" before concluding
anything structural. Sanity-checked on pure noise first (naive test: 7 spurious flags on
2000 steps; calibrated test: 0, and 3 injected jumps recovered exactly). Re-ran the
disentanglement grid: **contamination barely moved (r=0.818 → 0.803)** despite the
calibrated test roughly halving raw `jump_corr` magnitudes. Better calibration fixed the
false-positive *rate*, not the *contamination*.

**Why — a structural identifiability limit, not a calibration bug:** at high `rho`,
every asset's diffusive shock is dominated by the same shared factor, so a single large
purely-diffusive draw and a genuine coordinated jump are close to observationally
identical *per asset* at daily sampling frequency. Both naive and calibrated tests
decide "is this a jump" per-asset, independently, then correlate flags after the fact —
but the confound lives in the *joint* cross-sectional pattern, which no per-asset test,
however well-calibrated, can see.

**A concrete idea this pointed to:** a shared diffusive draw moves every asset in a
fixed ratio proportional to its own `sigma_i` (rank-1 structure); a common jump draws an
independent size per asset even when timing is shared. That shape difference requires
looking at the whole cross-section jointly — setting up Experiments 3d–3f.

**Verdict:** three results of increasing specificity — NN doesn't beat correlation at
all (Exp2); NN doesn't disentangle jump/diffusive coupling, it's just blind to jumps
(3A.5); a properly calibrated classical test can't disentangle them either, for an
understood structural reason, not lack of effort (3A retest).

## Experiments 3d/3e — Shape-signal check and first classical pipeline

**3d — does the shape signal exist at all?** Using ground-truth jump timing and true
`sigma_i` (best case, isolating the shape question from detection noise): regressed each
event's return vector against known `sigma_i`. Diffusive events fit well (R² 0.32→0.88
as rho 0.3→0.9); jump events don't (R² ~0.05 flat, regardless of rho). Clean separation,
even in a single-path illustration with only 3 real jump events.

**3e — does it survive estimated sigma and a real detector?** Swapping true `sigma_i`
for a bipower-variation estimate: **no degradation** (0.049→0.050 jump R² true→est at
rho=0.3, similarly negligible elsewhere). Then built a full pipeline: per-asset BNS
flags → cross-sectional candidate events (≥10% of assets flagged at once) → R²-vs-sigma
threshold classifier.

First attempt at a 30% co-occurrence threshold gave a "perfect" 1.000 accuracy —
**caught before being reported**: only 1-4 candidates were detected against ~104 true
jump events, an artifact of the detector almost never firing, not evidence of a working
pipeline. Loosened to 10%: **detector recall 51.9-56.7%, precision 93.7-98.2%,
classifier recall among candidates 100%.**

**Verdict at this point:** classification looked solved; detection (a per-asset
co-occurrence test misses ~45-48% of true jumps, since jump sizes are drawn
independently per asset and weak individual draws slip under a strict per-asset
threshold) was identified as the real, specific bottleneck.

## Experiment 3f — Joint detector

**Setup:** replace independent per-asset thresholding with one pooled statistic per
timestep — `J[t] = sum_i (r[t,i]/sigma_hat_i)^2` — against a robust
`median + k*MAD` threshold.

**Result: detector recall reaches 100% at loose thresholds (k=3-8) and stays above 90%
through k=20**, trading cleanly against precision (k=10: ~99-100% recall at 65-76%
precision; k=15: ~91-98% recall at 78-87% precision). This closed the exact detection
gap 3e identified — completely, not partially.

**Verdict at this point (later corrected — see Exp4a): "fourth negative result for the
NN, consolidate rather than build Phase B."** A complete classical two-stage pipeline
(joint detect → R²-vs-sigma classify) appeared to exist with no training required.

## Experiment 4a — Multi-factor stress test (and a correction to 3d/3e/3f)

**Motivation:** every result above was validated on a generator with a single, rank-1,
time-invariant common factor — exactly the untested assumption the spec's own
Experiment 3 ("more than one common factor") and Experiment 4 ("structurally different
generator") sections flagged. Extended `jump_diffusion_generator.py` with a
`factor_dispersion` parameter: a second independent common factor with heterogeneous,
fixed per-asset loadings (`factor_dispersion=0` is exactly backward-compatible with
everything above; verified bit-for-bit identical output).

**Result 1 — the joint detector is robust to multi-factor structure**: recall stays
97.5-100% across every dispersion level. Pooling squared standardized moves doesn't
care how many factors drive them.

**Result 2 — the shape signal (R² vs. sigma) genuinely degrades under multi-factor
structure**, as expected: diffusive R² collapses toward the jump baseline as dispersion
increases (e.g. rho=0.9: 0.877→0.154).

**Result 3 — the real surprise, and the correction:** checking *specificity* (correctly
labeling a true diffusive **candidate** — one an actual detector flagged, not a
ground-truth-selected extreme event — as diffusive) instead of the aggregate "precision"
metric reported in 3e/3f revealed the classifier was never as solved as claimed, **even
at `factor_dispersion=0`, the original single-factor case**: at rho=0.3, specificity was
just **5.2%** (verified same story with 3e's original BNS-based detector: 0%). This was
hidden in the earlier aggregate metrics because the detector-flagged candidate pool is
jump-heavy at low rho, so trivially-easy jump-class recall dominated the aggregate
stats — the same *shape* of mistake as the earlier caught "1-4 sample, 1.000 accuracy"
trap, one level more subtle (sample size was fine this time; class imbalance was the
problem). Swept every possible threshold directly on the real candidate population:
at rho=0.3, best achievable accuracy is only 68.5% (genuine distributional overlap, not
a calibration bug); at rho=0.9 it's 88.4% at the correctly-calibrated threshold (~0.23,
not the 0.5 used throughout 3e/3f).

**Corrected verdict:** classification was solved only for the most extreme diffusive
events (what 3d's ground-truth-matched test happened to sample), not for the real
population a detector flags. The remaining gap: moderate/borderline events, especially
at low-to-moderate rho.

## Experiment 4b — Fix: adding a magnitude feature

**Motivation:** R² tests *shape* but discards *scale*. Jump sizes are drawn from a
fixed distribution independent of rho; a genuinely common diffusive event's typical
size scales with `sqrt(rho)` — a large, unused discriminating signal, especially at low
rho where diffusive candidates are necessarily small.

**Result:** a linear classifier on `[R², mean(|return|)]`, evaluated on a held-out
split, reaches **93.5-98.0% accuracy across every rho and every multi-factor dispersion
level tested** — including the rho=0.3 case (68.5% ceiling) and the multi-factor stress
that broke shape-alone. Verified not overfitting (held-out ≈ in-sample accuracy).

**Caveat, precisely characterized (after an initial imprecise version was itself
corrected):** a fully unsupervised version (2-cluster k-means, no ground-truth labels
used to fit) is reliable at the rho extremes (~83-96%, where one feature dominates
cleanly enough that k-means' own objective happens to align with the true label
boundary) but has a genuine, dispersion-independent weak spot at moderate rho (~0.6):
even best-of-20-restarts k-means only reaches ~75-80% there — verified this is a real
structural limit of unsupervised clustering on these two features, not seed-instability
(a first-pass single-seed read had wrongly attributed the weak spot to a rho×dispersion
interaction; multi-seed testing showed it's rho alone, dispersion-independent).

**Verdict:** the classification gap is fixable classically, but the fix requires modest
labeled calibration data — a materially different requirement than the fully
assumption-driven pipeline used everywhere upstream of this step. Still nowhere near
NN territory (a 3-parameter linear classifier), and it resolves the gap decisively
wherever tested. Strengthens "don't build Phase B," while correcting *why*.

## Current complete pipeline (fully classical, no NN)

```
joint detection statistic (sum of squared standardized cross-sectional moves,
robust per-path threshold)
        -> candidate coordinated-move timesteps
        -> [R^2-vs-estimated-sigma, mean(|return|)] linear classifier
           (fit on a modest labeled calibration set)
        -> jump vs. diffusive label
```

## Overall verdict

**The learned scalar gate never earned its complexity, at any point in this
investigation.** Six rounds of testing — including the one round (4a/4b) specifically
designed to break the earlier "consolidate" conclusion rather than confirm it — found a
classical alternative that matched or beat it at every task: coupling recovery, sample
efficiency, jump-noise robustness, jump/diffusive disentanglement, event detection, and
event classification. Two rounds directly corrected earlier findings that turned out to
be overstated on closer inspection (3e/3f's "perfect" pipeline result before honest
recall accounting; the same pipeline's real classification failure hidden by class
imbalance) — evidence this was a genuine search for a real answer, not a foregone
conclusion.

**Caveats that remain, honestly stated:**
- All generators tested (single-factor and two-factor) are still static/time-invariant
  in their loading structure. Time-varying coupling, asymmetric/regime-dependent
  coupling, and genuinely nonlinear interaction structures are untested.
- Every generator tested has i.i.d.-over-time returns by construction — the model's
  *temporal* pathway has never once been exercised against data where it has real
  signal to find. All six rounds tested the cross-sectional pathway exclusively.
- The classification step of the classical pipeline needs modest labeled calibration
  data; real markets don't hand that out for free (though a small labeled set might be
  bootstrapped from known crash episodes or prior jump-risk decomposition work).
- `rho` itself isn't observable on real data — not a problem for the pipeline above
  (it doesn't use rho directly), but relevant to any future rho-adaptive extension.

## Experiment 5 — the temporal pathway (5a/5b/5c)

The one axis of the original spec never exercised: every generator through Exp4b had
i.i.d.-over-time returns, so the model's *temporal* pathway (own-history causal conv)
never had real signal to find. Added a threshold-AR(1) momentum term to the generator
(nonlinear — a regime switch based on the size of the previous move, not one global
AR coefficient — so linear AR(1) is a genuinely weaker classical baseline than a
threshold-aware fit).

**5a** confirmed the injected structure is real and detectable at this project's
standard panel scale (Ljung-Box Q=44.4 vs. critical ~11) and genuinely nonlinear
(threshold-AR(1) beats linear AR(1) by 16.7%, R²=0.139 vs. 0.119).

**5b** trained the `TemporalPathway` module in isolation (no gate, no cross-sectional
pathway — isolating the capability in question, since the full model's much stronger
rho=0.5 cross-sectional signal would otherwise dilute this weaker one, the same
failure mode diagnosed in Exp3 Phase A.5). A first pass at 300 epochs gave a dramatic,
unstable result (mean R²=0.05, one seed at R²=-0.13) — caught before being reported:
loss hadn't converged, not a capability failure. Trained to actual convergence
(~epoch 1500-2000): mean R²=0.122, std=0.0037 — a clean statistical tie with linear
AR(1) (0.124), short of threshold-AR(1) (0.145).

**5c**, per an explicit pre-commitment to run one generous capacity/budget increase
and treat it as final regardless of outcome: hidden width 16→64, more epochs (6000),
multiple seeds. A first attempt reused 5b's learning rate and completely collapsed
(100% dead ReLU units, loss frozen for 3000+ epochs) — caught via checkpoint
diagnostics before being reported, fixed with the standard Adam default learning rate
(1e-3, not 5b's 1e-2) rather than any exotic tuning. Corrected result: mean R²=0.131,
std=0.0052, across 3 seeds — a small, real, consistent improvement over 5b (now
modestly beats linear AR(1), not just ties it) but the gap to threshold-AR(1) held in
every single seed, by more than that baseline's own noise band. Full detail, including
an effective-capacity caveat (~48% of the wider layer's units were dead throughout,
so this tested an effective width nearer 33 than the nominal 64): `results/
experiment5c_findings.md`.

**Verdict:** the NN can learn real lag structure from scratch (a genuine capability
demonstration — the first time in the whole project it matched, then modestly beat, a
classical baseline rather than losing outright) but does not close the gap to a
classical model when handed the correct functional form in advance, even after a
good-faith attempt to close that gap with more capacity and budget. Per the
pre-committed stopping rule, this closes the temporal-pathway question.

## A structural caveat surfacing across the whole investigation, not just Exp5

Raised directly mid-Exp5c and worth stating as a standing qualifier on every verdict
in this document, not only the temporal one: every generator used in this project was
built from a named, closed-form stochastic primitive — a linear Gaussian factor, a
compound Poisson jump process, a threshold-AR(1) process. Classical statistics has
purpose-built, often near-optimal estimators for exactly these named families, because
"a generator simple enough to hand-code with known ground truth" and "a model class
classical statistics already solved" are close to the same property. Concretely:
correlation is the literal sufficient statistic for a linear Gaussian factor (Exp1-2);
the rank-1 shape test in Exp3d-4b is the textbook statistic for exactly the rank-1
structure the generator was built to have; Exp5's threshold-AR classical baseline uses
the identical functional form the generator was built from. This does not make any
individual result false — every finding above is an honest, correct answer to "does
this NN architecture beat classical tools on this family of generators." But it scopes
every verdict in this document more narrowly than it has sometimes been phrased:
**this is a validated conclusion about textbook-tractable synthetic generators, not
yet a general claim about real cross-sectional coupling estimation.** The one clean
way out of this trap, per the original spec, is real market data (`docs/research_spec.md`
Experiment 5 in its own numbering — distinct from this document's Exp5), since real
markets are not the output of any closed-form process chosen in advance for analytical
convenience. A structurally different synthetic generator not built from named
textbook primitives (e.g. an agent-based simulation) would be a synthetic alternative;
neither has been tried.

## Experiment 6 — Real data (the actual escape hatch, exercised)

Pulled a real panel — 30 large-cap Coinbase-listed assets, daily closes, 2023-08-28 to
2026-08-27 (`data_real/coinbase_daily_panel_3y.parquet`, built fresh; no reusable batch
downloader existed in prior personal projects, though both exchanges' relevant
endpoints turned out to be public/unauthenticated). Three checks pre-registered in
`results/experiment6_prereg.md` *before* running anything, matching the discipline that
made Exp5c trustworthy — deciding in advance what would count as "real structure
found" vs. "nothing beyond noise" for each tool, since real data has no ground truth to
fall back on.

**Rank structure**: top eigenvalue of the realized correlation matrix explains 66% of
variance — the rank-1 assumption behind Exp1-4b's entire toolkit roughly holds on real
data. **Time-varying coupling**: rolling 30-day correlation swings 0.31→0.89 across the
window — not a minor wobble, a real falsification of the static-rho assumption used
everywhere else in this project. **Joint statistic**: fires on 2.9% of days (below the
5-15% pre-registered "sensible" band, but non-degenerate) — every one of 4 spot-checked
flagged dates corresponds to a real, independently verified market event (each
checked against a linked news source): a Trump crypto-reserve announcement that named the exact three
assets the statistic flagged as biggest movers; the October 2025 "10/10" $19B
liquidation crash (the single highest-scoring day in the panel); a Filecoin-specific
AI/DePIN rally the statistic still caught via correlated infrastructure tokens (real
evidence of a *thematic*, non-rank-1 coupling channel no generator in this project ever
modeled); and a broad February 2026 selloff. **Temporal structure**: held-out R² is
negative for both linear AR(1) and threshold-AR(1) (0/30 assets clear the pre-registered
bar) — a clean, informative negative, unlike Exp1's version of the same non-finding
(which was true only because that generator was built without autocorrelation).
Retroactively sharpens Exp5c: the synthetic NN edge there depended on a generator built
to contain lag-1 structure; this result is silent on whether daily real markets contain
anything analogous to find, and suggests hourly data (not daily) is where documented
real microstructure effects would actually live if this gets pursued further.

**Why the mixed outcome matters more than a clean one would have**: two checks
confirmed synthetic assumptions, one falsified a core one (static rho) — exactly the
credible, could-have-gone-either-way outcome pre-registration exists to produce. Given
the meta-caveat above, an all-green result here would have been the suspicious one.
Full detail: `results/experiment6_findings.md`.

**Where this leaves the project**: the temporal question (Exp5) and the cross-sectional
question (Exp1-4b) are both closed for the generator family tested, and real data has
now both confirmed (rank-1, event detection) and corrected (static rho) that work. The
one well-motivated, non-speculative open thread is the time-varying-rho finding itself
— the single place real data diverged from every synthetic assumption tested, and the
one condition flagged as most likely to matter that was never actually tried: does a
rolling/adaptive classical estimator handle it, or is this where the NN's flexibility
finally earns something real. Nine rounds in, this is also a complete and well-told
story on its own if the investigation stops here.

## Experiment 7 — Time-varying rho (regime switch), the literal spec item never run

Extended the generator with a single ground-truth regime switch (`rho_2`/
`regime_switch_frac`, backward-compatible when unset) — the exact "low rho, then a
shift to high rho mid-series" test named in the original spec's Experiment 3 list.

**Fixed-window rolling correlation** has a genuine, quantified, unavoidable tradeoff:
detection lag scales with window size (≈half the window), noise falls with window
size, and no single window does both well (window=5: lag≈2, std≈0.15; window=90:
std≈0.04, lag≈53). **EWMA correlation Pareto-dominates it** — same pattern as every
prior classical escalation (BNS, joint statistic, magnitude+shape) — achieving lower
noise at matched lag across the whole range tested, with no special tuning required.

**A CUSUM change-point detector was attempted as the next escalation and did not
resolve cleanly** — most simulated paths false-alarmed 500-900 steps before the true
break. Diagnosed directly: first, jump contamination (fixed via a winsorized,
bipower-variation-style robust statistic, same principle as the rest of this project);
after that fix, a deeper issue remained — the coupling statistic's ~190 pairwise terms
share common per-asset shocks, violating the i.i.d.-increment assumption standard CUSUM
threshold calibration depends on. Properly fixing this needs permutation/bootstrap
calibration against the statistic's real null distribution — genuine additional
statistical work, not a same-session correction. Chasing it further to keep raising
the classical bar before considering anything adaptive was recognized as the same kind
of scope creep already flagged on the NN side (endless architecture retries) — the
discipline against moving goalposts cuts both ways. Full detail: `results/
experiment7_findings.md`.

**Verdict**: EWMA is the confirmed classical ceiling *for this pass*, not a final
ceiling — CUSUM remains a plausible further improvement, explicitly flagged unproven
with the specific reason documented (autocorrelated test statistic, invalid standard
calibration) rather than dismissed. A genuine finding in its own right: even purely
classical escalation has a point where the next tool needs real, nontrivial work to
apply correctly — climbing the classical ladder isn't free either, a natural
complement to the textbook-tractable-generator caveat above. **This is the first
condition in the whole investigation where a real, confirmed limitation of the best
cheaply-available classical tool has been found, rather than a smarter classical
substitute closing the gap outright** — the first genuinely well-motivated case for
testing something adaptive/learned, if this continues, earned through the same
discipline as every negative result before it.

## Experiment 8 — Adaptive Gate vs. EWMA Frontier (final round)

Pre-registered before any model code existed (`results/experiment8_prereg.md`): a
small GRU gate, trained via supervised regression directly against ground-truth
`rho_hist(t)` — a deliberately more favorable regime than any classical tool here was
given, since EWMA/CUSUM never see ground truth. Evaluated on held-out paths with the
identical metrics as Exp7.

**Result: detection lag=11.8, within-regime noise=0.0070 — roughly 15x lower noise
than EWMA's frontier at comparable lag (~0.104 interpolated).** A result this large
demanded direct scrutiny before trusting it: every train/test path shared the same
switch location, raising the obvious risk of calendar-time memorization rather than
genuine signal-driven tracking. **Tested directly** by evaluating the trained model
(no retraining) on switch locations never seen in training (t=600, t=1400 vs. the
trained t=1000) — noise performance held essentially identical across all three
(0.0063-0.0067), directly ruling out memorization. The win's sheer magnitude also
rules out "just a rediscovered fixed EWMA" on its own: no fixed-decay linear
estimator can reach this noise level at this lag, full stop — that combination sits
outside the frontier established directly in Exp7. Full detail: `results/
experiment8_findings.md`.

**Correction, added after further scrutiny: Experiment 8 is not commensurable with
Experiments 1-7, and shouldn't be read as "round ten in the same ledger."** Every
comparison in Exp1-7 was symmetric — both sides inferred an unlabeled quantity from
data alone (realized correlation and the NN gate both estimate rho blind; EWMA and
rolling correlation both estimate blind). Exp8's three methods (GRU, supervised-optimal
AR(1), LightGBM) are all doing direct supervised regression against a target
(`rho_hist`) shown to them during training — a different task than any classical tool
in this project was ever asked to perform, even sharing the same input signal.
Comparing a method regressed against the answer key to one that has to infer the
answer without it is not an apples-to-apples architecture comparison.

This was checked, not just asserted: giving the classical side the *same* privileged
access (the supervised-optimal AR(1) control, fitting its one decay parameter to
directly minimize error against ground truth) still left most of the gap standing —
lag=35.5, noise=0.089, roughly 10x worse than the GRU. So the remaining spread among
GRU/LightGBM/AR(1) — all three given equal, privileged access to ground truth — is a
real finding about model flexibility (AR(1)'s one fixed dial vs. LightGBM's windowed
features vs. the GRU's unbounded adaptive memory). But it answers a narrower question
than "does the NN beat classical statistics": *given a privilege nothing in real
deployment provides, how much does architecture still matter?* That belongs in its own
ledger, separate from Exp1-7's symmetric, apples-to-apples results.

## Final project verdict

**Nine rounds answered the question this investigation opened with, under real,
repeated scrutiny, symmetrically for both sides: a classical, non-trained tool matched
or beat this NN architecture at coupling recovery, sample efficiency, jump-noise
robustness, jump/diffusive disentanglement, detection, classification, and
daily-frequency temporal structure — on synthetic generators and real market data
alike.** That is the complete, closed answer to "is this learned scalar gate a useful,
non-redundant cross-sectional coupling estimator": no, not once, under fair comparison.

**A tenth, differently-scoped round (Experiment 8) then asked a separate question**:
if a method is given something no classical tool here was ever offered — direct
supervision against ground-truth regime labels — does architecture still matter? Yes,
substantially (confirmed via a fairness control, not assumed). But this is a real,
useful side-finding about model flexibility under an unrealistic information
advantage, not a tenth data point on the original question, and not evidence that the
NN "wins" in the same sense the other nine results say it loses. Its practical
relevance depends entirely on whether labeled regime-change examples can be sourced
for real deployment — Experiment 6's independently-verified market events are a
plausible starting point, but that bridge is unbuilt.

A complete, honestly-earned, precisely-scoped answer to the question this
investigation opened with — and an honest accounting of where a follow-up question
changed the game rather than extending the same one.

## One level up: where this sits in the original, broader ambition

Everything above is a complete, rigorous answer to a well-scoped question. It's worth
recording, explicitly, that the question was always a small piece of a much larger one
this project started from: building a real time-series-DL research program for
financial markets, combining synthetic data generation with real market data —
open-ended, not a single question.

That ambition narrowed in two stages before any of the ten experiments above began:

1. An early pass identified roughly nine distinct facets of financial time series that
   are genuinely troublesome for generic architectures: cross-sectional coupling,
   volatility clustering, jumps/fat tails, long memory, regime switching, microstructure
   noise, near-random-walk weak signal, the leverage effect, structural breaks.
2. Everything actually built and tested across all ten experiments lives in exactly
   **one row of that table** (cross-sectional coupling), plus one late, single-round
   side-excursion into **one other row** (temporal/autocorrelation structure, Exp5).
   Two facets touched, out of roughly nine identified — and only one specific
   architectural idea (the gated `alpha` mixing attention vs. causal-conv) was ever
   tested, against one family of classical alternatives, on generators built from a
   narrow set of closed-form primitives, plus one real-data check.

**Concretely untouched from the original ambition:**
- **Volatility clustering** (GARCH-family territory) — never built, never tested.
- **Long memory / long-range dependence** — never tested; would likely need a
  different generator and a different architecture family (state-space/Mamba-class).
- **Microstructure noise** — the original motivating interest (tick data, order
  books) was set aside early in favor of daily/coarser data, never revisited.
- **The multi-facet "fingerprint" idea** — running several facet-detectors jointly on
  one asset to build a combined state vector — never built, since only one
  facet-detector was ever built.
- **Facet interaction/co-occurrence** — whether vol clustering precedes jumps, whether
  coupling spikes lead regime breaks — entirely unexplored.
- **Boosted trees or other non-NN ML as a first-class alternative** — only tested
  narrowly, as a late control on Experiment 8's specific sub-question, not against any
  of Experiments 1-7's questions.
- **A structurally different generator** (agent-based, copula-based) — named
  repeatedly as the real escape from the textbook-tractable-generator trap, never
  built; real data (Exp6) filled that role, but only for the two facets actually tested.

**The honest scope statement, one level broader than this document's own scope
caveats**: this document's caveats correctly say "this is a claim about
textbook-tractable generators, not real coupling estimation in general." One level up
from that: the whole ten-round investigation is itself a deep dive into one cell of a
roughly nine-facet table, using one architecture and one narrow family of generators.
Going deep on one facet with real rigor — including catching real overreach twice
(the 3e/4a corrections) and correcting an overreach in how Experiment 8's result was
framed — is a better outcome than shallow coverage of all nine facets would have been.
But if the original, broader research program is still the goal, this is the first
completed chapter, not the whole book.

## Experiment 9 — Realistic (Real-Shaped) Rho Curve, and a Correction to Exp8's Story

Exp8 tested an idealized clean single switch between two fixed levels. Real data
(Exp6) actually showed continuous wandering, never settling at plateaus. This
experiment injected the *actual* real rolling-correlation curve from Exp6 as
ground-truth `rho(t)` (the generator gained a `rho_path` parameter for this,
backward-compatible), and re-ran the same model comparison against this harder,
more realistic target.

**Fixed-decay classical filters failed outright**, more decisively than in Exp7/8:
EWMA and even a supervised-optimal AR(1) (given full ground-truth access to pick its
best decay) both scored **negative R²** (-1.76) — worse than predicting the mean. On a
continuously wandering target, unlike a clean step, a lagging filter is perpetually
behind, everywhere, all the time, not just briefly around one transition.

**First reading — before catching the same kind of confound Exp8 already knew to
check for, but bigger here**: LightGBM scored R²=0.328, the GRU scored R²=0.503 —
looked like Exp8's story repeating and strengthening. But every train/test path
shared the identical curve at identical positions — a bigger memorization risk than
Exp8's single switch point, since a whole complex curve gives a recurrent model far
more position-tied structure to potentially exploit.

**Tested directly, the same way Exp8 was**: evaluated both models on a time-reversed
version of the same curve with fresh seeds. **LightGBM held up (0.328→0.292) — its
relative lag/rolling features have no channel for absolute position to leak in.
The GRU collapsed (0.503→0.035)** — evidence it had learned an implicit internal
"clock" from processing the same sequence-from-start repeatedly against the same
time-indexed target, not genuinely adaptive tracking. Full detail: `results/
experiment9_findings.md`.

**This does not contradict Exp8's own result** (which passed an analogous check —
different switch locations — because a single switch point offers far less
memorizable structure than a whole unique curve). It's a real, separate finding:
**the recurrence that helped in Exp8 became an exploitable liability here, once
tested on a harder, more realistic target and checked with the same rigor.**

**Corrected final picture on the adaptive-gate question**: fixed-decay classical
tools genuinely fail at tracking realistic, continuously-varying coupling — that part
holds up and is, if anything, a stronger classical failure than Exp7/8 showed. But
"recurrence specifically wins" does not survive contact with a more realistic target
under fair scrutiny — a structurally simpler, position-invariant model (LightGBM)
won once the shortcut was removed. The honest final word on Exp8/9 together: something
flexible and non-fixed-decay is necessary once coupling moves continuously rather than
switching cleanly, but which specific architecture provides that flexibility matters
a great deal, and recurrence is not automatically the right choice — it can just as
easily open a new way to cheat as it can provide genuine adaptivity, and only checking
directly, every time, tells you which happened.

## Experiment 10 — Diverse Curves: A Clean Win, and a Deeper Problem

Built specifically to fix Exp9's exposed gap: every training/test path here gets its
own distinct, randomly generated curve (smoothed random walks, sinusoids, random
parameters) — no curve repeats, so "day t → value" isn't even a coherent shortcut.

**Result 1 — the clean, memorization-proof comparison, finally decisive**: EWMA and
supervised-optimal AR(1) both score R²=0.027 (essentially nothing) against diverse,
plateau-free curves. LightGBM: 0.632. **GRU: 0.667 — a real, validated win**, with no
shared curve for a shortcut to exploit. This is the result Exp8 seemed to show and
Exp9 showed wasn't fully trustworthy — now genuinely earned.

**Result 2 — testing both trained models against the actual real Exp6 curve (a shape
never included in training) reveals a third, more fundamental problem**: LightGBM
roughly breaks even (R²=-0.003, no better than the mean); **the GRU does worse than
the mean (R²=-0.23)**. Both flexible models, despite genuinely succeeding on their own
training distribution, fail to transfer to the one real-world pattern that motivated
this entire thread. Full detail: `results/experiment10_findings.md`.

**Why this is the more important finding**: this is a third instance of the
"textbook-tractable generator" theme running through the whole project. Every positive
result in the Exp7-10 arc (EWMA beating rolling-window, the GRU's Exp8 win, Exp9's
uncorrected reading, this experiment's clean win) was trained and evaluated on
synthetic curve families invented for convenience — never on, or validated against,
how real coupling actually moves. The one time real dynamics were used as the actual
target rather than the training distribution, both flexible models failed to transfer,
the GRU worse than LightGBM despite winning everywhere else.

**Final word on the whole adaptive-tracking arc (Exp7-10)**: fixed-decay classical
tools have a real, confirmed, repeatedly-validated limitation. Flexible models can
genuinely address it, within a training distribution that matches what they're tested
on. Whether any of this helps on real markets remains unproven and, on the one direct
check run, unpromising. The open problem this arc actually leaves behind is not
architectural — it's sourcing or constructing a training distribution of coupling
trajectories that genuinely resembles real market dynamics, which is a harder, and
different, problem than anything solved by Exp7-10.
