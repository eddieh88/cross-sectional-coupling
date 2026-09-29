# Experiment 3d/3e — Shape-Signal Pre-Registration Check: Findings

Before scoping Phase B's architecture (a two-gate NN meant to separate diffusive
coupling from jump-timing coupling — see `results/experiment3_findings.md`), we tested
the mathematical claim its design would rest on: a shared **diffusive** draw moves every
asset in a fixed ratio proportional to its own `sigma_i` (rank-1 structure:
`diffusion_i = sigma_i * sqrt(rho) * z_common + idio`), while a common **jump** draws an
*independent* size per asset even when timing is shared. If that shape difference isn't
visible at realistic sample sizes using classical tools, there's no reason to expect an
NN to find it either — this is the cheapest possible test of the claim, run before any
architecture gets built.

Ground truth (which timestep is a real common jump, true per-step `z_common`) came
directly from generator instrumentation (`jump_diffusion_generator.py`'s
`return_diagnostics=True`), isolating the shape question from jump-*detection* noise
entirely, in the first pass.

## 3d — Does the shape signal exist at all? (best case: true sigma, ground-truth timing)

For each event (a full cross-sectional return vector at one timestep), regress it
against the known per-asset `sigma_i` vector and record R². Diffusive events should fit
well; jump events shouldn't.

| rho | jump R² (mean) | diffusive R² (mean) | overlap |
|---|---|---|---|
| 0.3 | 0.049 | 0.316 | 1.0% |
| 0.6 | 0.050 | 0.593 | 0.0% |
| 0.9 | 0.052 | 0.877 | 0.0% |

Separation held even in a **single-path illustration with only 3 real jump events** —
not an artifact of pooling across hundreds of paths (rho=0.9: jump R²s `[0.04, 0.04,
0.24]` vs. diffusive R²s `[0.87, 0.89, 0.94]`). **Clean positive result** — the first one
in this diagnostic chain.

Two things this first pass left open, flagged for follow-up rather than assumed away:
whether it survives *estimated* sigma (real data never gives you true `sigma_i`), and
whether a real *detector* (not ground-truth timing) can find candidate events at all.

## 3e, Part 1 — Does the signal survive estimated (not known) sigma?

Estimated `sigma_i` via bipower variation on each path's own return history (jump-robust
by construction, no ground truth used) instead of the generator's true value.

| rho | jump R² (true → est) | diffusive R² (true → est) |
|---|---|---|
| 0.3 | 0.049 → 0.050 | 0.316 → 0.322 |
| 0.6 | 0.050 → 0.051 | 0.593 → 0.594 |
| 0.9 | 0.052 → 0.053 | 0.877 → 0.872 |

**Essentially no degradation.** Estimation noise in `sigma_i` is negligible relative to
the shape signal at this sample size (300 steps/path). This meaningfully strengthens the
case that the shape signal could transfer to a setting where sigma has to be estimated,
not just assumed — though it says nothing yet about non-constant/time-varying volatility,
which real markets have and this synthetic setup doesn't.

## 3e, Part 2 — Does a full classical pipeline (detect + classify) actually work?

Replaced ground-truth jump timing with a real detector: per-asset BNS jump flags
(`src/baseline.py`), with a cross-sectional "candidate event" defined as a timestep where
at least some fraction of assets get individually flagged at once. Then applied the
R²-vs-estimated-sigma regression as a **threshold classifier** (R² > 0.5 → predict
diffusive, else jump) on whatever the detector flags.

**First attempt, 30% co-occurrence threshold — misleading:** only 1–4 candidate events
detected per condition, against ~104 true jump events actually present in the panel.
Accuracy on those candidates was a perfect 1.000 — but that's an artifact of the detector
almost never firing at all (~1–4% recall of true jump events), not evidence the pipeline
works. Caught before it got reported as a finding.

**Loosened to a 10% co-occurrence threshold — the honest picture:**

| rho | candidates | detector recall (of all true jumps) | precision | classifier recall (among candidates) |
|---|---|---|---|---|
| 0.3 | 62 | 56.7% | 95.2% | 100% |
| 0.6 | 64 | 56.7% | 93.7% | 100% |
| 0.9 | 57 | 51.9% | 98.2% | 100% |

**The classifier is essentially perfect once a candidate is flagged** — 100% recall
among candidates in all three conditions, and every classification error (93.2–98.2%
precision) comes from a small number of diffusive events that slipped past the detector
threshold, not from misclassifying jumps as diffusive. Phase A/3d's claim holds up:
shape *classification* is a solved problem, classically, with no training required.

**But detection is the real bottleneck: the detector misses roughly half of all true
jump events entirely.** Cause: jump sizes are drawn independently per asset
(`N(jump_mean, jump_std)`), so for any given jump event some assets get a small draw
relative to their own volatility — too small to individually cross a strict per-asset
BNS threshold, even though the *joint* cross-sectional pattern at that timestep may
still carry real information that something happened. Requiring several assets to
*independently* cross their own threshold at once discards that joint information.

## Verdict — a sharper, more specific case for Phase B than before

Shape *classification* (jump vs. diffusive, given a flagged candidate) is now a
genuinely solved problem, classically — no NN needed for that half. What is **not**
solved classically is **detection**: finding candidate coordinated-move events by
requiring independent per-asset threshold crossings misses about half the true events,
because it never looks at the cross-section jointly.

This meaningfully changes what a well-justified Phase B would actually be for. The
original framing ("attention pathway learns to find jump-vs-diffusion structure") was
too broad — classification doesn't need it. The narrower, evidence-backed case is:
**a pathway that improves detection sensitivity by evaluating the whole cross-section's
move pattern jointly**, rather than requiring multiple assets to each independently
clear their own threshold — catching real jump events that per-asset testing misses
because no single asset's move was extreme enough on its own, while the *shape* across
all of them still would have revealed it. Classification, once a candidate exists, can
stay classical (the R²-vs-sigma test already does it essentially perfectly).

If Phase B is scoped, this result argues for building it as a **detector**, evaluated
against the ~50% of jump events the classical pipeline currently misses — not as an
end-to-end classifier, which would be solving a problem that's already solved.

Both caveats from `results/experiment3_findings.md` still apply: this is validated on a
generator with a single, rank-1, time-invariant common factor — a synthetic-data result
that justifies testing Phase B on synthetic data, not yet a claim about real markets.
