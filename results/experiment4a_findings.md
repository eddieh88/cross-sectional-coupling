# Experiment 4a — Multi-Factor Stress Test: Findings (and a correction to 3e/3f)

This was meant to answer one question: does the fully-classical pipeline from 3d/3e/3f
survive a genuinely multi-factor common-diffusion structure, or does its win depend on
the generator's rank-1 assumption by construction (the spec's Experiment 3/4 concern,
never actually tested)? It ended up surfacing something more consequential — a
calibration blind spot in how 3e/3f's classification stage was evaluated, present even
in the original single-factor (rank-1) case. **This corrects, not just extends, the
previous "classification is a solved problem" verdict.**

## Setup

`jump_diffusion_generator.py` gained a `factor_dispersion` parameter (0 = original
rank-1 case, exact bit-for-bit backward compatible). `factor_dispersion > 0` activates a
second independent common factor with a fixed, heterogeneous per-asset loading angle
`theta_i ~ Uniform(0, factor_dispersion * pi/2)`, combining as `cos(theta_i)*z1 +
sin(theta_i)*z2`. Verified this genuinely breaks rank-1 structure: correlation-matrix
eigenvalues shift real mass from the 1st to a 2nd component as dispersion increases
(e.g. rho=0.8: top-2 eigenvalues go from `[16.2, 0.2]` at dispersion=0 to `[14.0, 2.4]`
at dispersion=1.0). Jump structure (still common-timed, independent per-asset sizes)
was left untouched, isolating the effect to diffusive commonality specifically.

Swept `factor_dispersion in [0.0, 0.3, 0.6, 1.0] x rho in [0.3, 0.6, 0.9]`, rerunning
both load-bearing pieces of the classical pipeline: the 3d-style shape check (ground-
truth timing, true sigma) and the 3f-style joint detector + R²-threshold classifier
(estimated sigma, real detector, k=10).

## Result 1 — the detector is robust to multi-factor structure

Joint-statistic detector recall stays 97.5–100% across every dispersion level tested, at
every rho. The joint statistic sums squared standardized moves regardless of which
factor(s) drive them, so pooling weak per-asset signals doesn't care whether there's one
common factor or two. **No degradation here — this piece of 3f holds up.**

## Result 2 — the shape signal genuinely degrades under multi-factor structure

Ground-truth-matched R² separation (3d-style, true sigma, most-extreme diffusive events
per path):

| rho | diffusive R² @ disp=0.0 | @ disp=0.3 | @ disp=0.6 | @ disp=1.0 | jump R² (~flat) |
|---|---|---|---|---|---|
| 0.3 | 0.316 | 0.273 | 0.194 | 0.087 | ~0.05 |
| 0.6 | 0.593 | 0.537 | 0.370 | 0.117 | ~0.06 |
| 0.9 | 0.877 | 0.840 | 0.595 | 0.154 | ~0.06 |

Jump R² is unaffected (jump sizes are still independent per asset regardless of
diffusive structure) but diffusive R² collapses toward the jump baseline as dispersion
increases — exactly as expected, since a genuinely rank-2 common factor breaks the
single-ratio proportionality the regression relies on. Overlap between the two
distributions rises from ~0–1% (single factor) to 30–38% at full dispersion. **This is a
real, expected degradation — not a surprise, and not what changes the verdict.**

## Result 3 — the actual surprise: classification was never as solved as reported, even at dispersion=0

Checking specificity (correctly labeling a true diffusive **candidate** — one the
detector actually flagged, not a ground-truth-selected extreme event — as diffusive)
instead of just jump-class precision:

| rho | specificity, joint detector, disp=0.0 | disp=0.3 | disp=0.6 | disp=1.0 |
|---|---|---|---|---|
| 0.3 | **5.2%** | 5.5% | 4.1% | 0.0% |
| 0.6 | 52.3% | 39.8% | 34.5% | 27.6% |
| 0.9 | 77.6% | 73.8% | 68.1% | 50.0% |

**At rho=0.3, the classifier mislabels essentially every true diffusive candidate as a
jump — already, at dispersion=0, before any multi-factor stress is applied.** Re-ran the
same check against Experiment 3e's original per-asset BNS detector (not the joint
statistic) to confirm this isn't specific to 3f's detector: same story — rho=0.3:
specificity=0%, rho=0.6: 20%, rho=0.9: 66.7%.

**Why 3e/3f didn't catch this:** the previously-reported "precision" and "100%
classifier recall" metrics were computed over the whole candidate pool, which is heavily
jump-dominated at low rho (e.g. 104 true jumps vs. 58 true diffusive candidates at
rho=0.3) — jump-class recall is trivially easy (jump R² is always near zero, always
correctly labeled), so it dominates the aggregate metrics and hides that the small
diffusive-candidate population is being classified almost at chance. This is the same
shape of mistake as the "1.000 accuracy, 1-4 samples" trap caught earlier in 3e — an
aggregate metric computed on the wrong population — just one level more subtle, since
this time the *sample size* was fine (58+ candidates) but the *class balance* buried the
failure.

**Is this fixable by recalibrating the threshold (0.5 was never validated against a
real detector's output, only against ground-truth extreme events)?** Swept every
threshold directly on the flagged-candidate R² distributions:

- **rho=0.3**: jump and diffusive R² are nearly indistinguishable (medians 0.019 vs.
  0.061). Best possible single threshold reaches only **68.5% overall accuracy** — a
  genuine, not-fixable-by-calibration overlap. Most flagged diffusive candidates at low
  rho are moderate, not extreme, draws — exactly the regime where the rank-1 shape
  signal is weakest, and the original 3d test happened to only ever sample the extreme
  tail of that distribution.
- **rho=0.9**: real signal is there, but the fixed 0.5 threshold was badly miscalibrated
  — the optimal threshold is ~0.23, reaching **88.4% accuracy** (much better than the
  ~78% specificity the 0.5 cutoff gave, but still well short of "solved").

## Verdict

**Correction to `experiment3d_shape_signal_findings.md`'s claim that "classification is
a solved problem, classically, with no training required":** that conclusion only holds
for the most extreme diffusive events (the ones Experiment 3d's ground-truth-matched
comparison happened to test), not for the actual population any real detector flags.
For the population that matters — real detected candidates — classification is
**good at high rho (~85-90% achievable with a properly calibrated, rho-dependent
threshold) and close to non-functional at low rho (~68%, near the class-imbalance
baseline)**.

This changes the shape of the remaining question. It's no longer "detection is the
bottleneck, classification is solved" (3d/3e/3f's framing). It's: **classification of
moderate (non-extreme) coordinated events, especially at low-to-moderate cross-sectional
coupling, is not solved classically with a simple linear R² test** — a materially
different and narrower gap than either the original Phase B framing (end-to-end
classifier) or the detector-only framing that replaced it.

Two things worth separating before deciding anything about Phase B:

1. **A cheap classical fix may still exist** — e.g. a rho-adaptive threshold (rho isn't
   observable in practice, but perhaps estimable from the same data), a nonlinear
   classifier on the same R² feature plus magnitude, or additional classical features
   (e.g. dispersion of standardized moves, not just R² against sigma). None of this
   has been tried yet.
2. **The multi-factor result (Result 2) compounds this** in the direction that would
   matter for Phase B, if it comes to that: rank-2+ common structure makes the *already
   weak* moderate-event classification problem strictly harder, not easier.

**Recommendation: don't build Phase B yet, and don't treat 3e/3f's "consolidate"
recommendation as settled either.** The one cheap thing left untried is a better
classical classifier for the low-rho / moderate-event case (item 1 above) — that's the
next few-hours check, same discipline as every step so far, before concluding a learned
detector's flexibility is actually needed.

Both prior caveats stand: single, time-invariant common-factor family (even the
dispersion extension is still a fixed, static loading structure); rho itself isn't
observable in practice, so any rho-dependent calibration would need to be self-tuning
from the data, not read off a known parameter as the diagnostics here do.
