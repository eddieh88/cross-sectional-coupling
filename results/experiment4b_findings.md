# Experiment 4b — Magnitude Feature Fixes the Classification Gap: Findings

Experiment 4a found the shape-only classifier (R² of a candidate event's return vector
against sigma_i) is close to non-functional on the population a real detector actually
flags, especially at low rho (best possible accuracy 68.5% at rho=0.3 — a genuine
distributional overlap, not a calibration bug). This tests the obvious fix: R² measures
*shape* but throws away *scale*. Jump sizes are drawn from a fixed distribution
independent of rho; a genuinely common diffusive event's typical size scales with
`sqrt(rho)`. At low rho, that's a large, unused discriminating signal.

## Method

Added a second feature — `mean(|return|)` across assets, i.e. raw event magnitude — to
each candidate the joint detector (3f) flags, alongside the existing R². Tested two
ways of turning `[R², magnitude]` into a jump/diffusive call, swept across the same
rho x factor_dispersion grid as 4a:

- **Supervised**: linear fit on a random half of candidates, evaluated on the held-out
  other half (needs *some* labeled examples to calibrate).
- **Unsupervised**: 2-cluster k-means on the standardized 2D feature space, with true
  labels used only *after* fitting, to name which cluster is "jump" (needs no labels at
  all to fit — closer to the rest of this project's classical toolkit).

## Result

| rho | disp | R²-only (4a) | [R²+mag] supervised | [R²+mag] unsupervised |
|---|---|---|---|---|
| 0.3 | 0.0 | 0.685 | **0.963** | 0.957 |
| 0.3 | 1.0 | 0.688 | **0.935** | 0.930 |
| 0.6 | 0.0 | 0.809 | **0.954** | 0.953 |
| 0.6 | 0.3 | 0.807 | **0.960** | 0.647 |
| 0.6 | 1.0 | 0.761 | **0.980** | 0.787 |
| 0.9 | 0.0 | 0.884 | **0.936** | 0.868 |
| 0.9 | 1.0 | 0.856 | **0.945** | 0.840 |

(Full grid across all 4 dispersion levels x 3 rho values in the script output —
consistent pattern throughout for the R²-only and supervised columns.)

**Note on the unsupervised column above: these are single-seed (`k-means seed=1`)
numbers and are superseded below** — rho=0.6's unsupervised accuracy is highly
seed-dependent (55.8–95.8% across 30 seeds, at every dispersion level), so the 0.647 /
0.697 / 0.787 values above are not a reliable characterization of that regime. See the
multi-seed analysis immediately following for the corrected picture.

**The supervised version decisively fixes the gap: 93.5–98.0% accuracy across every
rho and every dispersion level tested, including the low-rho case (68.5%) and the
multi-factor stress (4a) that shape-alone couldn't handle.** No degradation pattern
with dispersion — if anything it's flat-to-improving, since the classifier can lean on
whichever of the two features still carries signal in a given regime. Verified this
isn't overfitting: held-out accuracy on unseen candidates matches in-sample accuracy
almost exactly (checked separately, ~1pt difference).

**The unsupervised version is a genuinely different, more fragile story — and the
first-pass characterization of *where* it's fragile was wrong.** The original single-seed
run reported a collapse specifically at rho=0.6 combined with dispersion>0 (64.7–78.7%).
Rerunning with 30 different k-means seeds at rho=0.6 showed accuracy ranging from
**55.8% to 95.8% at every dispersion level, including dispersion=0** — that original
result was a single lucky/unlucky seed, not a real rho x dispersion interaction.

Checked whether standard best-of-N-restarts (fit k-means from 20 different
initializations, keep the lowest-inertia solution — the normal fix for k-means's
random-init sensitivity) recovers the good result. **It doesn't**, which is the more
interesting finding:

| rho | best-of-20-restarts unsupervised acc (all 4 dispersion levels) |
|---|---|
| 0.3 | 93.0–95.7% |
| 0.6 | **74.7–79.5%** |
| 0.9 | 82.7–86.8% |

**The real pattern is a dispersion-independent weak spot centered at moderate rho, not
an rho x dispersion interaction.** At the extremes, one feature dominates cleanly enough
that k-means' own variance-minimizing objective happens to align with the true label
boundary — magnitude dominates at low rho (jumps have a fixed scale, diffusive events
are still small), shape dominates at high rho (the rank-1 signal is strong). At
rho≈0.6, neither feature dominates enough, so k-means' natural 2-way split of the point
cloud (by total variance) doesn't reliably coincide with the jump-vs-diffusive split,
even at its best-optimized (lowest-inertia) solution — a genuine structural limit of
*unsupervised* clustering on these two features, not noise or an initialization bug.

## Verdict

**The classification gap Experiment 4a exposed is fixable classically — but the fix
needs at least some labeled calibration data, not just structural assumptions about the
generator.** This is a meaningfully different requirement than everything upstream of
it (the BNS test, the joint detection statistic, even the original R²-vs-sigma idea) —
those need zero labeled examples, only known/estimable quantities (sigma_i). A
`[R², magnitude]` classifier needs a training set of confirmed jump vs. diffusive
events to fit even a 3-parameter linear boundary.

That's a real caveat for real-data transfer (Experiment 5) — real markets don't hand
you confirmed jump labels either, though a modest number might come from known
crash/event episodes (the "two identified crash events" the original spec mentions from
prior Deribit work) or from the ground-truth-timing trick already validated in 3d
applied to historical data with well-documented jump events.

**This does not become a case for Phase B.** A 2-feature, 3-parameter linear classifier
is about as cheap as classical methods get — nowhere near the complexity or flexibility
an NN would bring, and it resolves the gap decisively wherever it's been tested,
including under the multi-factor stress that was the open question. If anything, this
strengthens the "don't build Phase B" recommendation, while correcting *why*: not
"classification is solved with zero supervision," but "classification is solved with a
cheap, minimally-supervised classical model, robust to both the diffusive multi-factor
stress and the rho range tested."

**Updated complete pipeline:**

```
joint detection statistic (sum of squared standardized cross-sectional moves,
robust per-path threshold)
        -> candidate coordinated-move timesteps
        -> [R^2-vs-estimated-sigma, mean(|return|)] linear classifier
           (fit on a modest labeled calibration set)
        -> jump vs. diffusive label
```

Same caveats as 4a stand: validated on a family of generators (single- and
two-factor) that are still static/time-invariant in their loading structure; rho
itself isn't observable on real data and wasn't needed here (the classifier operates
on the two features directly, not on rho), which is actually a point in its favor for
real-data transfer.
