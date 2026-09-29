# Experiment 3f — Joint Detection Statistic: Findings

Experiment 3e left one specific gap: a per-asset co-occurrence detector (requiring
several assets to *each independently* cross their own BNS threshold) missed roughly
half of all true jump events, because jump sizes are drawn independently per asset —
some assets get a small draw too weak to individually trip a strict per-asset test, even
when the event is real. This tests the obvious classical fix before conceding the point
to a trained model: pool every asset's signal into one joint number, rather than
requiring several to independently clear a bar.

## Method

For each path, per-asset standardized moves `Z[t,i] = r[t,i] / sigma_hat_i` (sigma_hat
from bipower variation, jump-robust, no ground truth), then a single joint statistic per
timestep:

```
J[t] = sum_i Z[t,i]^2
```

A coordinated event — diffusive or jump — inflates many `Z[t,i]` at once, even if none
individually looks extreme, so `J[t]` rises even when the per-asset method would see
nothing. Threshold: `median(J) + k * MAD(J)`, robust and calibrated per-path (not a
theoretical chi-square reference, since the exact null distribution of `J` depends on
the cross-sectional correlation structure — i.e. on `rho` itself — which isn't known in
practice; this is an honest practical detector, not a rigorously derived asymptotic
test). Flagged candidates are then classified jump-vs-diffusive using the same R²-vs-
estimated-sigma test validated in Experiments 3d/3e.

## Result

| k | rho=0.3 recall / precision | rho=0.6 | rho=0.9 |
|---|---|---|---|
| 3 | 1.000 / 0.108 | 1.000 / 0.093 | 1.000 / 0.325 |
| 5 | 1.000 / 0.251 | 1.000 / 0.229 | 1.000 / 0.484 |
| 8 | 1.000 / 0.515 | 1.000 / 0.520 | 1.000 / 0.675 |
| 10 | 1.000 / 0.654 | 1.000 / 0.662 | 0.990 / 0.757 |
| 15 | 1.000 / 0.776 | 0.981 / 0.810 | 0.913 / 0.872 |
| 20 | 0.962 / 0.862 | 0.904 / 0.879 | 0.798 / 0.902 |

**Detector recall reaches 100% at loose thresholds and stays above 90% across a wide,
tunable range, trading cleanly against precision as `k` increases.** This is not "a good
chunk of the missing 45%" — it's the whole gap closed, with room to tune toward whichever
recall/precision balance a use case needs. The classifier stage (R²-vs-sigma), as in
Experiments 3d/3e, correctly recognizes essentially every true jump among candidates —
the false positives that limit precision at low `k` are diffusive events, not misclassified
jumps.

A good practical operating point: `k=10` gives ~99-100% recall at 65-76% precision;
`k=15` trades a little recall (91-98%) for noticeably better precision (78-87%).

## Verdict

**Fourth negative result for the neural network, and the most decisive one.** A
two-line joint statistic (sum of squared per-asset standardized moves, robust
per-path threshold) fully resolves the one gap Experiment 3e identified as a
plausible, mechanistically-justified case for a Phase B detector. The complete
picture across this whole diagnostic arc:

1. **Experiment 2**: NN doesn't beat plain correlation on the core diffusive-coupling
   question at all.
2. **Experiment 3 Phase A.5**: the existing single-gate NN doesn't disentangle jump vs.
   diffusive coupling — it's blind to jump structure entirely.
3. **Experiment 3 Phase A retest**: a properly calibrated *per-asset* classical jump
   test can't disentangle them either, for a real structural reason.
4. **Experiment 3f**: a *joint* classical statistic closes that exact gap, with no
   training required.

**Recommendation: consolidate, don't build Phase B.** A complete, fully classical
two-stage pipeline now exists for the entire coupling-disentanglement question this
investigation set out to answer:

```
joint detection statistic (sum of squared standardized cross-sectional moves,
robust per-path threshold)
        -> candidate coordinated-move timesteps
        -> R^2-vs-estimated-sigma shape classifier
        -> jump vs. diffusive label
```

No neural network outperformed or was even needed alongside any piece of this. The
same synthetic-data caveat applies throughout: this is validated on a generator with a
single, rank-1, time-invariant common factor. It justifies confidence in the classical
pipeline *for this class of problem*, not yet a claim about real markets' (likely
multi-factor, time-varying) coupling structure — that would be the natural next
question if this line of work continues, via Experiment 4 (a structurally different
generator) rather than via building the NN this diagnostic chain has now four times
found unnecessary.
