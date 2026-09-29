# Experiment 10 — Diverse Curves: Findings

Built to fix Exp9's exposed gap: there, every training and test path shared one
identical real curve, so a recurrent model could partly win by memorizing "day t ->
value" instead of reading the signal (confirmed by its collapse on a reversed-curve
check). Here, every single training and test path gets its own distinct, randomly
generated curve (smoothed random walks and sinusoids, random parameters) — no curve
repeats, so "day t -> value" isn't even a coherent shortcut anymore.

## Result 1 — the clean, memorization-proof comparison: GRU genuinely wins

| model | held-out R² (diverse, never-seen curves) |
|---|---|
| EWMA (best halflife) | 0.027 |
| Supervised-optimal AR(1) | 0.027 |
| LightGBM | 0.632 |
| **GRU** | **0.667** |

Classical fixed-decay filters barely register (0.027 — essentially no signal) against
continuously-varying, plateau-free curves. Both flexible models succeed genuinely, and
this time the GRU's edge over LightGBM is real: no shared curve exists across examples,
so there is nothing for a position-based shortcut to latch onto. **This is the clean,
decisive validation Experiment 8 seemed to show and Experiment 9 showed wasn't fully
trustworthy — now actually earned.**

## Result 2 — the new, more important problem: neither model transfers to the real curve

Evaluated both trained models (no retraining) against fresh synthetic paths generated
using the actual real Exp6 correlation curve — a shape never included in training,
qualitatively different from the smoothed-random-walk/sinusoid families used there.

| model | R² on the real curve |
|---|---|
| EWMA / supervised AR(1) | -1.70 to -1.74 |
| LightGBM | -0.003 (no better than predicting the mean) |
| **GRU** | **-0.23 (worse than predicting the mean)** |

**Both flexible models, despite genuinely succeeding on their own training
distribution, fail to transfer to the one real-world pattern this whole investigation
was motivated by.** LightGBM degrades gracefully (roughly breaks even); the GRU
degrades badly, doing worse than a trivial constant guess. A plausible reason the GRU
transfers worse despite winning within-distribution: the extra flexibility that let it
adapt well to the training curves' specific character (their particular smoothness,
typical rate of change, value range) may have let it specialize more tightly to that
distribution — genuinely useful within it, actively misleading outside it.

## Why this matters more than the memorization finding

Exp9 found a *methodological* problem (a specific test setup could be gamed).
Exp10 fixes that cleanly and confirms a real capability gap exists between flexible
and fixed-decay models — but then surfaces a *different, more fundamental* problem:
**genuine, validated, memorization-free adaptive tracking ability is still only as
good as the match between training data and real target dynamics.** This is a new,
third instance of the "textbook-tractable generator" theme that has run through this
entire project (see `results/SUMMARY.md`'s meta-caveat): even after doing everything
right — diverse training curves, no shortcut available, a clean and fair win — the
result still doesn't transfer to real data, because the *synthetic curve-generating
process itself* (smoothed random walks, sinusoids) doesn't match how real coupling
actually evolves.

## Verdict

Three separable findings now sit on top of each other for the adaptive-gate question:
1. Fixed-decay classical filters (EWMA, even supervised-optimally-tuned AR(1))
   genuinely fail at tracking continuously-varying coupling — confirmed repeatedly,
   robust across every version of this test (Exp7, Exp8, Exp9, Exp10).
2. Flexible, non-fixed-decay models (GRU, LightGBM) can genuinely succeed at this —
   confirmed cleanly here, without the memorization confound that undermined Exp9's
   version of the same claim.
3. **None of that success has been shown to transfer to real market dynamics.**
   Every positive result in this entire adaptive-tracking thread (Exp8's win, Exp9's
   uncorrected reading, Exp10's clean win) was trained and evaluated on synthetic
   curve families invented for convenience, not derived from or validated against how
   real coupling actually moves. The one time real dynamics were used as the *target*
   rather than the *training distribution* (this experiment), both flexible models
   failed.

**Honest bottom line for the whole Exp7-10 arc**: fixed-decay classical tools have a
real, confirmed limitation. Flexible models can address it — in principle, and within
a matching training distribution. Whether any of this actually helps on real markets
remains untested and, on the one direct check available, unpromising. The right next
step, if this thread continues, is not another architecture or training trick — it's
building (or sourcing) a training distribution of coupling trajectories that actually
resembles real market dynamics, which is a data problem, not a modeling problem, and
a substantially harder one than anything solved in Exp7-10.
