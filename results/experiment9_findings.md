# Experiment 9 — Realistic (Real-Shaped) Rho Curve: Findings

Experiment 8 tested tracking an idealized, clean single switch between two fixed rho
levels. Real data (Exp6) actually showed a continuously wandering coupling level, never
settling at fixed plateaus. This experiment injects the *actual* real rolling-correlation
curve from Exp6 (lightly smoothed) as the ground-truth `rho(t)` for a fresh synthetic
generator run (`jump_diffusion_generator.py` gained a `rho_path` parameter for this —
an arbitrary per-step array, overriding `rho`/`rho_2`, verified backward-compatible),
and re-runs the same model comparison against this harder, more realistic target.

## Setup

Real curve: 1066 days, smoothed rolling correlation, range [0.33, 0.89] (essentially the
Exp6 curve). 40 training paths / 20 test paths, all sharing this identical curve as
ground truth with independent random noise per path — same philosophy as Exp7/8.
Compared: the raw per-step coupling statistic (no smoothing), EWMA (best of a heuristic
halflife grid), a supervised-optimal single-decay AR(1) (fit directly against ground
truth, same privilege as Exp8's control), LightGBM (lag/rolling features), and the GRU
adaptive gate — all evaluated via held-out R² against the true curve (no single
discrete event here, so no "detection lag" metric; the target moves continuously).

## First result: classical fixed-decay filters don't just underperform, they fail outright

| model | held-out R² |
|---|---|
| raw signal (no smoothing) | -36.59 |
| EWMA (best heuristic halflife=20) | **-1.76** |
| supervised-optimal AR(1) (decay=0.963) | **-1.76** |
| LightGBM | 0.328 |
| GRU | 0.503 |

**Both EWMA and the supervised-optimal AR(1) score negative R²** — worse than simply
predicting the curve's own mean, even with the AR(1) given full ground-truth
supervision to pick its best decay. This is a much starker classical failure than
Exp7/8's step-switch case, and the mechanism is different: on a step function, a
lagging filter only pays a real cost briefly, around the one transition, then tracks a
constant well for a long stretch. On a *continuously wandering* target, a fixed-decay
filter is perpetually lagging a moving target — there's no stable plateau to settle
into, so the "always a beat behind" cost applies everywhere, all the time, and
compounds into a negative R² over the whole series. That both LightGBM and the GRU
score positive here shows this specific failure isn't inherent to the task; it's
specific to being restricted to one fixed dial.

## Second result — and the important correction: the GRU's edge over LightGBM was a memorization artifact

A result this dramatic demanded the same scrutiny as Exp8's win, arguably more:
**every training and test path shares the exact same curve at the exact same
positions** — a bigger memorization risk than Exp8's single switch point, since a
whole complex curve gives a recurrent model far more curve-specific, position-tied
structure to potentially memorize rather than genuinely read from the input.

**Tested directly**: retrained both models identically, then evaluated on a
**time-reversed** version of the same curve, with fresh random seeds (`rho(t) →
rho(T-t)`) — if a model is reading the actual input signal, performance should hold up
reasonably; if it memorized "day t → value" by position, it should collapse, since day
t now needs a completely different answer than what was true in the forward version.

| model | forward curve (original) | reversed curve (OOD) |
|---|---|---|
| LightGBM | 0.328 | **0.292** — holds up |
| GRU | 0.503 | **0.035** — collapses |

**LightGBM's result is genuine.** Its features (lags 1-20, rolling stats over 5-40 step
windows) are all defined relative to the current step — they mean the same thing at
day 200 as day 800, forward or reversed — so there was never a channel for absolute
position to leak into its predictions. Its performance holding up almost exactly
confirms it's reading recent shape, not memorizing a timeline.

**The GRU's headline result does not survive this check.** A recurrent model that
processes every sequence from t=0 forward, trained repeatedly against the *same*
time-indexed target, has a real incentive to learn an implicit internal counter — "after
processing roughly this many ordinary-looking steps, the answer tends to be around
here" — regardless of what the recent data actually says. Reversing the curve breaks
that shortcut entirely, and the GRU's R² collapsed from 0.503 to 0.035, actually
*worse* than LightGBM once the shortcut is removed.

## Why this doesn't contradict Experiment 8

Exp8's GRU passed an analogous out-of-distribution check (different switch locations
never seen in training) and held up. The difference: a single switch point is a nearly
trivial thing to memorize past — there's very little curve-specific structure a
position-counter could latch onto beyond "before/after one moment." A whole complex,
uniquely-shaped real curve gives a recurrent model vastly more memorizable,
position-tied structure to exploit. The vulnerability found here is specific to this
combination (one shared, information-rich curve + full-sequence recurrent processing
from a fixed start), not a general flaw in recurrent architectures, and not a reversal
of Exp8's own, separately-validated result.

## Verdict

**Corrected ranking, once genuinely checked: LightGBM > GRU on this realistic,
continuously-varying target** — the opposite of the uncorrected, forward-only reading.
Two things survive scrutiny cleanly: (1) fixed-decay classical filters (EWMA, even
supervised-optimally-tuned AR(1)) fail outright at tracking continuously-wandering
coupling, a real and more decisive classical failure than Exp7/8 showed; (2) a
position-invariant flexible model (LightGBM) genuinely succeeds where they fail. What
does *not* survive scrutiny is the specific claim that recurrence itself is the
winning ingredient under these more realistic conditions — here, the architecture-level
freedom that made the GRU flexible also made it exploitable, and a simpler,
structurally-constrained model won once that exploit was removed.

This is a meaningful qualification of Experiment 8's story, not a repeat of it: the
"NN wins when given privileged supervision" finding from Exp8 doesn't generalize
cleanly to a harder, more realistic target — recurrence specifically turned out to be
a liability here, not an asset, once tested fairly.
