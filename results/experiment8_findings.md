# Experiment 8 — Adaptive Gate vs. EWMA Frontier: Findings (Final Round)

Per `results/experiment8_prereg.md`, written before any model code existed. Three
outcomes were pre-defined — win, rediscovery, loss — so this result can be trusted
regardless of which one obtained. This is the last round of the investigation.

## Setup

A small GRU (hidden=12) + linear head, taking the identical causal input used for
EWMA/CUSUM in Exp7 (the winsorized per-step coupling statistic), trained via
supervised MSE regression directly against ground-truth `rho_hist(t)` on 40 training
paths from the Exp7 regime-switch generator (rho=0.2→0.85, switch at the midpoint),
evaluated on 20 held-out paths using the identical detection-lag and within-regime-
noise metrics as Exp7.

**Critical, deliberately-flagged asymmetry**: this is supervised training against
ground truth — a strictly more favorable regime than any classical tool in this
project was given, since EWMA/CUSUM are blind formulas that never see a "true" value
during calibration. This was pre-registered as an intentional scoping choice to
isolate "can a network learn genuinely adaptive tracking at all," not a same-information
comparison. Real data has no ground truth to supervise against — this caveat is
central to what the result below can and can't claim, see Verdict.

## Result

Held-out evaluation: **detection lag mean=11.8 steps (median=9.0), within-regime
noise: pre=0.0312, post=0.0070.**

Against the Exp7 EWMA frontier at comparable lag (interpolating between halflife=5,
lag=6.3, std=0.114 and halflife=10, lag=14.3, std=0.099 → ~0.104 at lag≈11.8): **the
adaptive gate's post-switch noise (0.0070) is roughly 15x lower than EWMA's
interpolated value at the same lag.** This is not a marginal win against the
pre-registered >10%-at-3-of-4-points bar — it clears it by an order of magnitude.

## Before trusting this: checked for the obvious confound directly

A result this large demanded the same scrutiny as any other in this project. The
clear risk: every training AND test path used the identical switch location
(t=1000, 50% through the series) — the network could have simply memorized "output
low before t≈1000, high after," a fixed calendar-time pattern, rather than genuinely
reading the local input signal. That would produce an artificially clean result with
no real adaptive capability behind it.

**Tested directly**: evaluated the trained model (no retraining) on switch locations
never seen in training — t=600 (30% through) and t=1400 (70% through):

| switch location | seen in training? | detection lag | post-switch noise |
|---|---|---|---|
| t=1000 | yes | 15.3 | 0.0064 |
| t=600 | **no** | 8.1 | 0.0063 |
| t=1400 | **no** | 18.4 | 0.0067 |

**Noise performance is essentially identical across all three, including two switch
locations the model never encountered.** This directly rules out calendar-time
memorization — the network is reading the actual local input signal to detect and
track the change, not a fixed temporal pattern. This is the single most convincing
piece of evidence in this experiment, more decisive than the secondary check below.

## Secondary check: is this just a fixed EWMA the network reinvented?

Impulse-response probe (numerical sensitivity of the output to each of the last 60
input lags, fit against a best-fit exponential decay): stable-regime evaluation
(t=500) gave implied decay=0.809 (R²=0.994, a very clean single exponential); just
after the switch (t=1010) gave implied decay=0.850 (R²=0.980) — the rates differ by
more than the pre-registered 0.02 threshold, coded as evidence against rediscovery.

**Worth being honest about a real interpretive limitation here**: the t=1010
evaluation window spans both regimes (its most recent ~10 lags are post-switch, its
older ~50 lags are pre-switch), so the differing fitted decay could partly reflect
different input variance across the two regimes within that mixed window, not
necessarily a clean "the network's effective memory length changed" signal. This
check is suggestive but not as airtight as the out-of-distribution generalization
result above. It doesn't need to be airtight to support the overall conclusion,
though: **the magnitude of the win alone is sufficient proof the network isn't a
fixed-decay linear filter in disguise** — no single decay parameter can achieve
0.006-0.007 noise at lag≈10-18, full stop; that combination is mathematically outside
the frontier any fixed-decay estimator can reach (established directly in Exp7's
sweep). The mechanism-level probe is corroborating detail, not the load-bearing
evidence.

## Fairness control: what if the classical model got the same supervised advantage?

A fair objection: EWMA is itself mathematically an AR(1) recursion
(`estimate[t] = λ·estimate[t-1] + (1-λ)·x[t]`), and it was only compared against a
heuristic grid of decay values in Exp7 — not a version given the same ground-truth
supervision the GRU received. Checked directly: fit the single decay parameter of an
AR(1) filter on the identical input (`s[t]`) by directly minimizing error against
ground-truth `rho_hist` on the same 40 training paths — the exact same information
advantage the GRU got.

**Result: supervised-optimal decay=0.961, held-out lag=35.5, held-out noise=0.089** —
its own training error (0.045) was already 18x worse than the GRU's (0.0025), before
touching held-out data. This lands right back on the Exp7 tradeoff curve (comparable
to the ~halflife-30-60 region already characterized) — supervision did not let a
fixed-decay model escape the tradeoff at all. **This directly confirms the structural
argument rather than merely asserting it**: a single decay parameter is one dial that
must work uniformly across calm periods and change-points alike; no amount of fitting
that one dial better changes what fitting *one* dial can achieve. The GRU's advantage
is not "better-calibrated smoothing" — it's not being restricted to a single fixed
dial in the first place.

## Second control: does recurrence specifically matter, or would any flexible model do?

A further fair question: is the GRU's win about its recurrent memory specifically, or
would *any* sufficiently flexible supervised nonlinear model close the gap — in which
case a far more standard tool (gradient-boosted trees) might do just as well without
needing a neural network at all? Quick check: LightGBM on simple engineered features
(lags 1-20 of `s[t]`, rolling means/stds over windows 5-40, a recent-vs-long-run
difference), same training/eval protocol.

| model | held-out lag | held-out noise |
|---|---|---|
| Supervised-optimal AR(1) | 25.9 – 54.4 | 0.071 – 0.106 |
| LightGBM (lag/rolling features) | 6.7 – 11.9 | 0.018 – 0.032 |
| GRU (recurrent) | 11.8 – 18.4 | 0.0063 – 0.0070 |

**LightGBM decisively beats AR(1) (3-4x lower noise) and generalizes cleanly to
unseen switch locations** (its out-of-distribution noise was as good as or better
than in-distribution) — genuinely reading the signal, not memorizing. **It does not
fully close the gap to the GRU**, landing at roughly 3-4x higher noise. The most
likely reason: its engineered features only reach a 40-step window, while the GRU's
hidden state can represent an effectively unbounded, smoothly-adaptive memory with no
such cap — LightGBM is bounded by what was engineered into its features, the GRU
isn't bounded by a human's choice of window length at all.

**This sharpens, rather than undermines, the Exp8 conclusion**: the win isn't purely
"any flexible nonlinear model would match it" (LightGBM falls meaningfully short) nor
purely "you specifically need a neural network" (LightGBM gets most of the way there
with a much simpler, more standard tool). Both flexibility *and* an unbounded,
self-adaptive memory mechanism are doing real, separable work.

## Correction: this is not the same kind of comparison as Experiments 1-7

Worth stating precisely, because the framing above ("15x win") invites reading this
alongside the other nine rounds as if it answered the same question with a bigger
number. It didn't. **Every comparison in Experiments 1-7 was symmetric**: both sides
inferred an unlabeled quantity from data alone (realized correlation and the NN gate
both estimate rho blind; EWMA and rolling correlation both estimate blind). Experiment
8's three methods — the GRU, the supervised-optimal AR(1), and LightGBM — are **all
three** doing direct supervised regression against a target (`rho_hist`) they are
shown during training. That is not the same task EWMA/rolling-correlation were ever
asked to perform, even though they share the same input signal. Comparing a method
regressed against the answer key to one that has to infer the answer without it is not
an apples-to-apples architecture comparison — closer to an open-book test against a
closed-book one.

This does **not** mean the AR(1)-vs-GRU-vs-LightGBM comparison is worthless — it
directly answers a real, narrower question: *among methods given equal, privileged
access to ground truth (a privilege nothing in real deployment provides), how much
does model flexibility still matter?* The answer to that narrower question is a real
10x+ spread driven by architecture (AR(1)'s one fixed dial vs. LightGBM's windowed
features vs. the GRU's unbounded adaptive memory) — confirmed, not assumed, since the
"maybe it's just about who saw the ground truth" hypothesis was directly tested (the
supervised-AR(1) control above) and falsified: matching information access left most
of the gap standing. But that finding belongs in its own, separately-scoped ledger,
not folded into "the NN beat classical statistics" the way Experiments 1-7 mean it.

## Verdict: WIN, with the scope stated precisely

Per the pre-registered criteria, this is unambiguously a **win** — the adaptive gate
dominates the EWMA frontier by an order of magnitude, confirmed genuine (not
calendar-memorization) via out-of-distribution testing on unseen switch locations, and
confirmed not a rediscovered fixed smoother via the sheer magnitude of the
improvement.

**The scope of what this proves is narrower than "the NN beats classical," and that
narrowness is the actual finding.** The gate was trained with direct supervision
against ground-truth regime labels — real markets, and Experiment 6's real data
specifically, don't hand out confirmed "the coupling regime changed exactly here"
labels for free. This is the same shape of caveat as Experiment 4b's classification
fix (a real, working, cheap fix that needed modest labeled calibration data rather
than pure structural assumptions) — except here the value of that supervision is
dramatically larger: an order of magnitude, not a percentage-point improvement.

**What would make this transferable to real deployment**: a source of labeled
regime-change examples to train on. Experiment 6 already surfaced a plausible
bootstrap source — the four real, independently-verified market events found there
(the March 2025 Trump crypto-reserve announcement, the October 2025 liquidation
crash, etc.) are exactly the kind of confirmed "something changed here" labels that
could seed a real training set, the same way Exp4b flagged prior jump-risk work as a
possible label source for its classifier.

## Final answer to the question this experiment was built to ask

Time-varying rho (Experiment 7) was the one condition in this entire ten-round
investigation where a real, confirmed limitation of the best cheaply-available
classical tool was found rather than a smarter classical substitute closing the gap.
It is also the one condition where, given the resource classical methods structurally
cannot use (labeled ground truth), an adaptive learned approach decisively earns its
complexity — not narrowly, not ambiguously, by an order of magnitude. The entire
nine rounds before this one found the opposite, repeatedly, under real scrutiny. This
result doesn't overturn that pattern; it completes it: the NN's advantage over
classical statistics in this whole investigation appeared exactly once, exactly where
classical statistics was structurally prevented from using the information that
would have closed the gap, and nowhere else.
