# Experiment 8 — Adaptive Gate vs. EWMA Frontier: Pre-Registration

Written before any model code exists. This is the last round of the investigation
regardless of outcome — the pre-registration exists specifically so a win, a loss, or
a "just rediscovered EWMA" result can all be trusted equally when reported.

## Why this test, and why now

Experiment 7 established EWMA correlation as the classical ceiling *for that pass* on
tracking a regime-switch in rho, with a real, quantified lag/noise frontier (e.g.
lag≈6→std≈0.114, lag≈28→std≈0.083). A further classical escalation (CUSUM) was
attempted and failed for a structural reason — its calibration theory doesn't apply to
this statistic's dependence structure — and fixing that properly is a distinct
side-project, not a same-pass fix. This is the first case in the whole investigation
where a real limitation of the best cheaply-available classical tool was found rather
than a smarter classical substitute closing the gap. It is therefore the one
legitimately earned opening for an adaptive/learned approach, per the original spec's
own §7 suggestion: alpha "made a function of recent conditions (e.g. output by a small
side-network conditioned on recent volatility/volume) rather than a single free
parameter trained once."

## Design (decided now, not adjusted after seeing results)

- **Task**: given a causal sequence of the same per-step coupling statistic used for
  EWMA/CUSUM in Exp7 (`s[t]`, winsorized), output an online estimate of the current
  correlation level at each timestep — the identical task EWMA and rolling correlation
  perform, on the identical input, so any difference is attributable to the estimator,
  not the information available to it.
- **Architecture**: a small GRU (1 layer, hidden width 8-16) + linear head — the
  smallest architecture that can express an input-dependent, adaptive effective memory
  (its update gate can learn to "forget" faster right after a real change than during a
  stable regime, which is the actual capability being tested, without hand-designing a
  CUSUM-style threshold).
- **Training**: supervised regression (MSE) directly against ground-truth `rho_hist(t)`,
  on the exact same generator configuration as Exp7 (rho=0.2→0.85, switch at the
  midpoint, 20 assets, 2000 steps), varying only the random seed across training paths.
  This is a deliberately more favorable and more direct training signal than the
  original gated model's reconstruction-based training (Exp1-5) — appropriate here
  because the question under test is specifically "can a network learn to adaptively
  track a changing correlation level at all," not a re-litigation of the original
  architecture's task design. Evaluated on held-out paths generated from held-out seeds,
  never seen during training.
- **Metrics**: identical to Exp7 — detection lag (steps after the true switch until the
  estimate crosses the regime midpoint) and within-regime noise (std of the estimate,
  away from the transition, in each stable regime) — computed the same way, on the same
  held-out evaluation protocol.

## Pre-registered win / rediscovery / loss criteria

- **Win**: the trained gate's (lag, within-regime noise) operating point(s) fall below
  EWMA's established frontier — lower noise than EWMA's interpolated value at the same
  lag, or lower lag at the same noise — at a majority of comparable operating points
  (at least 3 of the ~4 lag regions spanned by EWMA's tested halflives), by a non-trivial
  margin (>10% relative reduction in noise at matched lag, not a difference within
  likely estimation noise).
- **Rediscovery (a real, named possible outcome, not just a negative result in
  disguise)**: the gate's operating point lands ON or statistically indistinguishable
  from EWMA's frontier — i.e., it learned to approximate a fixed-decay exponential
  weighting, not a genuinely input-dependent adaptive one. Checked directly, not just by
  eyeballing the lag/noise numbers: fit the gate's *implied* impulse-response weighting
  (how much its output at time t is influenced by inputs at t-1, t-2, ... — approximated
  via a numerical sensitivity/gradient probe) against a best-fit exponential decay curve.
  A high-R² fit to a single fixed decay confirms rediscovery; a poor fit, or one where
  the effective decay rate itself visibly shifts around the true regime break, is
  evidence of genuine adaptivity.
- **Loss**: the gate's operating points are dominated by EWMA's frontier at every
  tested point (worse noise at every matched lag) — the fair-shot version of every
  other negative result in this project.

## What happens after this result

Regardless of which of the three outcomes above obtains, this is the final round of
the investigation. All three are complete, honest answers to the question this
specific test was designed to ask.
