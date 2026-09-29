# Experiment 7 — Time-Varying Rho (Regime Switch): Findings

The literal test named in the original spec's Experiment 3 list and never run until
now: *"regime-switching coupling within a single simulated series (e.g. low rho, then
a shift to high rho mid-series, mimicking a correlation breakdown). Does a
rolling-window version of alpha track the shift, with what lag?"* Directly motivated by
Experiment 6's real-data finding: rolling correlation on real crypto swung 0.31→0.89
across a 3-year window — a real falsification of the fixed-rho assumption used
throughout Exp1-4b, not a minor wobble.

## Setup

`jump_diffusion_generator.py` gained `rho_2` / `regime_switch_frac`: a single, one-time
switch from `rho` to `rho_2` partway through the series, with ground truth (`rho_hist`)
returned via diagnostics. `rho_2=None` reproduces the original fixed-rho behavior
exactly (verified bit-for-bit). Tested rho=0.2 → rho=0.85 (roughly matching the real
swing magnitude from Exp6), switching at the midpoint of a 2000-step, 20-asset,
30-path panel.

## Result 1 — fixed-window rolling correlation: a genuine, unavoidable tradeoff

Across window lengths 5 to 90: detection lag scales almost exactly with window size
(≈half the window, as expected — a rolling mean crosses the regime midpoint once about
half its data is post-switch), while within-regime noise falls as window grows.
Window=5: lag≈2 steps, std≈0.13-0.18 (nearly half the entire regime gap — unusably
noisy). Window=90: std≈0.03-0.07 (precise) but lag≈53 steps (over two months to catch
a shift that already happened). **No fixed window does both well — a real,
quantified, structural tradeoff**, not a tuning failure.

## Result 2 — EWMA correlation: a real, cheap classical improvement

Standard RiskMetrics-style EWMA correlation Pareto-dominates the fixed-window
estimator, consistent with every prior classical escalation in this project (BNS over
naive threshold, joint statistic over per-asset, magnitude+shape over shape alone):
at matched detection lag, EWMA achieves meaningfully lower within-regime noise across
the whole range tested (e.g. lag≈6: rolling std=0.133 vs. EWMA std=0.114; lag≈28:
rolling≈0.097 (interpolated) vs. EWMA=0.083). **This confirms the tradeoff itself isn't
fundamental to "using a window" — a smarter, still completely standard classical
estimator moves the frontier.** But the tradeoff doesn't vanish; EWMA still trades lag
against noise via its decay parameter, just more efficiently than a hard cutoff.

## Result 3 — CUSUM change-point detection: attempted, diagnosed, not resolved

The genuinely different next classical tool (not just re-tuning EWMA/window) would be
an explicit change-point detector: decide "a break happened here," then reset to a
short/expanding window immediately. Built a standard one-sided CUSUM (Page's test) on
a per-timestep coupling statistic (mean standardized pairwise product across all
asset pairs).

**First attempt failed outright**: most simulated paths triggered a "detection" 500-900
steps *before* the true break — a false-alarm problem. Diagnosed directly (not
assumed): a single common jump event (this generator still has active jump processes)
produced one massive transient spike in the coupling statistic, which alone accounted
for most of the false CUSUM accumulation in the traced example — the exact
jump-vs-diffusive contamination problem Experiment 3 solved, recurring in a new
statistic that wasn't built jump-robust in the first pass.

**Fixed the jump-contamination piece** (winsorizing the statistic against a robust
median/MAD band, the same principle behind bipower variation used throughout this
project) — **this did not fix the underlying problem.** Most paths still false-alarm
hundreds of steps early (though a few now detect near-perfectly, e.g. one path at
lag=+1). Traced this to a different, more fundamental issue: the ~190 pairwise
products behind the coupling statistic are not independent — they share common
per-asset shocks — so the statistic has real short-term autocorrelation that violates
the i.i.d.-increment assumption standard CUSUM threshold calibration (`h=5σ₀`,
`k=0.5σ₀`) depends on. Getting CUSUM working correctly here would require calibrating
its threshold against the statistic's actual false-alarm rate under permutation/
bootstrap resampling of the real null — legitimate, real statistical work, not a
same-session fix.

## Why this isn't "try harder classically," and why it's the right place to stop

Every previous classical escalation in this project succeeded by substituting a
smarter estimator for a cruder one *within a well-understood theoretical framework*
(BNS is calibrated jump-detection theory properly applied; EWMA is a standard
reweighting of the same correlation estimator). What happened with CUSUM here is
different in kind: the diagnosis isn't "the tool needs tuning," it's "the tool's
standard calibration theory doesn't apply to this statistic's actual dependence
structure." That's a real side-project (permutation-calibrated CUSUM), not a
same-session correction — and chasing it further to keep raising the classical bar
before considering anything adaptive/learned would be the same kind of scope creep
already flagged as worth avoiding on the NN side (endlessly retrying architectures
after a fair test). The discipline cuts both ways.

## Verdict

**EWMA correlation is the confirmed classical ceiling for this pass — not the ceiling,
full stop.** It resolves the fixed-window tradeoff into a genuinely better one with
essentially no calibration cost, consistent with this project's whole pattern.
**CUSUM/change-point detection remains a plausible further classical improvement,
explicitly flagged as unproven rather than dismissed** — documented with the specific
technical reason it's harder than it looks (an autocorrelated test statistic invalidating
standard threshold calibration), so anyone picking this up later doesn't repeat the
same naive-calibration mistake.

**A small but genuine finding in its own right**: even within "purely classical
escalation," there's a real point where the next-best-known tool requires nontrivial,
dedicated statistical work to apply correctly to this specific data's dependence
structure — climbing the classical ladder isn't free either. That's a natural
complement to the earlier meta-caveat about textbook-tractable generators: the
relevant ceiling here isn't only about what a neural network can't beat, it's also
about what's cheap versus genuinely expensive to get right on the classical side.

Time-varying rho is the one condition, across this entire project, where a real,
structural limitation of the best cheaply-available classical tool (EWMA's persistent
lag/noise tradeoff) has been found and confirmed, rather than a smarter classical
substitute closing the gap outright. If an adaptive/learned approach is tested next,
this is the first genuinely well-motivated case for it in the whole investigation —
earned through the same discipline as every negative result before it, not assumed.
