# Experiment 6 — Real Data (Coinbase, 30 assets, 3yr daily): Pre-Registration

Written before running the classical toolkit on `data_real/coinbase_daily_panel_3y.parquet`,
per the same discipline that made Exp5c's result trustworthy rather than spinnable either
way. Real data has no ground truth, so "what counts as a finding" has to be decided in
advance, not read off after seeing the numbers.

## Panel

30 large-cap Coinbase-listed assets, daily closes, 2023-08-28 → 2026-08-27, verified clean
(no zero-variance assets, no gaps, three checked large single-day moves confirmed as real
rather than data errors). Mean pairwise correlation 0.647 already establishes crypto's
well-known baseline co-movement.

## 1. Correlation / joint cross-sectional statistic

**Not interesting on its own:** finding *some* coupling. Crypto's "BTC-dominance" common
factor is well documented and expected regardless of any tool used — a high mean pairwise
correlation is a near-certainty, not a result.

**What would actually be worth reporting, decided in advance:**
- **Rank structure.** PCA/eigen-decomposition of the realized correlation matrix.
  - *Consistent with the synthetic assumption*: top eigenvalue explains a large majority of
    variance (rough threshold: >60%), i.e. close to rank-1 — the single-common-factor
    picture this project's classical tools were validated against roughly holds.
  - *A genuinely different, informative finding*: top eigenvalue explains substantially
    less (e.g. <45%) with real mass in a 2nd/3rd component — real coupling is
    multi-factor, which is exactly the stress condition Exp4a/4b tested synthetically, so
    if this shows up for real, that result becomes directly relevant rather than
    hypothetical.
- **Time-varying coupling.** Rolling 30-day mean pairwise correlation across the full
  window.
  - *Static*: rolling correlation stays roughly flat, fluctuating within a band consistent
    with sampling noise at this window length.
  - *Time-varying (the real-market stylized fact)*: visible regime shifts, e.g.
    correlation rising during a broad selloff — this alone would flag that a single
    fixed-rho framing (used throughout Exp1-4b) is too simple for real data, independent
    of any NN-vs-classical question.
- **Joint detection statistic (Exp3f method, k=10 operating point).** Pre-registered
  "sensible" range: statistic flags roughly 5-15% of days as candidates — enough to be a
  meaningfully rare signal, not so rare it never fires (uninformative) or so common it's
  just describing an average day (also uninformative, given the high baseline
  correlation).

## 2. BNS jump detection — no ground truth, so the validation method is decided now

There is no true jump/diffusive label on real data. The stand-in check: take the top ~10-15
flagged candidate dates (by joint-statistic magnitude or breadth of assets flagged) and look
each one up against real news/market events — **via a sourced lookup, not memory.** Day-level
recall of which specific news item moved a specific altcoin on a specific date isn't
reliable, so every date-to-event match reported will be backed by a linked source.

**What counts as a pass:** most top-flagged dates correspond to a real, identifiable
market-wide or asset-specific event (ETF/regulatory news, exchange incident, macro shock,
etc.) — a plausibility check on precision. No claim about recall is possible or will be made.

## 3. Threshold-AR / temporal structure — the one place I expect the LEAST signal

At daily frequency, real markets are close to informationally efficient — the expectation
going in is weak-to-no exploitable serial structure, unlike the synthetic version of this
same question (Exp1), where "no temporal signal" was true only because the generator was
built without any autocorrelation. A real-data null result here would be a genuine,
informative finding about actual markets, not a construction artifact repeating itself.

**Pre-registered pass/fail, on a proper walk-forward split** (train on the first ~2 years,
evaluate held-out on the final ~1 year, matching the spec's own recommendation for real-data
testing rather than an arbitrary random split):
- *Real structure found*: Ljung-Box test rejects the no-autocorrelation null at 5% on
  held-out returns AND a fitted AR(1)/threshold-AR model achieves held-out R² meaningfully
  above zero (threshold set low relative to the synthetic case, e.g. R²>0.02 — daily real
  markets are only ever expected to show weak effects like short-term reversal, not the
  strong 0.12-0.15 range seen in the synthetic experiment).
- *Nothing beyond noise*: Ljung-Box not significant, or held-out R² statistically
  indistinguishable from zero. This is the pre-registered "expected" outcome, and would be
  reported as a real, informative negative — not dismissed as a null result not worth
  mentioning.

## Method note applying across all three

Every classical tool reused here (realized correlation, joint statistic, BNS, threshold-AR)
carries the meta-caveat from `results/SUMMARY.md`: it was validated on generators built to
match that exact tool's assumptions. Running it on real data is the actual test of whether
that validation generalizes — which is the entire point of this experiment, and exactly why
results should be reported as found, including anywhere they come back weaker than the
synthetic work implied.
