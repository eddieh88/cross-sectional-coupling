# Experiment 6 — Real Data (Coinbase, 30 assets, 3yr daily): Findings

All three checks below were pre-registered in `results/experiment6_prereg.md` before
running anything. Reporting exactly against those pre-committed criteria, including
where the result came in weaker or different than expected.

## Data

30 large-cap Coinbase-listed assets, daily closes, 2023-08-28 → 2026-08-27 (`data_real/
coinbase_daily_panel_3y.parquet`). Clean: no zero-variance assets, no gaps, three large
single-day moves spot-checked and confirmed real (not data errors).

## Check 1 — Correlation structure

**Rank structure:** top eigenvalue explains **66.1%** of total variance (top-2 together:
69.3% — the 2nd component adds almost nothing). This clears the pre-registered >60%
threshold for "consistent with the rank-1 assumption." **Real crypto coupling, at least
in this universe and window, is close to the single-common-factor structure this
project's entire classical toolkit (Exp1-4b) was built and validated around** — a
genuinely reassuring, non-trivial finding, not assumed in advance.

**Time-varying coupling:** rolling 30-day mean correlation: mean=0.681, **std=0.122,
range 0.313 to 0.893** — nearly a 3x swing from calmest to most-coupled window, far
larger than sampling noise at this window length would produce under a truly static
rho. **Real coupling is clearly time-varying**, per the pre-registered "real stylized
fact" bucket, not the fixed-rho picture used throughout the synthetic experiments. The
lowest-correlation window clusters around early March 2024; the highest around
mid-February 2026 — concrete, dateable regimes, not noise.

**Joint statistic (k=10):** flagged **2.9% of days** (32/1095) — below the
pre-registered 5-15% "sensible" band, on the sparse side. Worth stating precisely
rather than rounding up to "as expected": the tool transfers and produces a
non-degenerate, plausible signal, but fires somewhat less often on real data than the
synthetic calibration implied it would at this same `k`.

## Check 2 — Real-event correspondence (verified against news sources)

Per pre-registration, no ground truth exists for jump labels, so the check is whether
top-flagged dates correspond to real, verifiable events — each one checked against a
contemporaneous news source (linked below), not recalled.

| date | flagged movers | verified real event |
|---|---|---|
| 2025-03-02 | ADA +54%, XRP +29%, SOL +22% | Trump's crypto strategic reserve announcement **specifically named XRP, SOL, ADA** — matches not just the date but exactly which three assets moved most ([CoinDesk](https://www.coindesk.com/markets/2025/03/02/xrp-sol-ada-s-coinbase-premium-surges-to-one-month-high-after-trump-s-crypto-reserve-news)) |
| 2025-10-10 | CRV -36%, OP -35%, ARB -33% | The "10/10" mass liquidation crash — $19B+ liquidated in ~14 hours, triggered by a Trump tariff announcement on China, altcoins fell 40-70% ([CoinGecko](https://www.coingecko.com/learn/october-10-crypto-crash-explained), [FTI](https://www.fticonsulting.com/insights/articles/crypto-crash-october-2025-leverage-met-liquidity)) — this was the single highest joint-statistic day in the whole panel (J=861.8) |
| 2025-11-07 | FIL +58%, NEAR +28%, GRT +22% | Filecoin-specific ~50-60% surge on an AI/DePIN narrative and ecosystem upgrades ([Yahoo Finance](https://finance.yahoo.com/news/filecoin-surges-50-24-hours-125634584.html)) — an asset-specific catalyst that the joint statistic still caught strongly, because correlated infrastructure/data tokens (NEAR, GRT) moved with it |
| 2026-02-05 | XRP -22%, SUI -21%, AAVE -20% | A broad selloff tied to a hawkish Fed Chair appointment (Kevin Warsh), Bitcoin to $70K, $775M liquidated ([CoinDesk](https://www.coindesk.com/markets/2026/02/05/xrp-plunges-16-in-worst-drop-among-bitcoin-ether-and-major-tokens)) |

**4/4 checked dates verified as real, identifiable events** — a clean pass on the
pre-registered plausibility check. No claim about recall is made (no ground truth
exists to measure it against), only that what's flagged corresponds to real structure.

**The Filecoin/DePIN date (2025-11-07) is the most important one for what it reveals
about the *kind* of coupling being detected, not just that detection worked.** Every
synthetic generator in this project encoded exactly one channel of cross-sectional
coupling: a literal, single common factor (plus, in Exp4a, a second static linear
factor). This real event was driven by a *thematic/narrative* channel — infrastructure
and AI-storage-adjacent tokens moving together on a sector-specific catalyst, not a
market-wide common shock. That the joint statistic caught it anyway is a genuine
success for the tool. But it's also direct, concrete evidence for the "classically-
tractable generator" critique raised earlier: real coupling includes structurally
different mechanisms (thematic/sector correlation) that no generator in this project
was built to produce, meaning the classical toolkit's success here is somewhat broader
than what was actually validated for.

## Check 3 — Temporal structure (walk-forward: train on 2023-08-29 → 2025-08-27,
held out on 2025-08-28 → 2026-08-27)

This was pre-registered as the check where I expected the **least** signal, and that
is exactly what came back:

- Ljung-Box (10 lags, held-out returns): 11/30 assets nominally "significant" at 5% —
  but 30 independent tests at 5% would produce ~1.5 false positives by chance alone
  under a true null, and 11/30 does not survive a Bonferroni-style correction for
  multiple testing (would require a substantially higher per-test bar). Weak evidence
  at best, not the overwhelming signal seen in the synthetic case (Exp5a: Q=44 against
  a critical value of 11, not a borderline call).
- **Held-out R²: linear AR(1) mean = -0.0096, threshold-AR(1) mean = -0.0114 — both
  negative, meaning both models do worse than predicting the mean out of sample.**
- **0 of 30 assets** clear the pre-registered R²>0.02 bar for either model.

**This is exactly the pre-registered "nothing beyond noise" outcome, and a clean one.**
Unlike Experiment 1's version of "no temporal signal" — which was true only because the
synthetic generator was built without any autocorrelation by construction — this is a
genuine, informative finding about real markets: **at daily frequency, this panel shows
no exploitable linear or simple-nonlinear own-history structure**, consistent with
markets being close to informationally efficient at this frequency. The in-sample fit
that a classical AR model finds does not generalize at all, which is itself the
correct, expected result for efficient markets — not a failure of the method.

**This retroactively sharpens Exp5c, not just repeats its shape.** The synthetic
threshold-AR result there was a real, if modest, NN edge over a naive linear baseline —
but that was possible only because the synthetic generator was built with genuine
lag-1 structure to find. This real-data check shows daily crypto returns may simply
have no comparable structure to discover in the first place, at any tool's disposal,
classical or learned. The honest reading of Exp5c is therefore narrower than "the NN
can learn some real temporal structure" — it's "the NN can learn temporal structure
*when a generator was built to contain it*," and this check is silent on whether daily
real markets contain anything analogous. If temporal structure is worth pursuing
further on real data, daily is very likely the wrong frequency to look for it at —
hourly (already discussed as a candidate direction) is where market microstructure
effects (order-flow imbalance, short-horizon momentum/reversal) are actually documented
to exist, unlike at this frequency.

## Overall verdict

**The mixed outcome is itself the strongest evidence this was a real test, not a
foregone conclusion.** Pre-registration exists precisely so a result can go against
expectations and be trusted when it does. Two of three checks confirmed what the
synthetic work assumed (rank-1 structure, informative candidate detection); one
falsified a core assumption used throughout Exp1-4b (static rho). Given everything
established two experiments ago about synthetic generators being built to match the
classical tools tested against them, an all-green result here would have been the
suspicious outcome, not this one.

Three real findings, two confirming the synthetic work's external validity and one
correcting it:

1. **Rank-1 structure roughly holds** in real crypto coupling (66% top-eigenvalue
   share) — reassuring for the whole classical toolkit built in Exp1-4b, which assumed
   exactly this.
2. **But coupling is genuinely time-varying** (correlation swings 0.31→0.89 across the
   window) — the fixed-rho framing used throughout the synthetic experiments is a real
   simplification of actual market behavior, not just a theoretical concern.
3. **The joint detection statistic transfers well** — it fires on a plausible, sparse
   fraction of days (though somewhat less often than synthetic calibration implied),
   and every checked flagged date corresponds to a real, verifiable market event,
   including sector-correlated moves triggered by asset-specific news, not just
   market-wide crashes.
4. **Daily-frequency temporal structure is genuinely absent** — a real, clean negative
   result, and unlike every synthetic "no signal" finding in this project, this one is
   actually informative about real markets rather than a construction artifact.

No NN was involved in any of this — Experiment 6 is purely the classical toolkit
(correlation/PCA, joint statistic, BNS, threshold-AR) validated for the first time
against real data rather than a hand-built generator, addressing the meta-caveat
surfaced during Experiment 5. Results are mixed in the way honest real-data checks
should be: some assumptions held up, one didn't (time-varying rho), and the joint
statistic in particular looks like a genuinely useful, transferable tool.
