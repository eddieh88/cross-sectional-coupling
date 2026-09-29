# Experiment 2 — Baseline Comparison: Findings

Per `docs/research_spec.md` §4, Experiment 2 is "the actual first real test": does the
learned `alpha` recover `rho` any better, faster (fewer samples), or more robustly
(under jump contamination) than a trivial rolling/realized cross-sectional correlation
computed directly on the same data, no NN involved? The spec is explicit that if the
baseline matches alpha's performance, that's a real, useful negative result, not a
failed exercise — it means the NN machinery isn't earning its complexity for this
question.

**Both estimators see exactly the same generated data at every comparison point** (same
tensor, not separately-drawn samples), so any difference is about the estimator, not
the data. Baseline: `src/baseline.py::realized_corr_estimate` — pools all
(path, timestep) observations and computes one correlation matrix, mean off-diagonal.
Two lines of actual computation, no training, no randomness beyond the data itself.

## 2a — Sample efficiency

Question: does alpha need less data than correlation to recover rho equally well?

Rho grid `[0.0, 0.3, 0.5, 0.7, 1.0]`, three panel sizes (`medium` reuses Experiment 1's
already-trained alpha values at the same config, not retrained):

| sample size | panel | alpha Pearson r | baseline Pearson r |
|---|---|---|---|
| small  | 5 paths × 50 steps (250 obs/asset)   | 0.8660 | 0.99998 |
| medium | 30 paths × 300 steps (9,000 obs/asset) | 0.9945 | 0.99999 |
| large  | 50 paths × 500 steps (25,000 obs/asset) | 0.9911 | 0.99997 |

**Baseline wins outright, at every sample size tested — including the smallest one.**
The two-line correlation calculation is already at Pearson r ≈ 1.000 with only 250
observations per asset; it doesn't need more data to get there. Alpha needs
substantially more data to even approach that (0.866 → 0.994 going from small to
medium), and even at the largest panel tested it doesn't fully catch up (0.991, not
better than medium's 0.994 — alpha's improvement with data isn't even monotonic across
this grid, unlike the baseline's, consistent with training-noise dominating any real
gain from more data at this scale).

Full data: `results/experiment2a_sample_efficiency.csv`, `results/experiment2a_summary.csv`.

## 2b — Robustness to jump contamination

Question: does alpha degrade less than correlation when jump noise increases, holding
`jump_coupling` fixed (only `jump_std` changes — jump *sizes* get noisier, not their
systemic/idiosyncratic split)?

Rho grid `[0.0, 0.3, 0.5, 0.7, 1.0]`, medium panel size (30 × 300), two conditions:

| condition | jump_std | alpha Pearson r | baseline Pearson r |
|---|---|---|---|
| baseline jumps | 0.05 | 0.9945 | 0.99999... |
| high jump noise | 0.15 (3×) | 0.9920 | 0.99999... |

**Both estimators are essentially unaffected in rank-correlation terms**, and the
baseline is still marginally ahead in both conditions. Neither estimator "wins" on
robustness at this noise level.

Worth noting separately from the Pearson-r result: both estimators' **raw values**
compress substantially under higher jump noise (e.g. at rho=1.0: baseline correlation
drops from 0.838 → 0.381; alpha drops from 0.855 → 0.698). This is expected — added
jump noise inflates return variance without adding cross-sectional structure, which
dilutes any correlation-based measure — but it means neither estimator's absolute level
is a stable, noise-invariant readout of rho; only their *relative ordering* across a rho
sweep survives jump contamination well. Since both degrade in the same direction and by
a similar (if not identical) amount, this isn't a differentiator between them either.

Full data: `results/experiment2b_jump_robustness.csv`, `results/experiment2b_summary.csv`.

## Verdict

Per the spec's own decision rule (§6, criterion #2): **the trivial baseline matches or
beats `alpha` on both axes tested (sample efficiency and jump-noise robustness), with
far less compute** (no training, no optimizer instability to debug, no architecture
choices, sub-second to compute). This is exactly the negative result the spec flagged as
the one to take seriously: *"without this, the project is a more expensive way to
compute something correlation already gives you."*

This doesn't invalidate Experiment 1 (the gate does recover a coupling signal — the
mechanism works) — it says that, so far, nothing tested makes the NN worth its
complexity over a two-line calculation. Two things this run did *not* test, both called
out in the spec as the more interesting potential differentiators, and worth resolving
before concluding the idea is dead:

1. **Disentangling diffusive coupling (rho) from jump coupling** — correlation conflates
   the two; a jump-aware gate might not. Untested here (jump_coupling was held fixed
   throughout).
2. **Non-linear / higher-order coupling structure** the generator doesn't produce —
   this generator has a single linear common factor, which is exactly the regime
   correlation is best at. A structurally different generator (Experiment 4) or a
   multi-factor stress test (Experiment 3) is where a nonlinear method would have room
   to show an advantage correlation structurally can't reach.

**Recommendation:** per the spec's own ordering, this is the point to seriously
reconsider continuing down the current path before investing in Experiment 3's full
architecture sweep. If continuing, the disentanglement test (jump vs. diffusive
coupling) is the cheapest remaining way to find a real edge, since it's a dimension
correlation cannot address at all by construction, rather than one where correlation
just happens to already be very good.
