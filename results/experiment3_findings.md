# Experiment 3 (Disentanglement Phase) — Findings

Before building a two-gate NN architecture meant to separate diffusive coupling (`rho`)
from jump-timing coupling (`jump_coupling`), we checked two cheap things: does the
*existing* single-gate model already conflate the two (Phase A.5), and can a *classical*
combination disentangle them instead (Phase A, then a retest of A with a properly
calibrated jump test)? All of this ran before touching architecture, per the same
sequencing used for Experiment 2.

## Phase A.5 — does the existing single-gate model conflate rho and jump_coupling?

Test: hold `rho=0.5` fixed, vary `jump_coupling` from 0.2 to 0.8, retrain, see if `alpha`
moves. (Experiment 1 already showed alpha tracks `rho` at fixed `jump_coupling` — this
fills in the other arm.)

**Result: r(alpha, jump_coupling) = −0.041.** Alpha values across the grid: 0.719, 0.707,
0.723, 0.700, 0.721 — noise-level, no systematic movement at all.

This was a genuine surprise — the intuitive hypothesis was that jumps (simultaneous large
cross-sectional moves) would look identical to diffusive coupling from the model's point
of view and get conflated into the same gate. Instead alpha is essentially **indifferent**
to jump structure, in either direction. Likely cause: jump events are rare (~3-4 per path
here) against thousands of ordinary timesteps; under an MSE loss averaged over the whole
panel, jump-timing coupling gets diluted into irrelevance. This is not clean
disentanglement — it's the aggregate loss not "seeing" jumps at all. Implication: a
jump-aware pathway bolted onto the same raw-return MSE task would likely hit the same
dilution problem; it would need a jump-specific objective (e.g. predicting jump-occurrence
indicators directly) to have a chance.

## Phase A — classical disentanglement, naive jump test

Bipower variation (BNS) estimates each asset's diffusive variance robustly to jumps;
flag a return as a jump if `r_t^2 > 9 * bipower_variation`. Two correlation signals:
`diffusive_corr` (on the jump-scrubbed series) and `jump_corr` (co-occurrence of jump
flags across assets).

| signal | r(own param) | r(other param) |
|---|---|---|
| diffusive_corr | r(rho) = 1.000 | r(jump_coupling) = 0.070 |
| jump_corr | r(jump_coupling) = 0.994 | **r(rho) = 0.818** |

`diffusive_corr` disentangles cleanly. `jump_corr` is badly contaminated by `rho`.

## Phase A retest — properly calibrated jump test (BNS/Huang-Tauchen ratio statistic)

Hypothesis going in: the naive test is a fixed multiplier, not a calibrated statistical
test — it has no defined false-positive rate, so nothing stops co-occurring false
positives (inevitable once `rho` correlates the underlying series) from producing a
spurious `jump_corr`-vs-`rho` relationship. Fix: a block-level ratio statistic
`z = ((RV-BV)/RV) / sqrt(vartheta/K * max(1, TQ/BV^2))`, normalized by realized tripower
quarticity — asymptotically N(0,1) under "no jump in this block," giving a real,
controllable false-positive rate (`src/baseline.py::_bns_jump_flags`).

**Sanity check first (isolated from rho):** on 2000 steps of pure Gaussian noise, the
naive test flagged 7 spurious jumps; the calibrated test flagged 0. Three injected jumps
were both recovered exactly by the calibrated test. The false-positive-rate problem is
real, and the fix works, in isolation.

**Re-running the disentanglement grid with the calibrated test:**

| method | jump_corr magnitude (range across grid) | r(jump_corr, rho) — contamination |
|---|---|---|
| naive | 0.10 – 0.49 | 0.818 |
| bns (calibrated) | 0.04 – 0.15 | **0.803** |

The calibrated test roughly halves the raw `jump_corr` values (it's correctly filtering
out more spurious co-movement overall) — **but the contamination correlation barely
moves** (0.818 → 0.803). Better calibration fixed the false-positive *rate*; it did not
fix the *contamination*.

### Why: this isn't a calibration bug, it's a structural identifiability limit

At `rho→1`, each asset's diffusive shock is dominated by the shared common factor
(idiosyncratic weight `sqrt(1-rho)→0`). A properly-calibrated per-asset test can be
exactly as rare and well-behaved as intended and *still* fire simultaneously across
assets whenever that shared factor happens to draw an extreme value — because at high
rho the assets' returns are, by construction, nearly identical draws of the same random
variable. A single large, purely diffusive, shared draw and a genuine coordinated jump
are close to observationally identical at this sampling frequency (daily, `dt=1/252`) —
both are "large, simultaneous, cross-sectionally correlated moves." The classical
jump-detection literature's usual identification argument (diffusive increments shrink
like `sqrt(dt)`, jump sizes don't) needs much higher-frequency sampling than daily bars
to bite; it doesn't help here.

This also reveals *why* the fix couldn't work by construction: both the naive and
calibrated tests decide "is this a jump" **per asset, independently**, then correlate
the resulting flags after the fact. The confound lives in the *joint* cross-sectional
pattern, not in any single asset's marginal distribution — no per-asset threshold, however
well-calibrated, can fix a problem that only exists in the relationship between assets.

**One concrete, testable idea this suggests, if Phase B is ever scoped:** the generator's
own structure gives a real, exploitable difference between the two mechanisms that no
per-asset test can see. A shared diffusive draw moves every asset in a *fixed, known
ratio* — proportional to each asset's own `sigma_i`, since `diffusion_i = sigma_i *
sqrt(rho) * z_common + idio` — a deterministic, rank-1 pattern. A common jump, by
contrast, draws an *independent* jump size per asset even when the timing is shared (per
`jump_diffusion_generator.py`'s own docstring) — the cross-sectional *shape* of the move
looks unstructured, not proportional to vol. Telling these apart requires looking at the
**pattern across the whole cross-section jointly** (is this vector of simultaneous moves
proportional to known vol ratios, or does it look like independent draws?), not
thresholding each asset's own return in isolation. This is exactly the kind of joint,
whole-cross-section computation an attention-based pathway is structurally suited to do
that a per-asset statistical test structurally cannot — a real candidate design for Phase
B's cross-sectional pathway, if it gets built, rather than another per-asset outlier
detector wearing a neural net.

## Verdict

Three negative-or-null results now, but of increasing specificity:

1. Experiment 2: NN doesn't beat a two-line correlation calculation on the *diffusive*
   coupling question at all.
2. Experiment 3 Phase A.5: the existing single-gate NN doesn't conflate jump and
   diffusive coupling — but only because it's blind to jump structure entirely, not
   because it disentangles them.
3. Experiment 3 Phase A retest: a properly calibrated classical jump test still can't
   cleanly separate the two, for a specific, understood structural reason (per-asset
   testing can't resolve a joint cross-sectional confound) — not because nobody tried
   hard enough classically.

(3) is the sharper, more useful negative result the earlier informal bet anticipated: it
rules out "just use a better classical test" and leaves a specific, falsifiable
hypothesis for what *would* actually need to differ architecturally (joint
cross-sectional pattern recognition, not per-asset jump flagging, combined with a
jump-specific training objective per finding #2). That is a real, narrower case for
Phase B than existed before this diagnostic — worth scoping deliberately if pursued,
rather than as a default "try the NN" step.
