# Turnover & Churn Heuristics (Debugging Reference)

This note is a **non-binding** set of rough heuristics to calibrate expectations and spot likely bugs quickly.
It is meant to prevent wasting time interpreting outputs that are obviously wrong (too much churn or too little).

## 1) First principles: what drives churn?

### The key quantities
- **Universe size** $N$: number of eligible candidates (after inception/history gating).
- **Portfolio size** $H$: number of holdings actually held.
  - Bounded by `target_n` and `mp_min_funds` / `mp_max_funds` (or equivalent).
- **Decision cadence**: how often membership can change.
  - *Important:* exports may include annual “periods” while membership changes occur at quarterly (or other) rebalances.

### The most common mistake
Always distinguish:
- **Event logs** (e.g., “skipped due to cooldown/turnover budget”) vs
- **Realized churn** (changes in the set of **non-zero weighted holdings**).

If you see “hundreds of changes,” first confirm whether you are counting *events* rather than *membership changes*.

## 2) Rough z-score exit expectations (good for order-of-magnitude)

If a selection score is z-scored cross-sectionally and is *approximately* standardized:
- Hard exit threshold $z < z_{hard}$ has probability $p_{hard} \approx \Phi(z_{hard})$.
  - Example: $z_{hard}=-1.5$ gives $p_{hard}\approx 0.0668$.
- Soft exit threshold $z < z_{soft}$ with “two strikes” has probability
  $p_{soft2} \approx \Phi(z_{soft})^2$ under independence.
  - Example: $z_{soft}=-0.5$ gives $p_{soft2}\approx 0.3085^2\approx 0.095$.

### Convert tail probabilities into expected exits
Exits are driven by the holdings set (size $H$), not the full universe.

- Expected hard exits per decision step: $\mathbb{E}[X_{hard}] \approx H\,p_{hard}$.
- Expected two-strike exits per decision step: $\mathbb{E}[X_{soft2}] \approx H\,p_{soft2}$.

With $H\in[6,12]$, these expectations are typically **sub-1 to low-single-digits per step**.
Correlations and regime clustering can create occasional spikes, but sustained double-digit exits per step should be treated as suspect.

## 3) Rank-based selection (no explicit z-threshold)

For pure rank selection, a useful mental model is:
- Membership changes are controlled primarily by:
  - `target_n` / min/max holdings,
  - tenure/cooldown/stickiness,
  - turnover budget constraints,
  - the stability of ranks across adjacent decision points.

Heuristic expectation:
- If ranks are noisy and there are no stickiness constraints, churn can be higher.
- With stickiness + cooldown + min tenure, realized churn should be meaningfully damped.

## 4) How universe size affects intuition

Universe size $N$ mainly affects **cross-sectional tail counts**, not direct exits.
- In the universe, the expected count below a threshold is roughly $N\,\Phi(z)$.
  - Example: with $N=40$, expected count below $-1.5$ is ~$2.7$.

But portfolio churn is still bounded by holdings size $H$ and constraints.
Therefore, use $N$ mostly to sanity-check whether the score distribution is behaving like a standardized cross-section.

## 5) Quick “smell tests” (parameter-aware, not hard rules)

These are prompts to trigger a faster debug cycle:

### Too much churn (likely bug or miscount)
- You see large churn totals but the **non-zero weight membership** barely changes, or vice versa.
- You see frequent rebalances where **adds + drops** is far above the intended cap.
- Churn is dominated by `mp_min_funds` reseeding rather than score triggers.

### Too little churn (likely exits blocked or score degenerate)
- Almost no exits across many decision points even though:
  - hard/soft exits are configured,
  - and scores show meaningful variation.
- Cross-sectional score std is near-zero (zscore calculation degenerates), producing no tail events.

### “Z-score looks wrong” checks
Even if z-scores are not truly normal, they should not look broken.
- If the observed tail frequency at $z<-0.5$ or $z<-1.5$ is *orders of magnitude* off vs $N\Phi(z)$, suspect:
  - wrong frame used for z-score,
  - inclusion of non-investable series,
  - degenerate variance from missing/inactive series,
  - incorrect standardization scope.

## 6) Recommended debugging workflow (fast)

1. Compute realized membership changes from non-zero weights (not event logs).
2. Verify eligibility gating (inception + min history) so the universe is stable and plausible.
3. Inspect cross-sectional z-score health: mean ~0, std ~1 (roughly), tails non-degenerate.
4. Only then interpret additions/drops against soft/hard exit logic.

## 7) Notes on parameter sensitivity

When updating intuition, these knobs matter most:
- $N$: eligibility gating + missing policy + inception/history rules.
- $H$ bounds: `target_n`, `mp_min_funds`, `mp_max_funds`.
- Selection mode: rank vs z-threshold vs blended.
- Exit logic: hard threshold, soft threshold, strike count, cooldown, tenure, stickiness.
- Budget semantics: whether caps apply to additions only vs additions+drops.

This document is intentionally non-prescriptive; it exists to keep calibration consistent and to prompt early suspicion when results look implausible.
