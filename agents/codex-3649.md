<!-- bootstrap for codex on issue #3649 -->

## Scope
- Recomputing identical features inflates runtime. Small caching plus vectorization trims cycles.

## Tasks
- [ ] Identify top 2–3 repeated computations in signals and wrap them with a simple memo (`functools.lru_cache` or dict keyed by params).
- [ ] Replace slow `groupby.apply` with vectorized operations where feasible.
- [ ] Add a one-line timing logger around key stages to verify improvement.

## Acceptance criteria
- [ ] On a medium sample dataset, end-to-end runtime improves meaningfully without altering results.
- [ ] Timing logs show cached stages shrinking materially on successive runs.
