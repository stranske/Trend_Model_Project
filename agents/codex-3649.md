<!-- bootstrap for codex on issue #3649 -->

## Scope
- Recomputing identical features inflates runtime. Small caching plus vectorization trims cycles.

## Current status snapshot
The following blocks **must** be copied verbatim into each keepalive update comment. Only check off a task when the acceptance
criteria below are satisfied, and re-post the entire set whenever a box flips state.

### Tasks
- [ ] Identify top 2–3 repeated computations in signals and wrap them with a simple memo (`functools.lru_cache` or dict keyed by params).
- [ ] Replace slow `groupby.apply` with vectorized operations where feasible.
- [ ] Add a one-line timing logger around key stages to verify improvement.

### Acceptance criteria
- [ ] On a medium sample dataset, end-to-end runtime improves meaningfully without altering results.
- [ ] Timing logs show cached stages shrinking materially on successive runs.

## Progress log
- 2024-07-05 – Kickoff. Received keepalive instructions; no tasks have been completed yet and both acceptance criteria remain unmet.
