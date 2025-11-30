<!-- bootstrap for codex on issue #3639 -->

## Scope
- [ ] Consolidate all historical-signal rolling/shift logic behind a single helper so future calculations cannot accidentally peek at same-day values.

## Tasks
- [ ] Implement `rolling_shifted(series, window, agg, min_periods=None)` which applies `series.shift(1)` before rolling.
- [ ] Support aggregations: mean, std, sum, max, min; add a hook to pass a callable.
- [ ] Replace ad-hoc rolling code with the helper across signals.
- [ ] Add tests verifying that today’s value never uses same-day input.

## Acceptance criteria
- [ ] All signal calculations that rely on history call the helper.
- [ ] Tests show identical results to the previous correct implementations and no look-ahead.

## Status
- _As of this sync no scope items, tasks, or acceptance criteria have been satisfied; all checkboxes remain unchecked and will only flip after the corresponding work and validation land._
