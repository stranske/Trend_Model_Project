# Issue #3682 — Universe membership enforcement

Keepalive workflow note: update this document whenever scope items, tasks, or acceptance criteria change status so the workflow can keep nudging until completion. Only check items off after their requirements are verifiably satisfied.

## Scope
- [ ] Load `Trend Universe Membership.csv` into a date×instrument boolean mask.
- [ ] Intersect the mask with availability (non-NaN price history) at each rebalance.
- [ ] On conflicts (in membership but no data, or vice versa), raise or skip by policy with a log entry.
- [ ] Tests: assets entering/exiting mid-sample and sparse histories.

## Tasks
1. [ ] Build `membership_mask(date)`.
2. [ ] Apply mask in weight construction/backtest.
3. [ ] Add tests for entry/exit and partial histories.

## Acceptance criteria
- [ ] No positions in instruments outside membership on any rebalance date.
- [ ] Tests cover both entry and exit scenarios.

## Status notes
- Initial bootstrap; no scope items, tasks, or acceptance criteria have been completed yet. Update sections above as progress is made.
