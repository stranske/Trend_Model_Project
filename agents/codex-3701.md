<!-- bootstrap for codex on issue #3701 -->

# Codex Checklist for Issue #3701

## Scope
- [x] Load `Trend Universe Membership.csv` into a date×instrument boolean mask.
- [x] Intersect the membership mask with availability (non-NaN price history) at each rebalance.
- [x] On conflicts (in membership but no data, or vice versa), raise or skip by policy with a log entry.
- [x] Add tests for assets entering/exiting mid-sample and sparse histories.

## Tasks
- [x] Build `membership_mask(date)`.
- [x] Apply the mask inside weight construction/backtest so trading only occurs in the in-universe names.
- [x] Add automated tests for entry/exit boundaries and sparse histories.

## Acceptance Criteria
- [x] No positions in instruments outside membership on any rebalance date.
- [x] Tests cover both entry and exit scenarios (including limited history edge cases).
