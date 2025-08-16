# Diversification Guard (Scaffold)

Goal: Enforce per-bucket caps at selection time to improve diversification.

Phase 0 (this commit)
- UI scaffold in Portfolio > Selection policy for:
  - Bucket mapping text area (name: bucket lines)
  - Max per bucket numeric input
- Values persisted to config under `portfolio.selection_policy.params.diversification_guard`.
- No enforcement yet.

Phase 1
- Config: Allow `buckets` mapping (dict) and/or parse `buckets_text` into a dict.
- Engine: During selection, compute bucket counts among candidates and enforce `max_per_bucket` by preferring higher-scoring funds within over-subscribed buckets.
- Conflicts: document conflict resolution when combined with turnover budgets and sticky rules.

Phase 2
- Optional: Support hierarchical buckets and per-bucket thresholds; allow caps by asset class/region.
- Add tests for tie-breaking and edge cases (all candidates same bucket).

Notes
- If bucket metadata is absent for a fund, treat as its own singleton bucket or assign to `Unknown`.
- Consider a grace window to avoid churn when bucket assignments change.
