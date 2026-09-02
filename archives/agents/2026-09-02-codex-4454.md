## Scope
The Explain Results Q&A flow treats metrics like "turnover" as a known request keyword, but the metric catalog currently does not include turnover/cost diagnostics. This causes unnecessary "requested data is unavailable" refusals for questions about turnover or transaction costs, even when the underlying analysis output contains those values.


The Explain Results Q&A flow treats metrics like "turnover" as a known request keyword, but the metric catalog currently does not include turnover/cost diagnostics. This causes unnecessary "requested data is unavailable" refusals for questions about turnover or transaction costs, even when the underlying analysis output contains those values.

## Tasks
- [x] Add extraction of `risk_diagnostics` scalar fields in `src/trend_analysis/llm/result_metrics.py` (support dicts and objects with attributes).
- [x] Include turnover fields (e.g., `turnover`, `turnover_value`) in the metric catalog with source `[from risk_diagnostics]`.
- [x] Include transaction cost fields (e.g., `transaction_cost`, `cost`, `per_trade_bps`, `half_spread_bps`) in the metric catalog when present.
- [x] When `details["turnover"]` is present as a `pd.Series` (multi-period), add scalar summaries under deterministic paths (e.g., `turnover.latest`, `turnover.mean`) with an explicit source label.
- [x] Update metric synonyms / keyword mapping so "turnover" and "transaction cost" are treated as available when entries exist.
- [x] Update `tests/test_result_metric_extraction.py` to validate new entries are extracted and formatted with correct `[from ...]` sources.
- [x] Update `tests/test_result_validation.py` to ensure `detect_unavailable_metric_requests()` does not flag turnover when turnover entries exist.
- [x] Add extraction of `risk_diagnostics` scalar fields in `src/trend_analysis/llm/result_metrics.py` (support dicts and objects with attributes).
- [x] Include turnover fields (e.g., `turnover`, `turnover_value`) in the metric catalog with source `[from risk_diagnostics]`.
- [x] Include transaction cost fields (e.g., `transaction_cost`, `cost`, `per_trade_bps`, `half_spread_bps`) in the metric catalog when present.
- [x] When `details["turnover"]` is present as a `pd.Series` (multi-period), add scalar summaries under deterministic paths (e.g., `turnover.latest`, `turnover.mean`) with an explicit source label.
- [x] Update metric synonyms / keyword mapping so "turnover" and "transaction cost" are treated as available when entries exist.
- [x] Update `tests/test_result_metric_extraction.py` to validate new entries are extracted and formatted with correct `[from ...]` sources.
- [x] Update `tests/test_result_validation.py` to ensure `detect_unavailable_metric_requests()` does not flag turnover when turnover entries exist.

## Acceptance Criteria
- [x] `extract_metric_catalog()` returns at least one turnover entry when turnover is present in `risk_diagnostics` or `details["turnover"]`.
- [x] `extract_metric_catalog()` returns at least one transaction cost entry when cost fields exist in `risk_diagnostics`.
- [x] `format_metric_catalog()` renders these entries with explicit `[from <source>]` annotations.
- [x] `detect_unavailable_metric_requests("Report turnover", entries)` returns no missing turnover keyword when turnover entries exist.
- [x] All unit tests pass.
- [x] `extract_metric_catalog()` returns at least one turnover entry when turnover is present in `risk_diagnostics` or `details["turnover"]`.
- [x] `extract_metric_catalog()` returns at least one transaction cost entry when cost fields exist in `risk_diagnostics`.
- [x] `format_metric_catalog()` renders these entries with explicit `[from <source>]` annotations.
- [x] `detect_unavailable_metric_requests("Report turnover", entries)` returns no missing turnover keyword when turnover entries exist.
- [x] All unit tests pass.
