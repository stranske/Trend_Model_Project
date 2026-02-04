# Keepalive Status — PR #4667

## Scope
PR #4667 aimed to resolve issue #4666 but failed to meet the acceptance criteria. This follow-up issue addresses the remaining gaps by focusing on implementing regime-conditional turnover caps, tracking realized turnover, and enhancing diagnostic outputs.

## Progress
7/7 tasks complete, 0 remaining.

## Checklist Reconciliation
Checklist reconciled on 2026-02-04 after reviewing recent commits and running `pytest tests/monte_carlo/test_runner_turnover_diagnostics.py -m "not slow"` plus `pytest tests/app/test_nl_operation_viewer_component.py -m "not slow"`.

## Tasks
- [x] Update the configuration schema/model to add and validate a new 'max_turnover' field that accepts regime-based turnover cap values.
- [x] Modify the Monte Carlo runner to record the realized turnover for each period and strategy path.
- [x] Enhance the Monte Carlo evaluation output to include an indicator for each period showing whether the turnover cap was binding.
- [x] Develop unit tests to confirm that regime-conditional turnover caps are correctly implemented and affect runner behavior.
- [x] Develop unit tests to verify that the Monte Carlo runner records the realized turnover accurately per period and strategy path.
- [x] Develop unit tests for the diagnostic output to confirm the presence and correctness of the cap-binding indicator.
- [x] Review and enhance redaction in the Streamlit NL operation viewer to ensure replay functionality does not leak sensitive data.

## Acceptance Criteria
- [x] The 'max_turnover' field in the configuration schema accepts an object with regime names as keys and numerical turnover caps as values, and the schema validation fails if this structure is not followed.
- [x] The Monte Carlo runner records the realized turnover for each period and strategy path, and this data is accessible via the evaluation objects.
- [x] The Monte Carlo evaluation output includes an explicit indicator showing whether the turnover cap was binding during each evaluation period.
- [x] Unit tests confirm that regime-conditional turnover caps are accepted through the configuration, correctly interpreted, and affect runner behavior as expected.
- [x] Unit tests verify that the Monte Carlo runner records the realized turnover for each period and strategy path accurately by comparing against known inputs/outputs.
- [x] Unit tests validate that the evaluation outputs include diagnostics indicating when turnover caps were binding, ensuring these indicators reflect the underlying runner behavior correctly.
- [x] The Streamlit NL operation viewer redacts sensitive data appropriately to ensure replay functionality does not leak sensitive data.
