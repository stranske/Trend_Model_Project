<!-- bootstrap for codex on issue #4294 -->
## Why
PR #4273 addressed issue #4185, but verification identified concerns (verdict: **CONCERNS**). This follow-up addresses the remaining gaps with improved task structure to ensure safety, reliability, and configurability.

## Tasks
- [x] [#4273](https://github.com/stranske/Trend_Model_Project/issues/4273)
- [x] [#4185](https://github.com/stranske/Trend_Model_Project/issues/4185)

## Acceptance Criteria
- [x] Update the prompt template to include a complete and explicit safety rules section that instructs the model not to invent keys and to flag unknown or extraneous keys.
- [x] Write and add unit tests that sample a statistically significant set of cases (e.g., 100 cases) to measure structured output parsing accuracy, enforcing a ≥95% success rate and failing the CI if the threshold is not met.
- [x] Implement a mechanism to flag or reject unknown keys in the output, and create dedicated tests that supply inputs with unknown keys to assert that these keys are not silently passed through.
- [x] Refactor the chain to allow temperature and model settings to be configurable externally, for example via environment variables, configuration parameters, or a settings file, and update tests/documentation to validate this configurability.
- [x] Enhance the integration test suite to include at least two LLM providers (e.g., OpenAI and Anthropic) to ensure provider-specific structured output enforcement behavior is correctly handled.
- [x] Investigate and resolve the pending CI workflow 'pr-00-gate.yml' so that it runs and verifies all required tests along with the primary CI workflow.
