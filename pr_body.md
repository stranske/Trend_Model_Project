<!-- pr-preamble:start -->
> **Source:** Issue #4185

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Follow-up for the ConfigPatch chain reliability checks, focusing on structured output conformance, safety rules, and CI verification.

#### Tasks
- [x] Develop and integrate an automated test suite that sends a batch of at least 100 test cases to the chain and verifies that ≥95% of the structured outputs conform to the expected JSON schema.
- [x] Implement a test case where an input containing unknown keys is provided and assert that the response either omits these unknown keys or flags them as errors.
- [x] Refactor the chain configuration to expose temperature and model settings through external parameters (e.g., environment variables or configuration arguments) and update the documentation accordingly.
- [x] Add tests to verify that temperature and model settings can be customized via external parameters.
- [x] Update the prompt template to include explicit safety rules that instruct the model not to invent additional keys or output unsafe content.
- [x] Create tests to verify that the safety rules in the prompt template are enforced when edge-case instructions are given.
- [ ] Investigate and trigger the CI workflow 'pr-00-gate.yml' to ensure full CI verification across all required workflows.

#### Acceptance criteria
- [x] At least 95% of the 100 structured output test cases conform to the expected JSON schema.
- [x] The system either omits unknown keys or flags them as errors in the output.
- [x] Temperature and model settings are configurable via environment variables or configuration arguments.
- [x] Documentation includes clear instructions on setting and modifying temperature and model parameters.
- [x] The prompt template includes explicit safety rules that prevent the model from inventing additional keys.
- [ ] The CI workflow 'pr-00-gate.yml' runs successfully and passes all required checks.

<!-- auto-status-summary:end -->
