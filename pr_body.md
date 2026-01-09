<!-- pr-preamble:start -->
> **Source:** Issue #4180

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
This is the core NL component: a LangChain chain that takes a user instruction and current config, then produces a valid `ConfigPatch` as structured output. Using a deterministic chain (not an agent) gives us predictable, auditable behavior.

#### Tasks
- [x] Design prompt template with sections:
- [x] - System prompt explaining the task
- [x] - Current config (or relevant excerpt)
- [x] - Allowed schema (from Issue 4)
- [x] - Safety rules (don't invent keys, flag risks)
- [x] - User instruction
- [x] Implement `ConfigPatchChain` using LangChain:
- [x] Define the class structure. (verify: confirm completion in repo)
- [x] Define methods. (verify: confirm completion in repo)
- [x] Implement the run method. (verify: confirm completion in repo)
- [x] Integrate with LangChain. (verify: confirm completion in repo)
- [ ] Configure model settings for reliability:
- [x] Set low temperature. (verify: confirm completion in repo)
- [ ] Enforce structured output. (verify: confirm completion in repo)
- [ ] Determine max tokens for patch size. (verify: confirm completion in repo)
- [x] Implement schema injection:
- [x] Include compact schema in prompt. (verify: confirm completion in repo)
- [x] Limit to relevant sections based on instruction. (verify: confirm completion in repo)
- [ ] Add self-check step:
- [ ] Validate patch references known keys. (verify: confirm completion in repo)
- [ ] Flag unknown keys for human review. (verify: confirm completion in repo)
- [ ] Implement rejection criteria based on unknown keys. (verify: confirm completion in repo)
- [x] Implement retry logic:
- [x] Implement retry on parsing failure. (verify: confirm completion in repo)
- [x] Include previous error in retry prompt. (verify: confirm completion in repo)
- [ ] Add provider abstraction:
- [ ] Support OpenAI. (verify: confirm completion in repo)
- [ ] Support Anthropic Claude. (verify: confirm completion in repo)
- [ ] Support local Ollama. (verify: confirm completion in repo)

#### Acceptance criteria
- [ ] Chain produces valid ConfigPatch for a defined set of common instructions.
- [ ] Define specific test cases and metrics for structured output parsing reliability (≥95% success rate).
- [ ] Unknown keys are flagged, not silently included.
- [ ] Local eval harness exists for testing prompts.
- [ ] Temperature and model settings are configurable.
- [ ] Chain works with at least 2 LLM providers.

<!-- auto-status-summary:end -->

## Issue #4185 Tasks
### Eval Harness & Testing
- [x] Create eval harness script at `tools/eval_config_patch.py`
- [x] Load test cases from JSON/YAML
- [x] Run ConfigPatchChain for each test case
- [x] Validate output structure and correctness
- [x] Calculate success rate metrics
- [x] Generate evaluation report
- [x] Define test cases in `tools/eval_test_cases.yml`
- [x] "Use risk parity weighting" → set analysis.weighting.scheme = "risk_parity"
- [ ] "Select top 12 funds" → set analysis.top_n = 12
- [ ] "Remove position limits" → remove constraints.max_weight + REMOVES_CONSTRAINT flag
- [ ] "Target 15% volatility" → set analysis.target_vol = 0.15
- [ ] "Use monthly frequency and risk parity" → Two operations in one patch
- [x] Add 5 additional edge cases (unknown keys, conflicting instructions, ambiguous requests)
- [ ] Add comprehensive integration tests in `tests/test_config_patch_chain_integration.py`
- [x] Test each eval harness case programmatically
- [ ] Verify risk flags are correctly detected
- [ ] Test with mocked LLM responses
- [ ] Achieve ≥95% success rate on test suite

## Issue #4300 Follow-up Tasks
### Tasks
- [x] Enhance the retry logic in the configuration patch processing function to retry up to `self.retries` times upon a parse failure. Add automated tests to simulate repeated parse failures and verify error logging.
- [x] Implement the `LLMProviderConfig` dataclass to encapsulate provider configuration parameters.
- [x] Develop the `create_llm()` factory function for dynamic provider instantiation and add unit and integration tests.
- [x] Update the `pyproject.toml` file to include all necessary provider dependencies with explicit version constraints.
- [ ] Create or update the documentation file to include setup instructions, configuration examples, and troubleshooting guidance.
- [x] Develop an example usage script demonstrating configuration and error handling for LLM providers.
- [ ] Expand evaluation test cases to include specific scenarios and integrate them into the eval harness.
- [ ] Review and update the unknown key validation logic to use the latest authoritative schema and handle dynamic keys.

### Acceptance Criteria
- [x] The retry logic in `config_patch.py` attempts to reprocess a configuration patch up to `self.retries` times upon a parse failure, logging each error encountered.
- [x] The `LLMProviderConfig` dataclass encapsulates provider configuration parameters and supports at least 2 LLM providers.
- [x] The `create_llm()` factory function dynamically instantiates LLM providers and passes all unit and integration tests.
- [x] The `pyproject.toml` file includes all necessary provider dependencies with explicit version constraints.
- [x] The `examples/nl_config_demo.py` script demonstrates configuration and error handling for multiple LLM providers.
- [x] The documentation in `docs/llm_providers.md` includes setup instructions, configuration examples, and troubleshooting guidance for the LLM provider abstraction layer.
- [x] The evaluation harness in `tools/eval_config_patch.py` includes at least 10 test cases, covering specified scenarios.
- [x] The unknown key validator in `validation.py` detects keys not defined in the authoritative schema and flags them with `needs_review=True`, handling dynamic array indices and wildcard keys.
