# AGENT_ISSUE_TEMPLATE

## Title
[NL-7-Completion] Complete ConfigPatch Chain Testing, Validation, and Provider Support

## Why
Issue #4185 implemented the core ConfigPatchChain infrastructure but left several critical acceptance criteria unmet. The chain lacks comprehensive testing, multi-provider support, retry logic, and programmatic validation of unknown keys. This issue completes the work to make the NL config system production-ready.

## Context
- **Parent Issue**: #4185 (merged in PR #4273)
- **What Exists**: Core chain, prompt templates, schema selection, basic unit tests
- **What's Missing**: Eval harness, provider abstraction, retry logic, unknown key detection, comprehensive test coverage

## Scope
Complete the remaining acceptance criteria from #4185:
1. Build an eval harness to test prompt effectiveness with real-world instructions
2. Implement and test multi-provider support (OpenAI, Anthropic, Ollama)
3. Add retry logic with error feedback
4. Implement programmatic unknown key validation
5. Achieve measurable ≥95% structured output parsing reliability

## Non-Goals
- Redesigning the existing ConfigPatchChain API (maintain backward compatibility)
- UI integration (separate issue)
- Full agent capabilities (this remains a deterministic chain)

## Tasks

### Eval Harness & Testing
- [ ] Create eval harness script at `tools/eval_config_patch.py`:
  - [ ] Load test cases from JSON/YAML
  - [ ] Run ConfigPatchChain for each test case
  - [ ] Validate output structure and correctness
  - [ ] Calculate success rate metrics
  - [ ] Generate evaluation report

- [ ] Define test cases in `tools/eval_test_cases.yml`:
  - [ ] "Use risk parity weighting" → set analysis.weighting.scheme = "risk_parity"
  - [ ] "Select top 12 funds" → set analysis.top_n = 12
  - [ ] "Remove position limits" → remove constraints.max_weight + REMOVES_CONSTRAINT flag
  - [ ] "Target 15% volatility" → set analysis.target_vol = 0.15
  - [ ] "Use monthly frequency and risk parity" → Two operations in one patch
  - [ ] Add 5 additional edge cases (unknown keys, conflicting instructions, ambiguous requests)

- [ ] Add comprehensive integration tests in `tests/test_config_patch_chain_integration.py`:
  - [ ] Test each eval harness case programmatically
  - [ ] Verify risk flags are correctly detected
  - [ ] Test with mocked LLM responses
  - [ ] Achieve ≥95% success rate on test suite

### Unknown Key Validation
- [ ] Implement `validate_patch_keys()` function in `src/trend_analysis/llm/validation.py`:
  - [ ] Check each operation path against schema
  - [ ] Return list of unknown keys
  - [ ] Add `needs_review: bool` field to ConfigPatch model

- [ ] Integrate validation into ConfigPatchChain.run():
  - [ ] Call validate_patch_keys() after parsing
  - [ ] Set needs_review=True if unknown keys found
  - [ ] Log warnings for unknown keys

- [ ] Add tests for unknown key detection:
  - [ ] Test with valid keys (should pass)
  - [ ] Test with unknown keys (should flag)
  - [ ] Test with typos (should suggest corrections)

### Retry Logic
- [ ] Implement retry logic in ConfigPatchChain.run():
  - [ ] Catch parsing failures (JSONDecodeError, ValidationError)
  - [ ] Retry up to `self.retries` times
  - [ ] Include previous error in retry prompt
  - [ ] Return error details if all retries exhausted

- [ ] Add retry prompt template in `src/trend_analysis/llm/prompts.py`:
  - [ ] Include original instruction
  - [ ] Include error message from failed attempt
  - [ ] Request corrected output

- [ ] Add retry tests:
  - [ ] Test successful retry after parse failure
  - [ ] Test exhausted retries scenario
  - [ ] Verify error details in response

### Provider Abstraction
- [ ] Create provider abstraction layer in `src/trend_analysis/llm/providers.py`:
  - [ ] Define `LLMProviderConfig` dataclass
  - [ ] Implement `create_llm()` factory function
  - [ ] Support OpenAI provider (langchain-openai)
  - [ ] Support Anthropic provider (langchain-anthropic)
  - [ ] Support Ollama provider (langchain-ollama)
  - [ ] Add environment variable detection for API keys

- [ ] Update pyproject.toml dependencies:
  - [ ] Add langchain-openai to llm extras
  - [ ] Add langchain-anthropic to llm extras
  - [ ] Add langchain-ollama to llm extras (optional for local)

- [ ] Add provider integration tests in `tests/test_llm_providers.py`:
  - [ ] Test OpenAI provider creation (with/without key)
  - [ ] Test Anthropic provider creation (with/without key)
  - [ ] Test Ollama provider creation
  - [ ] Test provider factory selection logic

- [ ] Document provider usage in `docs/llm_providers.md`:
  - [ ] Environment variable requirements
  - [ ] Example usage for each provider
  - [ ] Cost and latency considerations

### Documentation & Examples
- [ ] Create example script `examples/nl_config_demo.py`:
  - [ ] Show basic ConfigPatchChain usage
  - [ ] Demonstrate provider switching
  - [ ] Show error handling
  - [ ] Include sample instructions

- [ ] Update README or create `docs/natural_language_config.md`:
  - [ ] Explain ConfigPatchChain purpose
  - [ ] Link to eval harness
  - [ ] Document safety rules
  - [ ] Provide troubleshooting tips

## Acceptance Criteria
- [ ] Eval harness exists at `tools/eval_config_patch.py` with ≥10 test cases
- [ ] Test suite achieves ≥95% success rate (measured and documented)
- [ ] Unknown keys are detected programmatically and flagged with `needs_review=True`
- [ ] Retry logic is implemented and tested (catches parse failures, includes error in retry)
- [ ] ConfigPatchChain works with at least 2 LLM providers (tested with integration tests)
- [ ] All provider dependencies are in pyproject.toml with proper version constraints
- [ ] Example script demonstrates usage with different providers
- [ ] Documentation explains setup, usage, and troubleshooting

## Implementation Notes

### Eval Harness Structure
```python
# tools/eval_config_patch.py
from pathlib import Path
import yaml
from trend_analysis.llm.chain import ConfigPatchChain
from trend_analysis.llm.prompts import build_config_patch_prompt

def load_test_cases(path: Path) -> list[dict]:
    """Load test cases from YAML file."""
    return yaml.safe_load(path.read_text())

def run_eval(chain: ConfigPatchChain, test_cases: list[dict]) -> dict:
    """Run eval on test cases, return metrics."""
    results = []
    for case in test_cases:
        try:
            patch = chain.run(
                current_config=case["config"],
                instruction=case["instruction"]
            )
            success = validate_patch(patch, case["expected"])
            results.append({"case": case["id"], "success": success})
        except Exception as e:
            results.append({"case": case["id"], "success": False, "error": str(e)})
    
    success_rate = sum(r["success"] for r in results) / len(results)
    return {"success_rate": success_rate, "results": results}
```

### Provider Factory Pattern
```python
# src/trend_analysis/llm/providers.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class LLMProviderConfig:
    provider: Literal["openai", "anthropic", "ollama"]
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None

def create_llm(config: LLMProviderConfig):
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.model, api_key=config.api_key)
    elif config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.model, api_key=config.api_key)
    elif config.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.model, base_url=config.base_url)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
```

### Retry Logic
```python
# In ConfigPatchChain.run()
def run(self, *, current_config, instruction, **kwargs) -> ConfigPatch:
    prompt_text = self.build_prompt(current_config, instruction, **kwargs)
    
    last_error = None
    for attempt in range(self.retries):
        try:
            response_text = self._invoke_llm(prompt_text)
            patch = self._parse_patch(response_text)
            # Validate unknown keys
            unknown_keys = validate_patch_keys(patch, self.schema)
            if unknown_keys:
                patch.needs_review = True
                logger.warning(f"Unknown keys detected: {unknown_keys}")
            return patch
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            if attempt < self.retries - 1:
                # Retry with error feedback
                prompt_text = build_retry_prompt(
                    original_instruction=instruction,
                    error_message=str(e),
                    failed_response=response_text
                )
                continue
            raise RuntimeError(f"All {self.retries} retries failed") from last_error
```

## Deferred Tasks (Requires Human)
- Selecting specific Anthropic and Ollama models for testing
- Deciding cost/latency tradeoffs between providers
- Setting up CI secrets for provider API keys

## Overall Notes
This issue focuses on production-readiness: testing, reliability, and flexibility. The eval harness is critical for measuring the 95% success rate requirement. Multi-provider support ensures users aren't locked into a single LLM vendor.
