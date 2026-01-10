# LLM provider abstraction

This guide explains how to configure and use the LLM provider abstraction built around
`LLMProviderConfig` and `create_llm()` in `trend_analysis.llm.providers`.

## Setup

1. Install the LLM extras so provider backends are available:
   `pip install -e ".[llm]"`
2. Choose a provider (`openai`, `anthropic`, or `ollama`) and the model name you want.
3. Set provider credentials or endpoints in your environment:
   - OpenAI: `OPENAI_API_KEY`
   - Anthropic: `ANTHROPIC_API_KEY`
   - Ollama (local): ensure the Ollama daemon is running and set `base_url` if needed.
4. Decide how you want to configure the provider:
   - Python: build a `LLMProviderConfig` and call `create_llm()`.
   - Config-driven: pass provider settings through the `llm` config section (if you are
     using config patching or the CLI).
5. Validate the provider by running a simple request (or a dry run) before integrating
   it into larger workflows.

## Configuration examples

### Example 1: OpenAI with environment API key

OpenAI uses `OPENAI_API_KEY` and supports optional retry/timeout tuning.

```python
import os

from trend_analysis.llm.providers import LLMProviderConfig, create_llm

config = LLMProviderConfig(
    provider="openai",
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=30,
    max_retries=2,
)

llm = create_llm(config)
```

### Example 2: Anthropic with explicit overrides

Anthropic uses `ANTHROPIC_API_KEY` and supports additional model params via `extra`.

```python
import os

from trend_analysis.llm.providers import LLMProviderConfig, create_llm

config = LLMProviderConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20240620",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    timeout=45,
    max_retries=3,
    extra={"temperature": 0.2, "max_tokens": 2048},
)

llm = create_llm(config)
```

### Example 3: Ollama local model with extra overrides

```python
from trend_analysis.llm.providers import LLMProviderConfig, create_llm

config = LLMProviderConfig(
    provider="ollama",
    model="llama3.1",
    base_url="http://localhost:11434",
    extra={"temperature": 0.1},
)

llm = create_llm(config)
```

### Example 4: Config-driven setup (YAML)

If you load provider settings from YAML, map the data into `LLMProviderConfig` in your
application code and inject secrets from the environment (do not commit API keys).

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: "<set-from-env-at-runtime>"
  timeout: 30
  max_retries: 2
  extra:
    temperature: 0.1
```

## Troubleshooting

- Missing provider dependency
  - Symptom: `RuntimeError: Provider dependency 'langchain_openai' is not installed.`
  - Fix: install the extras (`pip install -e ".[llm]"`) and retry.
- Missing API key
  - Symptom: `ValueError: API key is required` or provider-specific auth error.
  - Fix: set the appropriate env var (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) and
    restart the process to refresh the environment.
- Authentication failures
  - Symptom: 401/403 responses or `Invalid API key` errors from the provider.
  - Fix: verify the relevant API key env var (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) and
    restart the process so the environment is refreshed.
- Ollama connection errors
  - Symptom: connection refused or timeout when using the `ollama` provider.
  - Fix: confirm the Ollama daemon is running and `base_url` points to the correct host/port.
- Timeouts or slow responses
  - Symptom: request timeouts or unusually long response times.
  - Fix: increase `timeout`, reduce model size, or lower `max_tokens` via `extra`.
- Rate limit or quota errors
  - Symptom: `429` responses or `Rate limit exceeded`.
  - Fix: reduce request volume, add backoff (via `max_retries`), or wait for quota reset.
- Unknown model name
  - Symptom: `Invalid model` or `Model not found` errors.
  - Fix: verify the model name is supported by the provider and matches your account access.
- Unknown provider name
  - Symptom: `ValueError: Unknown provider: ...`
  - Fix: use one of the supported providers: `openai`, `anthropic`, or `ollama`.
