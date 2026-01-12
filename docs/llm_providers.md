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

The provider config accepts the following common fields:

- `provider`: one of `openai`, `anthropic`, or `ollama`
- `model`: provider-specific model name
- `api_key`: API key string (usually supplied from env vars)
- `base_url`: override for self-hosted or local endpoints (Ollama)
- `timeout`: request timeout in seconds
- `max_retries`: retry attempts inside the provider SDK
- `extra`: provider-specific kwargs (e.g., `temperature`, `max_tokens`)

### Example 1: OpenAI with environment API key (Python + YAML)

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

### Example 2: Anthropic with explicit overrides (Python + YAML)

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

```yaml
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20240620
  api_key: "<set-from-env-at-runtime>"
  timeout: 45
  max_retries: 3
  extra:
    temperature: 0.2
    max_tokens: 2048
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

## Troubleshooting

| Error or symptom | Likely cause | Resolution |
| --- | --- | --- |
| `RuntimeError: Provider dependency 'langchain_openai' is not installed.` | LLM extras not installed | Run `pip install -e ".[llm]"` and retry. |
| `ValueError: API key is required` | Missing API key environment variable | Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, then restart the process. |
| `Invalid API key` / `401` / `403` | Incorrect or expired credentials | Rotate the key and update the env var, then restart. |
| Connection refused / timeout (Ollama) | Ollama daemon not running or wrong `base_url` | Start Ollama and confirm `base_url` matches the host/port. |
| `Rate limit exceeded` / `429` | Provider rate limit or quota exhaustion | Reduce request volume, raise `max_retries`, or wait for quota reset. |
| `Invalid model` / `Model not found` | Model name not supported for your account | Verify the model name and account access. |
| `ValueError: Unknown provider: ...` | Unsupported provider value | Use `openai`, `anthropic`, or `ollama`. |
