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
4. Build a `LLMProviderConfig` and call `create_llm()` to get a LangChain client.

## Configuration examples

### Example 1: OpenAI with environment API key

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

### Example 2: Ollama local model with extra overrides

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

- Missing provider dependency
  - Symptom: `RuntimeError: Provider dependency 'langchain_openai' is not installed.`
  - Fix: install the extras (`pip install -e ".[llm]"`) and retry.
- Authentication failures
  - Symptom: 401/403 responses or `Invalid API key` errors from the provider.
  - Fix: verify the relevant API key env var (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) and
    restart the process so the environment is refreshed.
- Ollama connection errors
  - Symptom: connection refused or timeout when using the `ollama` provider.
  - Fix: confirm the Ollama daemon is running and `base_url` points to the correct host/port.
- Unknown provider name
  - Symptom: `ValueError: Unknown provider: ...`
  - Fix: use one of the supported providers: `openai`, `anthropic`, or `ollama`.
