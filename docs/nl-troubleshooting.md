# Natural Language Troubleshooting

Use this guide to resolve common `trend nl` issues.

## Common Errors

### ModuleNotFoundError: No module named 'langchain'

Install the optional LLM extras:

```bash
pip install trend-model[llm]
```

### Python version 3.9 is not supported

The natural language features require Python 3.10+. Upgrade your environment
with `pyenv` or another version manager:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

### Pydantic validation error

Ensure Pydantic v2 is installed:

```bash
pip install "pydantic>=2.0,<3.0"
```

### Missing API keys

Set the provider API key via environment variables. For OpenAI and Anthropic,
use:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

You can also use the generic `TREND_LLM_API_KEY` override.

## FAQ

### Why was my change rejected?

The NL engine only allows keys defined in the config schema. If the instruction
mentions unknown keys or unsafe edits, the patch is returned with no operations
and a summary explaining the refusal. Review the diff output and adjust the
instruction to target supported settings.

### How do I confirm risky changes?

Risky edits (constraints, leverage, validation removals) trigger a confirmation
prompt. Re-run with `--no-confirm` to apply without interactive approval, or
review the diff and accept the prompt when asked.

### Can I use local models?

Yes. Set `TREND_LLM_PROVIDER=ollama`, point `TREND_LLM_BASE_URL` at your local
endpoint, and supply a local model name via `TREND_LLM_MODEL`.
