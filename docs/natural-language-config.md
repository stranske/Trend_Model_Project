# Natural Language Configuration

## Overview

The `trend nl` command lets you describe configuration changes in plain language.
It generates a structured ConfigPatch, applies it to a YAML config, and can
optionally validate and run the analysis. Use it to iterate quickly on settings
without manually editing YAML.

## Prerequisites

- Python 3.10+ (the NL tooling relies on libraries that require 3.10 or newer).
- An API key for your chosen provider (for example `OPENAI_API_KEY` or
  `ANTHROPIC_API_KEY`).

## Installation

```bash
pip install trend-model[llm]
```

## Quick Start

```bash
trend nl "use risk parity weighting" --in config/demo.yml --diff
```

## Configuration

### Provider setup

Natural language edits use LangChain-backed providers. The CLI reads provider
settings from environment variables:

- `TREND_LLM_PROVIDER` (default: `openai`) - one of `openai`, `anthropic`, `ollama`.
- `TREND_LLM_API_KEY` - optional override for API keys.
- Provider-specific keys:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
- Optional overrides:
  - `TREND_LLM_MODEL` (default: `gpt-4o-mini`)
  - `TREND_LLM_TEMPERATURE` (default: `0.0`)
  - `TREND_LLM_BASE_URL` (useful for local or proxy endpoints)
  - `TREND_LLM_ORG` (OpenAI organization)

Example setup for OpenAI:

```bash
export TREND_LLM_PROVIDER=openai
export OPENAI_API_KEY="..."
export TREND_LLM_MODEL="gpt-4o-mini"
```

Example setup for Anthropic:

```bash
export TREND_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="..."
```

Example setup for a local Ollama model:

```bash
export TREND_LLM_PROVIDER=ollama
export TREND_LLM_BASE_URL="http://localhost:11434"
export TREND_LLM_MODEL="llama3.1"
```

### CLI overrides

Use command-line flags to override the model or temperature without changing
environment variables:

```bash
trend nl "lower max weight" --model gpt-4o-mini --temperature 0.2
```

## Usage Examples

Common operations include previewing diffs, writing to a new config, or applying
changes directly before running a backtest.

Preview changes without writing files:

```bash
trend nl "reduce max weight to 8%" --in config/demo.yml --diff
```

Write updates to a new file:

```bash
trend nl "enable risk parity weighting" --in config/demo.yml --out config/updated.yml
```

Apply changes and run the pipeline:

```bash
trend nl "set top_n to 12" --in config/demo.yml --run
```

Apply a risky change without interactive confirmation:

```bash
trend nl "remove max weight constraints" --in config/demo.yml --run --no-confirm
```

Add a data path and keep a dry-run copy in stdout:

```bash
trend nl "set data.csv_path to demo/demo_returns.csv" --in config/demo.yml --dry-run
```

## Known Limitations

- The LLM can only edit keys that exist in the allowed schema; unknown keys are
  rejected or flagged for review.
- Some changes are considered risky (constraints or validation changes) and
  require confirmation unless `--no-confirm` is used.
- Outputs are best-effort; you should review diffs for critical changes.
- Large configs or complex instructions can produce slower responses.
- Offline/local use requires a supported provider (for example `ollama`).
