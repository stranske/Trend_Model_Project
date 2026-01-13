# Natural Language CLI Reference

This page documents `trend nl`, which edits configuration files from natural
language instructions.

## Synopsis

```bash
trend nl "<instruction>" [--in <file>] [--out <file>] [--diff] [--dry-run] [--run] \
  [--no-confirm] [--provider <name>] [--explain]
```

## Flags

- `--diff` - Preview changes without applying them to disk.
- `--in <file>` - Specify the input config file (default: `config/defaults.yml`).
- `--out <file>` - Specify a different output file to write updates.
- `--run` - Apply changes and run the analysis.
- `--no-confirm` - Skip confirmation for risky changes.
- `--provider <name>` - Specify the LLM provider (`openai`, `anthropic`, `ollama`).
- `--dry-run` - Print the updated config to stdout without writing the file.
- `--explain` - Print the LLM change summary alongside optional diff output.

## Examples

Preview a diff from the demo config:

```bash
trend nl "use risk parity weighting" --in config/demo.yml --diff
```

Write updates to a new file:

```bash
trend nl "set max weight to 8%" --in config/demo.yml --out config/updated.yml
```

Run the pipeline after applying a change:

```bash
trend nl "set top_n to 12" --in config/demo.yml --run
```

Skip confirmation for a risky change:

```bash
trend nl "remove max weight constraints" --in config/demo.yml --run --no-confirm
```

Use a specific provider:

```bash
trend nl "add a 3% turnover cap" --in config/demo.yml --provider anthropic --diff
```

Print the updated YAML without writing it:

```bash
trend nl "set selection mode to rank" --in config/demo.yml --dry-run
```

Show the change summary with a diff preview:

```bash
trend nl "lower volatility target to 8%" --in config/demo.yml --explain --diff
```
