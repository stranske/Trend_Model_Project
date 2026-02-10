# Environment Variables

## Anthropic API Key Resolution

When the provider is set to `anthropic`, the API key resolver checks variables in this order:

1. `CLAUDE_API_STRANSKE` (primary)
2. `ANTHROPIC_API_KEY` (fallback, used only if `CLAUDE_API_STRANSKE` is unset)

If neither is set, the Anthropic client will raise a missing API key error.
