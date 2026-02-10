## LLM Components Review

Components in `streamlit_app/components/` that import LLM modules:

- `streamlit_app/components/llm_settings.py`
- `streamlit_app/components/explain_results.py`
- `streamlit_app/components/comparison_llm.py`
- `streamlit_app/components/nl_operation_viewer.py`

## Components Requiring Refactor

None at this time. The components that perform LLM API calls already rely on the shared
resolver in `streamlit_app/components/llm_settings.py` for API key resolution.

## Environment Variable Audit

Audit command: `grep -E "os\\.environ|getenv" streamlit_app/components/*.py`

Findings:
- Only `streamlit_app/components/llm_settings.py` reads Anthropic API key variables.
- Other components read non-key environment variables (for example, provider name or
  upload limits) and do not require refactoring for Anthropic key resolution.
