## LLM Components Review

Components in `streamlit_app/components/` that import LLM modules:

- `streamlit_app/components/llm_settings.py`
- `streamlit_app/components/explain_results.py`
- `streamlit_app/components/comparison_llm.py`
- `streamlit_app/components/nl_operation_viewer.py`

## Components Requiring Refactor

None at this time. The components that perform LLM API calls already rely on the shared
resolver in `streamlit_app/components/llm_settings.py` for API key resolution.
