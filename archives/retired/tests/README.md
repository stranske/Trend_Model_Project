# Archived Test Files

**Archived:** December 7, 2025

These test files were archived because they test obsolete page structures that no longer exist.

## Archived Files

| File | Reason |
|------|--------|
| `test_streamlit_run_page.py` | Tests for obsolete `3_Run.py` page (replaced by `3_Results.py`) |
| `test_streamlit_run_page_new.py` | Tests for obsolete `3_Run.py` page (replaced by `3_Results.py`) |
| `test_streamlit_fallback_banner.py` | Tests for obsolete `3_Run.py` page (replaced by `3_Results.py`) |
| `test_disclaimer.py` | Tests for obsolete `3_Run.py` page (replaced by `3_Results.py`) |
| `test_streamlit_configure_guardrails.py` | Tests for obsolete `2_Configure.py` page (replaced by `2_Model.py`) |

## Page Restructure Summary

The Streamlit app underwent a restructure:

- `2_Configure.py` → `2_Model.py` (Model Configuration page)
- `3_Run.py` → `3_Results.py` (Results & Analysis page)

The new pages have different internal structures and functions, making the old tests incompatible.

## Current Test Coverage

Active tests for the current page structure are in:
- `tests/app/test_model_page_helpers.py` - Tests for `2_Model.py`
- `tests/app/test_results_page.py` - Tests for `3_Results.py`
- `tests/test_streamlit_smoke_ci.py` - Smoke tests for all pages
