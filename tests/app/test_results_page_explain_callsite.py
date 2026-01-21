from pathlib import Path


def test_results_page_calls_explain_results() -> None:
    contents = Path("streamlit_app/pages/3_Results.py").read_text(encoding="utf-8")
    assert "run_key = _current_run_key(" in contents
    assert "explain_results.render_explain_results(result, run_key=run_key)" in contents
