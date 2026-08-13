from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_quick_start_matches_active_profile() -> None:
    source = (REPO_ROOT / "streamlit_app" / "app.py").read_text(encoding="utf-8")

    assert (
        "custom_analysis_available = demo_profile.custom_analysis_enabled(_active_profile)"
        in source
    )
    assert "if custom_analysis_available:" in source
    assert '"- Use the **Custom Analysis** section' in source
    assert 'st.markdown("\\n".join(quick_start))' in source


def test_developer_surfaces_are_explicitly_marked_or_gated() -> None:
    data_source = (REPO_ROOT / "streamlit_app" / "pages" / "1_Data.py").read_text(encoding="utf-8")
    validation_source = (
        REPO_ROOT / "streamlit_app" / "developer_settings_validation.py"
    ).read_text(encoding="utf-8")

    # Indentation-insensitive: the exact-whitespace form of this assertion broke on any
    # re-indent without a behaviour change. The behavioural pair in
    # tests/app/test_data_page.py (test_debug_surfaces_hidden_without_flag /
    # test_debug_surfaces_visible_with_flag) is what actually pins the gate.
    assert 'if st.session_state.get("show_perf_diagnostics"):' in data_source
    assert 'with st.expander("Debug: Fund selection state"' in data_source
    assert data_source.index(
        'if st.session_state.get("show_perf_diagnostics"):'
    ) < data_source.index(
        'with st.expander("Debug: Fund selection state"'
    ), "the Debug expander must sit inside the show_perf_diagnostics gate"
    assert 'page_title="Developer: Settings Validation"' in validation_source
    assert 'st.title("🔧 Developer: Settings Validation")' in validation_source
    assert "not demo_profile.custom_analysis_enabled(active_profile)" in validation_source
