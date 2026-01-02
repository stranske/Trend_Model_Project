from __future__ import annotations

from pathlib import Path

from scripts import evaluate_settings_effectiveness as effectiveness


def test_extract_settings_from_model_page_collects_keys(tmp_path: Path) -> None:
    sample = """
PRESET_CONFIGS = {"Baseline": {"alpha": 1}}

def _initial_model_state():
    return {"beta": 2}

def build():
    model_state = {}
    model_state["gamma"] = 3
    st.session_state["model_state"]["delta"] = 4
    _ = model_state.get("epsilon", 5)
    candidate_state = {"zeta": 6}
"""
    sample_path = tmp_path / "model_page.py"
    sample_path.write_text(sample, encoding="utf-8")

    baseline, keys = effectiveness.extract_settings_from_model_page(sample_path)

    assert baseline == {"alpha": 1}
    for key in {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}:
        assert key in keys


def test_extract_settings_from_model_page_includes_ui_settings() -> None:
    model_page = effectiveness.MODEL_PAGE

    baseline, keys = effectiveness.extract_settings_from_model_page(model_page)

    assert baseline, "Baseline preset should be parsed from the model page."
    expected_keys = {
        "lookback_periods",
        "rank_pct",
        "buy_hold_initial",
        "shrinkage_method",
        "report_rolling_metrics",
        "transaction_cost_bps",
    }
    assert expected_keys <= keys
