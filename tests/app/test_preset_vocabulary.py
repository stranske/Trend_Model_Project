from __future__ import annotations

from pathlib import Path

from streamlit_app.components import demo_runner

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_home_demo_preset_label_is_distinct_from_model_preset_label() -> None:
    app_source = (REPO_ROOT / "streamlit_app" / "app.py").read_text(encoding="utf-8")
    demo_runner_source = (REPO_ROOT / "streamlit_app" / "components" / "demo_runner.py").read_text(
        encoding="utf-8"
    )
    model_source = (REPO_ROOT / "streamlit_app" / "pages" / "2_Model.py").read_text(
        encoding="utf-8"
    )

    assert demo_runner.DEMO_PRESET_SELECTOR_LABEL == "Demo Settings Preset"
    assert "demo settings preset" in demo_runner.DEMO_PRESET_SELECTOR_HELP.lower()
    assert "model configuration presets" in demo_runner.DEMO_PRESET_SELECTOR_HELP.lower()
    assert "DEMO_PRESET_SELECTOR_LABEL" in app_source
    assert "Strategy Preset" not in app_source
    assert "Strategy Preset" not in demo_runner_source
    assert "Preset Configuration" in model_source
    assert demo_runner.DEMO_PRESET_SELECTOR_LABEL != "Preset Configuration"
