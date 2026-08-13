from __future__ import annotations

from pathlib import Path

from streamlit_app.components import demo_runner
from trend_analysis.presets import get_trend_preset
from trend_analysis.signal_presets import get_trend_spec_preset

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


def test_cli_and_streamlit_share_preset_resolution() -> None:
    """The signal view and UI payload must derive from one canonical preset."""

    full_preset = get_trend_preset("balanced")
    signal_preset = get_trend_spec_preset("balanced")
    ui_payload = demo_runner._load_preset("balanced")

    assert signal_preset.spec == full_preset.trend_spec
    assert ui_payload == full_preset.config_mapping()
    assert ui_payload["signals"]["window"] == signal_preset.spec.window
