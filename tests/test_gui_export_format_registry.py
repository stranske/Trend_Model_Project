"""Regression tests for the ipywidgets export-format selector."""

import pytest

from trend_analysis import export
from trend_analysis.gui import app
from trend_analysis.gui.store import ParamStore


@pytest.mark.parametrize("format_name", sorted(export.EXPORTERS))
def test_launch_accepts_every_registered_export_format(monkeypatch, format_name):
    """The real Dropdown accepts every format exposed by the exporter registry."""

    store = ParamStore(cfg={"export": {"formats": [format_name]}})
    monkeypatch.setattr(app, "load_state", lambda: store)
    monkeypatch.setattr(app, "discover_plugins", lambda: None)

    root = app.launch()

    assert root.children[6].value == format_name


def test_launch_survives_an_unselectable_persisted_format(monkeypatch, tmp_path):
    """A persisted retired format degrades in memory instead of bricking the GUI."""

    state_file = tmp_path / "trend_gui_state.yml"
    state_file.write_text("export:\n  formats:\n    - parquet\n")
    monkeypatch.setattr(app, "STATE_FILE", state_file)
    monkeypatch.setattr(app, "WEIGHT_STATE_FILE", tmp_path / "trend_gui_weights.pkl")
    monkeypatch.setattr(app, "discover_plugins", lambda: None)

    with pytest.warns(UserWarning, match="parquet"):
        root = app.launch()

    assert root.children[6].value == "xlsx"


def test_launch_preserves_a_plugin_exporter_from_persisted_state(monkeypatch, tmp_path):
    """Plugin discovery must precede validation of persisted export formats."""

    state_file = tmp_path / "trend_gui_state.yml"
    state_file.write_text("export:\n  formats:\n    - plugin-format\n")
    monkeypatch.setattr(app, "STATE_FILE", state_file)
    monkeypatch.setattr(app, "WEIGHT_STATE_FILE", tmp_path / "trend_gui_weights.pkl")

    def discover_plugin_exporter() -> None:
        monkeypatch.setitem(app.export.EXPORTERS, "plugin-format", lambda *_args: None)

    monkeypatch.setattr(app, "discover_plugins", discover_plugin_exporter)

    root = app.launch()

    assert root.children[6].value == "plugin-format"


def test_normalized_export_formats_keeps_valid_siblings():
    assert app._normalized_export_formats(["csv", ""]) == ["csv"]
