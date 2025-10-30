from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import trend_analysis.pipeline as pipeline_mod
from tests.test_trend_portfolio_app_helpers import _DummyStreamlit
from trend_analysis.config import DEFAULTS as DEFAULT_CFG_PATH


class _RunButtonStreamlit(_DummyStreamlit):
    """Streamlit stub that triggers the single-period run CTA."""

    def button(self, label: str, *args: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        # Only the "Run Single Period" button should fire during the test. All
        # other UI buttons remain inactive to avoid unintended side effects.
        if label == "Run Single Period":
            return True
        return False


def test_run_tab_applies_session_state_and_invokes_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure the run tab wires session-state edits back into the config."""

    captured: dict[str, Any] = {}

    def fake_run(cfg: Any) -> pd.DataFrame:
        captured["config"] = cfg
        return pd.DataFrame({"metric": [0.1, 0.2]})

    monkeypatch.setattr(pipeline_mod, "run", fake_run)

    streamlit_stub = _RunButtonStreamlit()

    defaults = yaml.safe_load(Path(DEFAULT_CFG_PATH).read_text(encoding="utf-8"))
    assert isinstance(defaults, dict)
    # Pre-load the session with defaults so the app skips reading from disk.
    streamlit_stub.session_state["config_dict"] = defaults
    # Populate a few dotted session keys that the run tab should fold back
    # into the nested configuration structure.
    csv_path = tmp_path / "custom.csv"
    csv_path.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    streamlit_stub.session_state["data.csv_path"] = str(csv_path)
    streamlit_stub.session_state["portfolio.constraints.max_weight"] = 0.25
    streamlit_stub.session_state["vol_adjust.window._months"] = 2

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    sys.modules.pop("trend_portfolio_app.app", None)

    app_mod = importlib.import_module("trend_portfolio_app.app")
    assert app_mod is not None

    # The single-period run should have invoked the stub pipeline with the
    # materialised Config object.
    assert "config" in captured

    config_dict = streamlit_stub.session_state["config_dict"]
    assert config_dict["data"]["csv_path"] == str(csv_path)
    assert config_dict["portfolio"]["constraints"]["max_weight"] == 0.25
    # The helper converts months into trading-day lengths (~21 trading days).
    assert config_dict["vol_adjust"]["window"]["length"] == 42

    # The pipeline stub should receive the Config object with the updates.
    cfg_obj = captured["config"]
    assert getattr(cfg_obj, "data")["csv_path"] == str(csv_path)
    assert getattr(cfg_obj, "vol_adjust")["window"]["length"] == 42

    # Avoid leaking the imported module to other tests.
    sys.modules.pop("trend_portfolio_app.app", None)
