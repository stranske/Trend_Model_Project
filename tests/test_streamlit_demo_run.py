from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List


class _ImportSpinner:
    def __call__(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ImportStub(SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.session_state = {}
        self.spinner = _ImportSpinner()

    def success(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


sys.modules.setdefault("streamlit", _ImportStub())

import pandas as pd
import pytest

import streamlit_app.demo_runner as demo_runner


class DummySpinner:
    """Minimal spinner context manager."""

    def __call__(self, _text: str):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyStreamlit:
    """Lightweight stand-in for the Streamlit module."""

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.spinner = DummySpinner()
        self.success_messages: List[str] = []
        self.error_messages: List[str] = []
        self._switched_pages: List[str] = []

    # UI helpers ---------------------------------------------------------
    def success(self, msg: str) -> None:
        self.success_messages.append(msg)

    def error(self, msg: str) -> None:
        self.error_messages.append(msg)

    def switch_page(self, page: str) -> None:
        self._switched_pages.append(page)


@dataclass
class DummySimResult:
    """Simple object exposing the interface expected by the results view."""

    portfolio: pd.Series
    weights: dict[pd.Timestamp, pd.Series]

    def __post_init__(self) -> None:
        self.fallback_info = None
        self.signal_pnls = None
        self.rebalancing_pnl = None

    def portfolio_curve(self) -> pd.Series:
        return (1 + self.portfolio.fillna(0.0)).cumprod()

    def drawdown_curve(self) -> pd.Series:
        curve = self.portfolio_curve()
        return curve / curve.cummax() - 1.0

    def event_log_df(self) -> pd.DataFrame:
        return pd.DataFrame({"event": ["demo"], "detail": ["ok"]})

    def summary(self) -> dict[str, Any]:
        return {"total_return": float(self.portfolio_curve().iloc[-1] - 1.0)}


def test_load_demo_dataset_smoke():
    """The bundled demo dataset should load with basic structure."""
    df, meta = demo_runner.load_demo_dataset()
    assert not df.empty
    assert "Mgr_01" in df.columns
    assert meta["n_rows"] == len(df)


def test_build_demo_configuration_maps_metrics():
    """Preset metrics should map onto the Streamlit metric registry."""
    df, _ = demo_runner.load_demo_dataset()
    preset = {
        "lookback_months": 12,
        "rebalance_frequency": "monthly",
        "min_track_months": 6,
        "selection_count": 5,
        "risk_target": 0.1,
        "metrics": {"sharpe_ratio": 0.6, "max_drawdown": 0.4},
        "portfolio": {"cooldown_months": 2, "max_weight": 0.2},
    }
    cfg, policy, mapping, overrides, benchmark, _cands = demo_runner.build_demo_configuration(
        df, preset, "Test"
    )
    assert cfg["lookback_months"] == 12
    metric_names = {m.name for m in policy.metrics}
    assert metric_names == {"sharpe", "drawdown"}
    assert overrides["selected_metrics"] == ["sharpe", "drawdown"]
    assert mapping["benchmark_column"] == benchmark


def test_run_one_click_demo_sets_state(monkeypatch):
    """Running the demo should populate session state and queue navigation."""
    df, _ = demo_runner.load_demo_dataset()
    dates = df.index[:3]
    dummy_result = DummySimResult(
        portfolio=pd.Series([0.01, -0.005, 0.007], index=dates),
        weights={dates[0]: pd.Series([0.5, 0.5], index=df.columns[:2])},
    )

    monkeypatch.setattr(
        demo_runner,
        "_execute_demo_simulation",
        lambda *_args, **_kwargs: dummy_result,
    )

    st_mod = DummyStreamlit()
    demo_runner.run_one_click_demo(st_module=st_mod, preset_name="Balanced")

    assert "returns_df" in st_mod.session_state
    assert "sim_config" in st_mod.session_state
    assert st_mod.session_state[demo_runner.NAV_QUEUE_KEY] == [
        "pages/4_Results.py",
        "pages/5_Export.py",
    ]
    assert st_mod._switched_pages == ["pages/4_Results.py"]
    assert st_mod.success_messages
    assert st_mod.session_state["sim_results"] is dummy_result


def test_handle_demo_navigation_advances_queue():
    """Navigation helper should advance through queued pages."""
    st_mod = DummyStreamlit()
    st_mod.session_state[demo_runner.NAV_QUEUE_KEY] = [
        "pages/4_Results.py",
        "pages/5_Export.py",
    ]
    # First call primes the redirect without switching
    assert (
        demo_runner.handle_demo_navigation(st_mod, "pages/4_Results.py")
        is None
    )
    assert st_mod._switched_pages == []
    assert st_mod.session_state[demo_runner.NAV_READY_FLAG] is True
    # Second call performs the switch
    assert (
        demo_runner.handle_demo_navigation(st_mod, "pages/4_Results.py")
        == "pages/5_Export.py"
    )
    assert st_mod._switched_pages == ["pages/5_Export.py"]
    assert st_mod.session_state[demo_runner.NAV_QUEUE_KEY] == []
    assert (
        demo_runner.handle_demo_navigation(st_mod, "pages/4_Results.py")
        is None
    )


def test_handle_demo_navigation_without_switch():
    """Gracefully handle missing switch_page attribute."""
    st_mod = DummyStreamlit()
    st_mod.switch_page = None
    st_mod.session_state[demo_runner.NAV_QUEUE_KEY] = ["pages/4_Results.py"]
    # Should not raise even though switch_page is unavailable
    assert (
        demo_runner.handle_demo_navigation(st_mod, "pages/4_Results.py")
        is None
    )
    assert st_mod.session_state[demo_runner.NAV_QUEUE_KEY] == []
