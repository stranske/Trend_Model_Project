from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.universe import apply_membership_windows, load_universe_membership


def _streamlit_stub(session_state: dict) -> SimpleNamespace:
    stub = SimpleNamespace()
    stub.session_state = session_state
    stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    stub.cache_resource = stub.cache_data
    return stub


def _fixture_returns() -> pd.DataFrame:
    index = pd.to_datetime(["2016-01-31", "2016-02-29", "2016-03-31", "2016-04-30", "2016-05-31"])
    return pd.DataFrame(
        {
            "Mgr_01": [0.01, 0.02, 0.01, 0.02, 0.01],
            "Mgr_02": [0.02, 0.01, 0.02, 0.01, 0.02],
            "Mgr_03": [0.03, 0.02, 0.04, 0.05, 0.06],
        },
        index=index,
    )


def _write_membership(path: Path) -> None:
    path.write_text(
        "fund,effective_date,end_date\n"
        "Mgr_01,2016-01-31,\n"
        "Mgr_02,2016-01-31,\n"
        "Mgr_03,2016-01-31,2016-03-31\n",
        encoding="utf-8",
    )


def test_exited_manager_excluded_after_exit_date(tmp_path: Path, monkeypatch) -> None:
    returns = _fixture_returns()
    membership_path = tmp_path / "membership.csv"
    _write_membership(membership_path)
    membership = load_universe_membership(membership_path)

    masked = apply_membership_windows(returns, membership)
    assert pd.isna(masked.loc[pd.Timestamp("2016-04-30"), "Mgr_03"])
    assert pd.isna(masked.loc[pd.Timestamp("2016-05-31"), "Mgr_03"])
    assert masked.loc[pd.Timestamp("2016-03-31"), "Mgr_03"] == pytest.approx(0.04)

    unmasked = apply_membership_windows(returns, {})
    assert unmasked.loc[pd.Timestamp("2016-05-31"), "Mgr_03"] == pytest.approx(0.06)

    session_state = {"universe_membership_path": str(membership_path)}
    monkeypatch.setitem(sys.modules, "streamlit", _streamlit_stub(session_state))

    from streamlit_app.components.analysis_runner import AnalysisPayload, _build_config

    model_state = {
        "multi_period_enabled": True,
        "multi_period_frequency": "M",
        "lookback_periods": 1,
        "evaluation_periods": 1,
        "date_mode": "explicit",
        "start_date": "2016-02-29",
        "end_date": "2016-05-31",
        "selection_count": 2,
        "metric_weights": {"sharpe": 1.0},
    }
    payload = AnalysisPayload(returns=returns, model_state=model_state, benchmark=None)
    with_membership = _build_config(payload)
    assert with_membership.data.get("universe_membership_path") == str(membership_path)

    session_state.pop("universe_membership_path")
    without_membership = _build_config(payload)
    assert "universe_membership_path" not in without_membership.data

    monkeypatch.setattr(
        "streamlit_app.components.analysis_runner.resolve_universe_membership_path",
        lambda: None,
    )
    broken = _build_config(payload)
    assert "universe_membership_path" not in broken.data

    monkeypatch.setattr(
        "streamlit_app.components.analysis_runner.resolve_universe_membership_path",
        lambda: str(membership_path),
    )
    restored = _build_config(payload)
    assert restored.data.get("universe_membership_path") == str(membership_path)
