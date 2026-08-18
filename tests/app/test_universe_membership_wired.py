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


def test_demo_membership_covers_demo_returns_managers() -> None:
    from utils.paths import proj_path

    demo_returns = proj_path() / "demo" / "demo_returns.csv"
    demo_membership = proj_path() / "demo" / "demo_universe_membership.csv"
    returns = pd.read_csv(demo_returns, nrows=1)
    manager_cols = [col for col in returns.columns if col.startswith("Mgr_")]
    membership = load_universe_membership(demo_membership)
    missing = sorted(set(manager_cols) - set(membership))
    assert not missing, f"demo membership missing managers: {missing}"


def test_membership_upload_state_only_changes_for_new_content(tmp_path: Path) -> None:
    from streamlit_app.components.universe_membership_input import (
        UNIVERSE_MEMBERSHIP_UPLOAD_DIGEST_KEY,
        UNIVERSE_MEMBERSHIP_UPLOAD_WIDGET_VERSION_KEY,
        clear_active_universe_membership,
        persist_membership_upload,
        set_active_universe_membership,
        summarise_membership,
    )

    upload_bytes = b"fund,effective_date,end_date\n" b"Mgr_01,2016-01-31,\n"
    first_path = persist_membership_upload(
        upload_bytes, upload_dir=tmp_path, filename="membership.csv"
    )
    second_path = persist_membership_upload(
        upload_bytes, upload_dir=tmp_path, filename="membership.csv"
    )
    assert first_path != second_path

    state: dict[str, object] = {UNIVERSE_MEMBERSHIP_UPLOAD_DIGEST_KEY: "digest"}
    assert set_active_universe_membership(first_path, summarise_membership(first_path), state)
    assert not set_active_universe_membership(first_path, summarise_membership(first_path), state)
    assert clear_active_universe_membership(state)
    assert UNIVERSE_MEMBERSHIP_UPLOAD_DIGEST_KEY not in state
    assert state[UNIVERSE_MEMBERSHIP_UPLOAD_WIDGET_VERSION_KEY] == 1


def test_invalid_membership_upload_is_removed(tmp_path: Path) -> None:
    from streamlit_app.components.universe_membership_input import (
        persist_membership_upload,
    )

    with pytest.raises(ValueError):
        persist_membership_upload(b"fund\nMgr_01\n", upload_dir=tmp_path, filename="membership.csv")
    assert not list(tmp_path.iterdir())


def test_execute_analysis_masks_single_period_membership(tmp_path: Path, monkeypatch) -> None:
    from streamlit_app.components import analysis_runner

    membership_path = tmp_path / "membership.csv"
    _write_membership(membership_path)
    session_state = {"universe_membership_path": str(membership_path)}
    monkeypatch.setattr(analysis_runner, "st", _streamlit_stub(session_state))
    monkeypatch.setattr(
        analysis_runner,
        "resolve_universe_membership_path",
        lambda: session_state.get("universe_membership_path"),
    )
    monkeypatch.setattr(analysis_runner, "_build_config", lambda payload: SimpleNamespace())
    monkeypatch.setattr(analysis_runner, "_validate_streamlit_payload", lambda payload: None)
    monkeypatch.setattr(analysis_runner, "_assert_config_feasible", lambda config: None)

    captured: list[pd.DataFrame] = []
    monkeypatch.setattr(
        "trend_analysis.api.run_simulation",
        lambda config, returns: captured.append(returns.copy()) or "result",
    )
    returns = _fixture_returns()
    returns.index.name = "Date"
    payload = analysis_runner.AnalysisPayload(
        returns=returns, model_state={"multi_period_enabled": False}, benchmark=None
    )

    assert analysis_runner._execute_analysis(payload) == "result"
    masked = captured.pop()
    assert pd.isna(masked.loc[masked["Date"] == pd.Timestamp("2016-04-30"), "Mgr_03"]).all()

    session_state.clear()
    assert analysis_runner._execute_analysis(payload) == "result"
    unmasked = captured.pop()
    assert unmasked.loc[unmasked["Date"] == pd.Timestamp("2016-04-30"), "Mgr_03"].iloc[
        0
    ] == pytest.approx(0.05)


def test_multi_period_api_passes_loaded_membership_to_engine(monkeypatch) -> None:
    from trend_analysis import api, multi_period
    from trend_analysis.multi_period import loaders

    membership = pd.DataFrame(
        {
            "fund": ["Mgr_01"],
            "effective_date": [pd.Timestamp("2016-01-31")],
            "end_date": [pd.NaT],
        }
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(loaders, "load_membership", lambda config: membership)
    monkeypatch.setattr(
        multi_period,
        "run",
        lambda config, returns, *, membership: captured.update(membership=membership) or [],
    )

    result = api._run_multi_period_simulation(
        SimpleNamespace(run_id="membership-test"),
        _fixture_returns().reset_index(names="Date"),
        {},
        7,
    )

    assert result.period_count == 0
    assert captured["membership"].equals(membership)
