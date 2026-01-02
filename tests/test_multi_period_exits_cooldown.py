from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.multi_period.engine as engine


class THCfg:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "missing_policy": "ffill",
            "risk_free_column": "RF",
        }
        self.performance: dict[str, Any] = {"enable_cache": False}
        self.vol_adjust: dict[str, Any] = {"target_vol": 1.0}
        self.run: dict[str, Any] = {"monthly_cost": 0.0}
        self.benchmarks: dict[str, Any] = {}
        self.seed = 0

        self.portfolio: dict[str, Any] = {
            "policy": "threshold_hold",
            "rebalance_freq": "",
            "indices_list": [],
            "selector": {"params": {"rank_column": "Sharpe"}},
            "threshold_hold": {
                "metric": "Sharpe",
                "target_n": 3,
                "z_entry_soft": 1.0,
                "z_exit_soft": -1.0,
                "soft_strikes": 1,
                "entry_soft_strikes": 1,
                "z_exit_hard": -1.5,
            },
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.0,
                "max_weight": 1.0,
            },
            "weighting": {"name": "equal"},
        }

    def model_dump(self) -> dict[str, Any]:
        return {"portfolio": self.portfolio, "multi_period": {}}


def _df_5_funds() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    base = np.linspace(0.01, 0.06, num=len(dates))
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "A": base,
            "B": base,
            "C": base,
            "D": base,
            "E": base,
            "RF": 0.0,
        }
    )


def _patch_metric_series(
    monkeypatch: pytest.MonkeyPatch, *, by_in_end: dict[str, dict[str, float]]
) -> None:
    import trend_analysis.core.rank_selection as rank_selection

    def fake_metric_series(
        frame: pd.DataFrame, metric: str, stats_cfg: object
    ) -> pd.Series:
        del metric, stats_cfg
        if frame.empty:
            return pd.Series(dtype=float)
        end_key = pd.Timestamp(frame.index.max()).strftime("%Y-%m")
        mapping = by_in_end.get(end_key, {})
        return pd.Series(
            {str(c): float(mapping.get(str(c), 0.0)) for c in frame.columns},
            index=[str(c) for c in frame.columns],
            dtype=float,
        )

    monkeypatch.setattr(rank_selection, "_compute_metric_series", fake_metric_series)


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine, "apply_missing_policy", lambda frame, *, policy, limit: (frame, {})
    )
    monkeypatch.setattr(engine, "_run_analysis", lambda *args, **kwargs: {})


def _count_reentries(results: list[dict[str, Any]], manager: str) -> int:
    reentries = 0
    prev_in = False
    for idx, res in enumerate(results):
        selected = set(res.get("selected_funds") or [])
        current_in = manager in selected
        if idx > 0 and current_in and not prev_in:
            reentries += 1
        prev_in = current_in
    return reentries


def test_threshold_hold_exit_drop_not_blocked_by_turnover_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = THCfg()
    cfg.portfolio["turnover_budget_max_changes"] = 1

    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)

    # Period 1 seed: A,B,C. Period 2: B hard-exits, D would enter.
    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": 0.0, "B": -10.0, "C": 0.5, "D": 10.0, "E": 0.4},
        },
    )
    _patch_pipeline(monkeypatch)

    results = engine.run(cfg, df=_df_5_funds())

    assert len(results) == 2
    period2 = results[1]
    selected = set(period2["selected_funds"])

    # Budget is too tight to allow the drop+add pair, but the exit drop must still occur.
    assert "B" not in selected
    assert "D" not in selected  # entry was budget-skipped

    changes = period2.get("manager_changes") or []
    assert any(
        ev.get("reason") == "turnover_budget" and ev.get("action") == "skipped"
        for ev in changes
    )
    assert any(
        ev.get("action") == "dropped" and ev.get("manager") == "B" for ev in changes
    )


def test_threshold_hold_cooldown_blocks_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = THCfg()
    cfg.portfolio["turnover_budget_max_changes"] = 10
    cfg.portfolio["cooldown_periods"] = 1

    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
        SimpleNamespace(
            in_start="2020-03", in_end="2020-04", out_start="2020-05", out_end="2020-05"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)

    # Seed A,B,C; then B exits and D enters; then C exits and B would re-enter,
    # but cooldown should block B in that immediate next period.
    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": 0.0, "B": -10.0, "C": 0.5, "D": 10.0, "E": 0.4},
            "2020-04": {"A": 0.0, "B": 10.0, "C": -10.0, "D": 0.4, "E": 9.0},
        },
    )
    _patch_pipeline(monkeypatch)

    results = engine.run(cfg, df=_df_5_funds())
    assert len(results) == 3

    period3 = results[2]
    selected = set(period3["selected_funds"])
    assert "B" not in selected

    changes = period3.get("manager_changes") or []
    assert any(
        ev.get("reason") == "cooldown" and ev.get("manager") == "B" for ev in changes
    )


def test_cooldown_reduces_reentry_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
        SimpleNamespace(
            in_start="2020-03", in_end="2020-04", out_start="2020-05", out_end="2020-05"
        ),
        SimpleNamespace(
            in_start="2020-04", in_end="2020-05", out_start="2020-06", out_end="2020-06"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)

    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": 0.0, "B": -10.0, "C": 0.5, "D": 10.0, "E": 0.4},
            "2020-04": {"A": 0.0, "B": 10.0, "C": -10.0, "D": 0.4, "E": 9.0},
            "2020-05": {"A": 0.0, "B": 10.0, "C": 0.4, "D": -10.0, "E": 9.0},
        },
    )
    _patch_pipeline(monkeypatch)

    def _run(cooldown_periods: int) -> list[dict[str, Any]]:
        cfg = THCfg()
        cfg.portfolio["turnover_budget_max_changes"] = 10
        cfg.portfolio["cooldown_periods"] = cooldown_periods
        return engine.run(cfg, df=_df_5_funds())

    no_cooldown = _run(0)
    cooldown = _run(2)

    assert _count_reentries(cooldown, "B") < _count_reentries(no_cooldown, "B")
    assert "B" in set(no_cooldown[2]["selected_funds"])
    assert "B" not in set(cooldown[2]["selected_funds"])


def test_min_funds_can_exceed_turnover_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = THCfg()
    cfg.portfolio["turnover_budget_max_changes"] = 1
    cfg.portfolio["threshold_hold"].pop("z_exit_hard", None)
    cfg.portfolio.setdefault("constraints", {})["min_funds"] = 3

    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)

    # In period 2, force both B and C to hard-exit; budget will allow no adds,
    # so min_funds must top up holdings anyway.
    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": 0.0, "B": -10.0, "C": -9.0, "D": 10.0, "E": 9.0},
        },
    )
    _patch_pipeline(monkeypatch)

    results = engine.run(cfg, df=_df_5_funds())
    assert len(results) == 2

    period2 = results[1]
    selected = list(period2["selected_funds"])
    assert len(selected) >= 3

    changes = period2.get("manager_changes") or []
    assert any(
        ev.get("reason") == "min_funds" and ev.get("action") == "added"
        for ev in changes
    )


def test_threshold_hold_hard_entry_blocks_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = THCfg()
    cfg.portfolio["threshold_hold"]["target_n"] = 3
    cfg.portfolio["threshold_hold"]["z_entry_soft"] = -5.0
    cfg.portfolio["threshold_hold"]["z_entry_hard"] = 1.0

    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)

    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 0.0, "B": 1.0, "C": 2.0, "D": 0.0, "E": 0.0},
        },
    )
    _patch_pipeline(monkeypatch)

    results = engine.run(cfg, df=_df_5_funds())
    assert len(results) == 1

    selected = set(results[0]["selected_funds"])
    assert selected == {"C"}


def test_threshold_hold_hard_exit_protects_holdings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = THCfg()
    cfg.portfolio["threshold_hold"]["target_n"] = 2
    cfg.portfolio["threshold_hold"]["z_entry_soft"] = -5.0
    cfg.portfolio["threshold_hold"]["z_exit_soft"] = 0.6
    cfg.portfolio["threshold_hold"]["z_exit_hard"] = -0.5
    cfg.portfolio["threshold_hold"]["soft_strikes"] = 1

    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)

    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 0.0, "B": 1.0, "C": 2.0, "D": 0.0, "E": 0.0},
            "2020-03": {"A": 0.0, "B": 1.0, "C": 2.0, "D": 0.0, "E": 0.0},
        },
    )
    _patch_pipeline(monkeypatch)

    results = engine.run(cfg, df=_df_5_funds())
    assert len(results) == 2

    selected = set(results[1]["selected_funds"])
    assert "B" in selected


def test_threshold_hold_hard_exit_blocks_low_weight_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = THCfg()
    cfg.portfolio["threshold_hold"]["target_n"] = 2
    cfg.portfolio["threshold_hold"]["z_exit_hard"] = 1.0
    cfg.portfolio["constraints"]["min_weight"] = 0.6
    cfg.portfolio["constraints"]["min_weight_strikes"] = 1
    cfg.portfolio["constraints"]["max_funds"] = 2

    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)

    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 2.0, "B": 1.0, "C": 0.0, "D": 0.0, "E": 0.0},
            "2020-03": {"A": 2.0, "B": 1.0, "C": 0.0, "D": 0.0, "E": 0.0},
        },
    )
    _patch_pipeline(monkeypatch)

    results = engine.run(cfg, df=_df_5_funds())
    assert len(results) == 2

    period2 = results[1]
    selected = set(period2["selected_funds"])
    assert "A" in selected

    changes = period2.get("manager_changes") or []
    assert not any(
        ev.get("action") == "dropped" and ev.get("manager") == "A" for ev in changes
    )
