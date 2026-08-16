from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from tests.test_multi_period_engine_threshold_events_extended import (
    ScriptedSelector,
    SequencedWeighting,
    StaticRebalancer,
    ThresholdConfig,
    _build_base_frame,
    _patch_metrics,
)
from tests.test_multi_period_exits_cooldown import (
    THCfg,
    _df_5_funds,
    _patch_metric_series,
    _patch_pipeline,
)
from trend_analysis.multi_period import engine as mp_engine


class _InjectedMetricFailure(RuntimeError):
    pass


def _zscore_conversion_lines() -> list[int]:
    source, start_line = inspect.getsourcelines(mp_engine._run_threshold_hold_multi_periods)
    lines = [
        start_line + offset
        for offset, line in enumerate(source)
        if "pd.to_numeric(sf.loc[" in line and '"zscore"' in line
    ]
    assert len(lines) == 5
    return lines


def _run_event_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ThresholdConfig()
    periods = [
        types.SimpleNamespace(
            in_start="2020-01-31",
            in_end="2020-03-31",
            out_start="2020-04-30",
            out_end="2020-04-30",
        ),
        types.SimpleNamespace(
            in_start="2020-02-29",
            in_end="2020-04-30",
            out_start="2020-05-31",
            out_end="2020-05-31",
        ),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda *_: periods)
    _patch_metrics(monkeypatch)
    monkeypatch.setattr(
        mp_engine,
        "AdaptiveBayesWeighting",
        lambda *args, **kwargs: SequencedWeighting(
            [
                {
                    "Alpha One": 0.7,
                    "Beta One": 0.1,
                    "Gamma One": 0.05,
                    "Delta One": 0.05,
                },
                {
                    "Alpha One": 0.8,
                    "Beta One": 0.04,
                    "Gamma One": 0.02,
                    "Delta One": 0.01,
                },
                {"Alpha One": 0.7, "Gamma One": 0.05, "Delta One": 0.2},
                {
                    "Alpha One": 0.7,
                    "Delta One": 0.2,
                    "Epsilon One": 0.1,
                    "Beta One": 0.05,
                },
            ]
        ),
    )
    monkeypatch.setattr(mp_engine, "Rebalancer", StaticRebalancer)

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_args, **_kwargs: ScriptedSelector(
            [
                "Alpha One",
                "Beta One",
                "Gamma One",
                "Delta One",
                "Alpha Two",
                "Epsilon One",
            ],
            cfg.portfolio["threshold_hold"]["target_n"],
        ),
    )
    monkeypatch.setattr(
        mp_engine,
        "_run_analysis",
        lambda *_args, **_kwargs: {"out_user_stats": {}, "out_ew_stats": {}},
    )

    mp_engine.run(cfg, df=_build_base_frame())


def _run_turnover_budget_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = THCfg()
    cfg.portfolio["turnover_budget_max_changes"] = 1
    periods = [
        types.SimpleNamespace(
            in_start="2020-01",
            in_end="2020-02",
            out_start="2020-03",
            out_end="2020-03",
        ),
        types.SimpleNamespace(
            in_start="2020-02",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-04",
        ),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    _patch_metric_series(
        monkeypatch,
        by_in_end={
            "2020-02": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0},
            "2020-03": {"A": 0.0, "B": -10.0, "C": 0.5, "D": 10.0, "E": 0.4},
        },
    )
    _patch_pipeline(monkeypatch)

    mp_engine.run(cfg, df=_df_5_funds())


@pytest.mark.parametrize(
    ("site_index", "run_scenario"),
    [
        pytest.param(0, _run_event_scenario, id="attempted-add-event"),
        pytest.param(1, _run_event_scenario, id="max-funds-ranking"),
        pytest.param(2, _run_turnover_budget_scenario, id="turnover-budget-ranking"),
        pytest.param(3, _run_event_scenario, id="drop-event"),
        pytest.param(4, _run_event_scenario, id="add-event"),
    ],
)
def test_metric_failure_is_visible_not_nan(
    monkeypatch: pytest.MonkeyPatch,
    site_index: int,
    run_scenario: Callable[[pytest.MonkeyPatch], None],
) -> None:
    """Every ranking-path conversion must surface a computation failure."""

    target_line = _zscore_conversion_lines()[site_index]
    engine_file = inspect.getsourcefile(mp_engine._run_threshold_hold_multi_periods)
    original_to_numeric = pd.to_numeric

    def fail_at_target(value: Any, *args: Any, **kwargs: Any) -> Any:
        caller = inspect.currentframe().f_back
        if (
            caller is not None
            and caller.f_code.co_filename == engine_file
            and caller.f_lineno == target_line
        ):
            raise _InjectedMetricFailure(f"injected failure at ranking site {site_index}")
        return original_to_numeric(value, *args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", fail_at_target)

    with pytest.raises(_InjectedMetricFailure, match=f"ranking site {site_index}"):
        run_scenario(monkeypatch)
