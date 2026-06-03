"""A30 — surface the multi-period risk-weighting silent fallback.

When ``create_weight_engine`` raises during multi-period (threshold-hold) engine
setup, the engine silently falls back to equal weight. Previously it did so
without recording anything, so the Results-page banner
(``streamlit_app/pages/3_Results.py``) never fired and a user could believe they
received risk-parity weights when they actually got equal weights.

These tests assert the fallback is now surfaced: each period result carries a
``weight_engine_fallback`` marker, and ``api.run_simulation`` propagates it to
``RunResult.fallback_info`` (which the banner reads).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class _Config:
    """Minimal threshold-hold config that requests a risk-weighting engine."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-04",
        }
    )
    data: Dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "unused.csv",
            "allow_risk_free_fallback": True,
        }
    )
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "threshold_hold",
            # Requesting a risk-weighting scheme is what drives the engine to
            # call create_weight_engine (and therefore the fallback path).
            "weighting_scheme": "risk_parity",
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "threshold_hold": {
                "target_n": 3,
                "metric": "Sharpe",
                "soft_strikes": 1,
                "entry_soft_strikes": 1,
                "z_exit_soft": -5.0,
                "z_entry_soft": -5.0,
            },
            "constraints": {
                "max_funds": 3,
                "min_weight": 0.2,
                "max_weight": 0.55,
                "min_weight_strikes": 1,
            },
            "weighting": {"name": "equal", "params": {}},
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


@dataclass
class _Period:
    in_start: str
    in_end: str
    out_start: str
    out_end: str


class _Selector:
    rank_column = "Sharpe"

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_frame, score_frame


class _Rebalancer:
    def __init__(self, *_cfg: Any) -> None:
        pass

    def apply_triggers(self, prev_weights: pd.Series, _sf: pd.DataFrame, **_kwargs: Any) -> pd.Series:
        return prev_weights.astype(float)


_METRIC_MAPS = {
    "AnnualReturn": {"Alpha One": 0.15, "Alpha Two": 0.12, "Beta One": 0.07, "Gamma One": 0.18},
    "Volatility": {"Alpha One": 0.25, "Alpha Two": 0.2, "Beta One": 0.15, "Gamma One": 0.3},
    "Sharpe": {"Alpha One": 0.9, "Alpha Two": 0.8, "Beta One": 0.4, "Gamma One": 1.1},
    "Sortino": {"Alpha One": 1.1, "Alpha Two": 0.9, "Beta One": 0.45, "Gamma One": 1.3},
    "InformationRatio": {"Alpha One": 0.6, "Alpha Two": 0.5, "Beta One": 0.3, "Gamma One": 0.9},
    "MaxDrawdown": {"Alpha One": -0.12, "Alpha Two": -0.11, "Beta One": -0.05, "Gamma One": -0.09},
}


def _make_df() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]
    )
    return pd.DataFrame(
        {
            "Date": dates,
            "Alpha One": [0.05, 0.04, 0.03, 0.02, 0.01],
            "Alpha Two": [0.06, 0.05, 0.04, 0.03, 0.02],
            "Beta One": [0.02, 0.03, 0.02, 0.01, 0.02],
            "Gamma One": [0.04, 0.05, 0.06, 0.07, 0.08],
        }
    )


def _wire_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the deterministic engine collaborators used by the threshold path."""

    periods = [
        _Period("2020-01-31", "2020-03-31", "2020-04-30", "2020-04-30"),
        _Period("2020-02-29", "2020-04-30", "2020-05-31", "2020-05-31"),
    ]
    monkeypatch.setattr(mp_engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(mp_engine, "Rebalancer", _Rebalancer)

    import trend_analysis.selector as selector_mod

    monkeypatch.setattr(
        selector_mod, "create_selector_by_name", lambda *a, **k: _Selector()
    )

    import trend_analysis.core.rank_selection as rank_sel

    def fake_metric_series(_frame: pd.DataFrame, metric: str, _stats_cfg: Any) -> pd.Series:
        return pd.Series(_METRIC_MAPS[metric], dtype=float)

    monkeypatch.setattr(rank_sel, "_compute_metric_series", fake_metric_series)

    def fake_run_analysis(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {"metrics": pd.DataFrame(), "details": {}, "seed": 123}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)


def test_multi_period_engine_surfaces_weight_engine_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When create_weight_engine raises, each period result records the fallback."""

    cfg = _Config()
    df = _make_df()
    _wire_engine(monkeypatch)

    import trend_analysis.plugins as plugins

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("synthetic engine failure")

    monkeypatch.setattr(plugins, "create_weight_engine", boom)

    results = mp_engine.run(cfg, df=df)

    assert results, "expected at least one period result"
    fallbacks = [
        r.get("weight_engine_fallback")
        for r in results
        if isinstance(r, dict) and r.get("weight_engine_fallback") is not None
    ]
    assert fallbacks, "silent fallback to equal weight was not surfaced"
    fb = fallbacks[0]
    assert fb["engine"] == "risk_parity"
    assert "synthetic engine failure" in fb["reason"]


def test_run_simulation_propagates_fallback_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api.run_simulation exposes the engine fallback on RunResult.fallback_info,
    which is what the Results-page banner reads."""

    from trend_analysis import api

    cfg = _Config()
    df = _make_df()
    _wire_engine(monkeypatch)

    import trend_analysis.plugins as plugins

    monkeypatch.setattr(
        plugins,
        "create_weight_engine",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic engine failure")),
    )

    result = api.run_simulation(cfg, df)

    assert isinstance(result.fallback_info, dict)
    assert result.fallback_info["engine"] == "risk_parity"
