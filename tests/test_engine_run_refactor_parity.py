from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from trend_analysis.multi_period import engine


class RefactorParityConfig:
    def __init__(self) -> None:
        self.data = {"allow_risk_free_fallback": True, "missing_policy": "ffill"}
        self.multi_period = {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-05",
        }
        self.portfolio = {
            "policy": "threshold_hold",
            "threshold_hold": {
                "metric": "Sharpe",
                "target_n": 2,
                "z_exit_soft": -10.0,
                "z_entry_soft": -10.0,
            },
            "constraints": {"max_funds": 2, "min_weight": 0.0, "max_weight": 1.0},
            "weighting": {"name": "equal"},
            "transaction_cost_bps": 0.0,
            "max_turnover": 1.0,
            "indices_list": None,
        }
        self.metrics = {"rf_override_enabled": True, "rf_rate_annual": 0.0}
        self.vol_adjust = {"target_vol": 1.0}
        self.run = {"monthly_cost": 0.0}
        self.benchmarks = {}
        self.seed = 7

    def model_dump(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "metrics": self.metrics,
            "vol_adjust": self.vol_adjust,
            "run": self.run,
            "benchmarks": self.benchmarks,
            "seed": self.seed,
        }


class SelectorStub:
    rank_column = "Sharpe"

    def select(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected = frame.sort_values("Sharpe", ascending=False).head(2)
        return selected, frame


class RebalancerStub:
    def apply_triggers(
        self,
        prev_weights: pd.Series,
        score_frame: pd.DataFrame,
        **_kwargs: Any,
    ) -> pd.Series:
        return prev_weights.reindex(score_frame.index).dropna()


def test_run_refactor_preserves_period_parity(monkeypatch) -> None:
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
    monkeypatch.setattr(
        engine, "apply_missing_policy", lambda frame, **_kwargs: (frame, {})
    )
    monkeypatch.setattr(
        engine, "Rebalancer", lambda *_args, **_kwargs: RebalancerStub()
    )

    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_args, **_kwargs: SelectorStub(),
    )

    pipeline_calls: list[
        tuple[str, tuple[str, ...], tuple[tuple[str, float], ...]]
    ] = []

    def fake_run_analysis(
        _df_arg: pd.DataFrame,
        _in_start: str,
        _in_end: str,
        out_start: str,
        _out_end: str,
        *_args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        manual_funds = tuple(kwargs["manual_funds"])
        custom_weights = tuple(sorted(kwargs["custom_weights"].items()))
        pipeline_calls.append((out_start, manual_funds, custom_weights))
        return {
            "fund_weights": dict(kwargs["custom_weights"]),
            "out_sample_scaled": pd.DataFrame(
                index=pd.date_range(f"{out_start}-28", periods=1)
            ),
            "score_frame": pd.DataFrame(index=manual_funds),
        }

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "FundA": [0.01, 0.02, 0.03, 0.04, 0.05],
            "FundB": [0.02, 0.01, 0.02, 0.01, 0.02],
            "FundC": [0.03, 0.03, 0.01, 0.01, 0.03],
        }
    )

    results = engine.run(RefactorParityConfig(), df=df)

    assert [result["period"] for result in results] == [
        ("2020-01", "2020-02", "2020-03", "2020-03"),
        ("2020-02", "2020-03", "2020-04", "2020-04"),
        ("2020-03", "2020-04", "2020-05", "2020-05"),
    ]
    assert len(pipeline_calls) == 3
    assert [call[0] for call in pipeline_calls] == ["2020-03", "2020-04", "2020-05"]
    assert all(len(call[1]) == 2 for call in pipeline_calls)
    assert [tuple(result["selected_funds"]) for result in results] == [
        call[1] for call in pipeline_calls
    ]
