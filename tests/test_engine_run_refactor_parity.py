from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

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
    monkeypatch.setattr(engine, "apply_missing_policy", lambda frame, **_kwargs: (frame, {}))
    monkeypatch.setattr(engine, "Rebalancer", lambda *_args, **_kwargs: RebalancerStub())

    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_args, **_kwargs: SelectorStub(),
    )
    original_setup_period = engine._setup_period
    original_weight_period = engine._weight_period
    original_apply_turnover_and_cost = engine._apply_turnover_and_cost
    original_assemble_period_result = engine._assemble_period_result
    setup_period_calls: list[str] = []
    weight_period_calls: list[tuple[str, ...]] = []
    turnover_cost_calls: list[tuple[str, ...]] = []
    assemble_period_calls: list[tuple[str, ...]] = []

    def wrapped_setup_period(pt: Any, **kwargs: Any) -> engine._PeriodSetup:
        setup_period_calls.append(pt.out_end)
        return original_setup_period(pt, **kwargs)

    monkeypatch.setattr(engine, "_setup_period", wrapped_setup_period)

    def wrapped_weight_period(**kwargs: Any) -> engine._PeriodWeights:
        weight_period_calls.append(tuple(kwargs["holdings"]))
        return original_weight_period(**kwargs)

    monkeypatch.setattr(engine, "_weight_period", wrapped_weight_period)

    def wrapped_apply_turnover_and_cost(
        **kwargs: Any,
    ) -> engine._TurnoverCostApplication:
        turnover_cost_calls.append(tuple(kwargs["manual_holdings"]))
        return original_apply_turnover_and_cost(**kwargs)

    monkeypatch.setattr(engine, "_apply_turnover_and_cost", wrapped_apply_turnover_and_cost)

    def wrapped_assemble_period_result(
        **kwargs: Any,
    ) -> engine._PeriodResultAssembly:
        assemble_period_calls.append(tuple(kwargs["realised_holdings"]))
        return original_assemble_period_result(**kwargs)

    monkeypatch.setattr(engine, "_assemble_period_result", wrapped_assemble_period_result)

    pipeline_calls: list[tuple[str, tuple[str, ...], tuple[tuple[str, float], ...]]] = []

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
            "out_sample_scaled": pd.DataFrame(index=pd.date_range(f"{out_start}-28", periods=1)),
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
    assert setup_period_calls == ["2020-03", "2020-04", "2020-05"]
    assert len(weight_period_calls) == 3
    assert len(turnover_cost_calls) == 3
    assert len(assemble_period_calls) == 3
    assert all(len(call[1]) == 2 for call in pipeline_calls)
    assert all(len(call) == 2 for call in weight_period_calls)
    assert all(len(call) == 2 for call in turnover_cost_calls)
    assert all(len(call) == 2 for call in assemble_period_calls)
    assert [tuple(result["selected_funds"]) for result in results] == [
        call[1] for call in pipeline_calls
    ]


def test_run_signature_matches_pre_refactor_contract() -> None:
    signature = inspect.signature(engine.run)

    assert list(signature.parameters) == ["cfg", "df", "price_frames", "membership"]
    assert signature.parameters["cfg"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["df"].default is None
    assert signature.parameters["price_frames"].default is None
    assert signature.parameters["membership"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["membership"].default is None


def test_threshold_path_extracted_helpers_are_called_in_required_order() -> None:
    helpers = (
        "_prepare_threshold_hold_period",
        "_weight_period",
        "_apply_turnover_and_cost",
        "_assemble_period_result",
    )
    source = textwrap.dedent(inspect.getsource(engine._run_threshold_hold_multi_periods))
    tree = ast.parse(source)
    call_lines: dict[str, int] = {}

    class HelperCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in helpers:
                call_lines.setdefault(node.func.id, node.lineno)
            self.generic_visit(node)

    HelperCallVisitor().visit(tree)

    assert list(call_lines) == [
        "_prepare_threshold_hold_period",
        "_weight_period",
        "_apply_turnover_and_cost",
        "_assemble_period_result",
    ]
    assert call_lines["_prepare_threshold_hold_period"] < call_lines["_weight_period"]
    assert call_lines["_weight_period"] < call_lines["_apply_turnover_and_cost"]
    assert call_lines["_apply_turnover_and_cost"] < call_lines["_assemble_period_result"]


def test_run_refactor_output_schema_and_deliberate_break(monkeypatch) -> None:
    periods = [
        SimpleNamespace(
            in_start="2020-01", in_end="2020-02", out_start="2020-03", out_end="2020-03"
        ),
        SimpleNamespace(
            in_start="2020-02", in_end="2020-03", out_start="2020-04", out_end="2020-04"
        ),
    ]
    monkeypatch.setattr(engine, "generate_periods", lambda _cfg: periods)
    monkeypatch.setattr(engine, "apply_missing_policy", lambda frame, **_kwargs: (frame, {}))
    monkeypatch.setattr(engine, "Rebalancer", lambda *_args, **_kwargs: RebalancerStub())

    from trend_analysis import selector as selector_mod

    monkeypatch.setattr(
        selector_mod,
        "create_selector_by_name",
        lambda *_args, **_kwargs: SelectorStub(),
    )

    def fake_run_analysis(
        _df_arg: pd.DataFrame,
        _in_start: str,
        _in_end: str,
        out_start: str,
        _out_end: str,
        *_args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "fund_weights": dict(kwargs["custom_weights"]),
            "out_sample_scaled": pd.DataFrame(index=pd.date_range(f"{out_start}-28", periods=1)),
            "score_frame": pd.DataFrame(index=tuple(kwargs["manual_funds"])),
        }

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="ME"),
            "FundA": [0.01, 0.02, 0.03, 0.04],
            "FundB": [0.02, 0.01, 0.02, 0.01],
            "FundC": [0.03, 0.03, 0.01, 0.01],
        }
    )
    results = engine.run(RefactorParityConfig(), df=df)

    assert [result["period"] for result in results] == [
        ("2020-01", "2020-02", "2020-03", "2020-03"),
        ("2020-02", "2020-03", "2020-04", "2020-04"),
    ]
    assert set(results[0]) >= {
        "period",
        "selected_funds",
        "score_frame",
        "turnover",
        "transaction_cost",
        "fund_weights",
        "manager_changes",
        "holding_tenure",
    }
    assert all(isinstance(result["period"], tuple) for result in results)
    assert all(len(result["period"]) == 4 for result in results)
    assert all(len(result["selected_funds"]) == 2 for result in results)
    assert all("fund_weights" in result for result in results)

    original_assemble_period_result = engine._assemble_period_result

    def drop_final_period(**kwargs: Any) -> engine._PeriodResultAssembly:
        assembled = original_assemble_period_result(**kwargs)
        if kwargs["period"].out_end == periods[-1].out_end:
            return engine._PeriodResultAssembly(
                result={},
                holdings_tenure=assembled.holdings_tenure,
                prev_final_weights=assembled.prev_final_weights,
                prev_weights=assembled.prev_weights,
            )
        return assembled

    monkeypatch.setattr(engine, "_assemble_period_result", drop_final_period)
    broken_results = engine.run(RefactorParityConfig(), df=df)

    with pytest.raises(AssertionError):
        assert all("period" in result for result in broken_results)
        assert [result["period"] for result in broken_results] == [
            ("2020-01", "2020-02", "2020-03", "2020-03"),
            ("2020-02", "2020-03", "2020-04", "2020-04"),
        ]
