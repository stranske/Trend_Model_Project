from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trend_analysis.pipeline as pipeline
import trend_analysis.pipeline_helpers as pipeline_helpers
from trend.diagnostics import DiagnosticPayload, DiagnosticResult
from trend_analysis.stages import portfolio as portfolio_stage
from trend_analysis.pipeline_entrypoints import (
    _resolve_single_period_monthly_cost,
    _resolve_single_period_weighting_scheme,
    run_from_config,
    run_full_from_config,
)


def _sample_stats(value: float = 0.1) -> portfolio_stage._Stats:
    """Helper to build Stats objects with predictable values."""

    return portfolio_stage._Stats(
        cagr=value,
        vol=value + 0.1,
        sharpe=value + 0.2,
        sortino=value + 0.3,
        max_drawdown=-(value + 0.4),
        information_ratio=value + 0.5,
        is_avg_corr=value + 0.6,
        os_avg_corr=value + 0.7,
    )


def _bindings_with_analysis(invoke_analysis_with_diag: object):
    """Inject analysis behavior through the public ConfigBindings seam."""

    return replace(
        pipeline._bindings(),
        invoke_analysis_with_diag=invoke_analysis_with_diag,
    )


def test_run_requires_csv_path() -> None:
    with pytest.raises(KeyError):
        pipeline.run({})


def test_same_config_same_numbers_across_entrypoints() -> None:
    """Single-period config resolution matches the multi-period cost contract."""

    portfolio = {
        "cost_model": {"per_trade_bps": 12, "half_spread_bps": 3},
        "weighting": {"name": "score_prop_bayes"},
    }

    assert _resolve_single_period_weighting_scheme(portfolio, dict.get) == "score_prop_bayes"
    assert _resolve_single_period_monthly_cost(portfolio, {}) == pytest.approx(0.0015)


@pytest.mark.parametrize(
    ("invalid", "error_fragment"),
    [(float("nan"), "finite"), (float("inf"), "finite"), (-1, "negative")],
)
def test_single_period_costs_reject_non_finite_and_negative_values(
    invalid: float, error_fragment: str
) -> None:
    with pytest.raises(ValueError, match=error_fragment):
        _resolve_single_period_monthly_cost(
            {"cost_model": {"per_trade_bps": invalid, "half_spread_bps": 0}}, {}
        )
    with pytest.raises(ValueError, match=error_fragment):
        _resolve_single_period_monthly_cost({}, {"monthly_cost": invalid})


def test_single_period_cost_model_honours_canonical_fields() -> None:
    portfolio = {
        "cost_model": {
            "per_trade_bps": 12,
            "half_spread_bps": 3,
        }
    }

    assert _resolve_single_period_monthly_cost(portfolio, {}) == pytest.approx(0.0015)


def test_zero_cost_model_preserves_configured_flat_monthly_cost() -> None:
    portfolio = {
        "cost_model": {
            "per_trade_bps": 0,
            "half_spread_bps": 0,
        }
    }

    assert _resolve_single_period_monthly_cost(
        portfolio,
        {"monthly_cost": 0.0025},
    ) == pytest.approx(0.0025)


def test_single_period_uses_nested_weighting_name() -> None:
    portfolio = {"weighting": {"name": "risk_parity"}}

    assert _resolve_single_period_weighting_scheme(portfolio, dict.get) == "risk_parity"


@pytest.fixture(name="sample_frame")
def _sample_frame_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="ME"),
            "FundA": [0.01, 0.02, 0.0, 0.03],
            "FundB": [0.0, -0.01, 0.02, 0.01],
        }
    )


@pytest.fixture(name="sample_split")
def _sample_split_fixture() -> dict[str, str]:
    return {
        "in_start": "2020-01",
        "in_end": "2020-02",
        "out_start": "2020-03",
        "out_end": "2020-04",
    }


@pytest.fixture(name="base_config")
def _base_config_fixture() -> dict[str, object]:
    return {
        "data": {
            "csv_path": "dummy.csv",
            "risk_free_column": None,
            "allow_risk_free_fallback": True,
        },
        "sample_split": {},
        "preprocessing": {},
        "run": {},
        "portfolio": {},
        "vol_adjust": {},
    }


def test_run_converts_stats_payload_to_frame(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())

    stats_payload = {
        "FundA": _sample_stats(0.1),
        "FundB": _sample_stats(0.2),
    }
    benchmark_ir = {
        "custom": {"FundA": 1.23, "FundB": 0.98, "equal_weight": 0.0},
        "user_weight": {"FundA": 0.5},
    }

    def fake_run_analysis(*args, **kwargs):
        assert args[0] is sample_frame
        assert kwargs["signal_spec"] is not None
        return {
            "out_sample_stats": stats_payload,
            "benchmark_ir": benchmark_ir,
            "risk_diagnostics": {},
            "fund_weights": {},
            "in_sample_stats": {},
        }

    result = run_from_config(base_config, bindings=_bindings_with_analysis(fake_run_analysis))
    assert list(result.index) == ["FundA", "FundB"]
    assert set(result.columns) >= {
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "information_ratio",
        "ir_custom",
    }
    # Synthetic weight aggregates still surface as dedicated columns.
    assert result.loc["FundA", "ir_user_weight"] == pytest.approx(0.5)
    assert result.loc["FundA", "ir_custom"] == pytest.approx(1.23)


def test_run_returns_empty_frame_when_analysis_none(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())
    result = run_from_config(base_config, bindings=_bindings_with_analysis(lambda *_, **__: None))
    assert result.empty


def test_run_preserves_diagnostic_result_payload(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())
    diagnostic = DiagnosticPayload("partial-result", "analysis completed with a warning")
    payload = {
        "out_sample_stats": {"FundA": _sample_stats(0.1)},
        "benchmark_ir": {},
        "risk_diagnostics": {},
        "fund_weights": {},
        "in_sample_stats": {},
    }

    result = run_from_config(
        base_config,
        bindings=_bindings_with_analysis(
            lambda *_, **__: DiagnosticResult(value=payload, diagnostic=diagnostic)
        ),
    )

    assert list(result.index) == ["FundA"]
    assert result.attrs["diagnostic"] is diagnostic


def test_run_preserves_diagnostic_result_without_payload(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())
    diagnostic = DiagnosticPayload("no-result", "analysis returned no payload")

    result = run_from_config(
        base_config,
        bindings=_bindings_with_analysis(
            lambda *_, **__: DiagnosticResult(value=None, diagnostic=diagnostic)
        ),
    )

    assert result.empty
    assert result.attrs["diagnostic"] is diagnostic


@pytest.mark.parametrize(
    "data_overrides, expected_allow",
    [
        ({}, False),
        ({"allow_risk_free_fallback": None}, False),
        ({"allow_risk_free_fallback": True, "risk_free_column": "RF"}, False),
        ({"allow_risk_free_fallback": True}, True),
    ],
)
def test_run_resolves_risk_free_defaults(
    data_overrides: dict[str, object],
    expected_allow: bool,
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
) -> None:
    base_config = {
        "data": {"csv_path": "dummy.csv"},
        "sample_split": {},
        "preprocessing": {},
        "run": {},
        "portfolio": {},
        "vol_adjust": {},
    }
    base_config["data"].update(data_overrides)

    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())

    captured_kwargs: dict[str, object] = {}

    def fake_run_analysis(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "out_sample_stats": {},
            "benchmark_ir": {},
            "risk_diagnostics": {},
            "fund_weights": {},
            "in_sample_stats": {},
        }

    run_from_config(base_config, bindings=_bindings_with_analysis(fake_run_analysis))

    assert captured_kwargs["allow_risk_free_fallback"] is expected_allow


@pytest.mark.parametrize(
    ("data_overrides", "expected"),
    [
        ({"risk_free_column": "RF", "allow_risk_free_fallback": True}, ("RF", False)),
        ({"allow_risk_free_fallback": True}, (None, True)),
        ({"allow_risk_free_fallback": False}, (None, False)),
        ({}, (None, False)),
    ],
    ids=["explicit-column", "fallback-enabled", "fallback-disabled", "missing-column"],
)
def test_both_entrypoints_share_risk_free_resolution(
    data_overrides: dict[str, object],
    expected: tuple[str | None, bool],
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
) -> None:
    config: dict[str, object] = {
        "data": {"csv_path": "dummy.csv", **data_overrides},
        "sample_split": {},
        "preprocessing": {},
        "run": {},
        "portfolio": {},
        "vol_adjust": {},
    }
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())

    def fake_run_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        captured.append((args, kwargs))
        return pipeline_helpers._empty_run_full_result()

    bindings = _bindings_with_analysis(fake_run_analysis)
    run_from_config(config, bindings=bindings)
    run_full_from_config(config, bindings=bindings)

    first_args, first_kwargs = captured[0]
    second_args, second_kwargs = captured[1]
    assert first_args[0] is sample_frame
    assert second_args[0] is sample_frame
    assert first_args[1:] == second_args[1:]
    first_without_signal = {k: v for k, v in first_kwargs.items() if k != "signal_spec"}
    second_without_signal = {k: v for k, v in second_kwargs.items() if k != "signal_spec"}
    assert first_without_signal == second_without_signal
    assert first_kwargs["signal_spec"] is not None
    assert second_kwargs["signal_spec"] is not None
    observed = [
        (kwargs["risk_free_column"], kwargs["allow_risk_free_fallback"]) for _, kwargs in captured
    ]
    assert observed == [expected, expected]


def test_both_entrypoints_preserve_object_backed_risk_free_settings(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
) -> None:
    config = SimpleNamespace(
        data=SimpleNamespace(
            csv_path="dummy.csv", risk_free_column="RF", allow_risk_free_fallback=True
        ),
        sample_split={},
        preprocessing={},
        run={},
        portfolio={},
        vol_adjust={},
    )
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())
    bindings = _bindings_with_analysis(
        lambda *_, **kwargs: captured.append(kwargs) or pipeline_helpers._empty_run_full_result()
    )
    run_from_config(config, bindings=bindings)
    run_full_from_config(config, bindings=bindings)

    assert [
        (kwargs["risk_free_column"], kwargs["allow_risk_free_fallback"]) for kwargs in captured
    ] == [("RF", False), ("RF", False)]


def test_run_full_propagates_analysis_payload(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())

    base_config["portfolio"] = {
        "weighting": {"name": "score_prop_bayes"},
        "cost_model": {"per_trade_bps": 12, "half_spread_bps": 3},
    }
    payload = {
        "out_sample_stats": {"FundA": _sample_stats(0.4)},
        "benchmark_ir": {},
        "risk_diagnostics": {"entries": 1},
    }
    captured_kwargs: dict[str, object] = {}
    captured_args: tuple[object, ...] = ()

    def fake_run_analysis(*_args, **kwargs):
        nonlocal captured_args
        captured_args = _args
        captured_kwargs.update(kwargs)
        return payload

    result = run_full_from_config(
        base_config,
        bindings=_bindings_with_analysis(fake_run_analysis),
    )
    assert result.unwrap() is payload
    assert captured_kwargs["weighting_scheme"] == "score_prop_bayes"
    assert captured_args[6] == pytest.approx(0.0015)


def test_run_full_returns_empty_when_analysis_none(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline_helpers, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(pipeline_helpers, "_build_trend_spec", lambda *_args, **_kwargs: object())
    result = run_full_from_config(
        base_config,
        bindings=_bindings_with_analysis(lambda *_, **__: None),
    )
    assert result.unwrap() is None


def test_empty_run_full_result_template() -> None:
    payload = pipeline_helpers._empty_run_full_result()
    assert set(payload.keys()) == {
        "out_sample_stats",
        "in_sample_stats",
        "benchmark_ir",
        "risk_diagnostics",
        "fund_weights",
    }


def test_compute_stats_includes_optional_avg_corr() -> None:
    data = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, 0.0, -0.01],
            "FundB": [0.0, 0.01, 0.02, 0.03],
        }
    )
    rf = pd.Series([0.0, 0.001, 0.002, 0.0])
    stats = portfolio_stage._compute_stats(
        data,
        rf,
        periods_per_year=12,
        in_sample_avg_corr={"FundA": 0.5},
        out_sample_avg_corr={"FundB": 0.25},
    )
    assert stats["FundA"].is_avg_corr == 0.5
    assert stats["FundB"].os_avg_corr == 0.25


def test_portfolio_stats_use_window_periods_per_year() -> None:
    returns = pd.DataFrame({"FundA": [0.015, -0.004, 0.005, -0.008] * 3})
    risk_free = pd.Series(0.0, index=returns.index)

    monthly = portfolio_stage._compute_stats(returns, risk_free, periods_per_year=12)["FundA"]
    weekly = portfolio_stage._compute_stats(returns, risk_free, periods_per_year=52)["FundA"]

    assert weekly.cagr > monthly.cagr
    assert weekly.vol > monthly.vol
    assert weekly.sharpe > monthly.sharpe


def test_calc_portfolio_returns_scales_weights(sample_frame: pd.DataFrame) -> None:
    weights = np.array([0.6, 0.4])
    portfolio = portfolio_stage.calc_portfolio_returns(weights, sample_frame[["FundA", "FundB"]])
    assert isinstance(portfolio, pd.Series)
    expected = (sample_frame[["FundA", "FundB"]] * weights).sum(axis=1)
    pd.testing.assert_series_equal(portfolio, expected)
