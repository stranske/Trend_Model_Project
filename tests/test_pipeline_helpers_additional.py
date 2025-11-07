"""Additional unit tests for ``trend_analysis.pipeline`` helper utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

import trend_analysis.pipeline as pipeline
from trend_analysis.pipeline import (
    _build_trend_spec,
    _cfg_section,
    _cfg_value,
    _derive_split_from_periods,
    _empty_run_full_result,
    _policy_from_config,
    _prepare_input_data,
    _preprocessing_summary,
    _resolve_sample_split,
    _run_analysis,
    _Stats,
    _section_get,
    _unwrap_cfg,
    compute_signal,
    single_period_run,
)
from trend_analysis.signals import TrendSpec
from trend_analysis.util.frequency import FrequencySummary
from trend_analysis.util.missing import MissingPolicyResult


class DummyMapping(Mapping[str, Any]):
    """Mapping-like object exposing a ``get`` method with optional default."""

    def __init__(self, data: Mapping[str, Any]):
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):  # pragma: no cover - mapping protocol
        return iter(self._data)

    def __len__(self):  # pragma: no cover - mapping protocol
        return len(self._data)

    def get(self, key: str, default: Any = None):
        if default is ...:
            raise TypeError("Unexpected ellipsis default")
        return self._data.get(key, default)


@pytest.fixture(name="monthly_frame")
def fixture_monthly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="M"),
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.0, 0.01, -0.02, 0.03],
        }
    )


def test_cfg_helpers_handle_mixed_inputs() -> None:
    cfg = SimpleNamespace(alpha=1, beta={"nested": 2})
    mapping = {"gamma": 3}

    assert _cfg_value(cfg, "alpha") == 1
    assert _cfg_value(mapping, "gamma") == 3
    assert _cfg_value(mapping, "missing", 5) == 5

    section = _cfg_section(cfg, "beta")
    assert isinstance(section, dict)
    assert section["nested"] == 2

    wrapped = {"__cfg__": {"delta": 4}}
    unwrapped = _unwrap_cfg(wrapped)
    assert unwrapped == {"delta": 4}

    assert _cfg_section({}, "missing") == {}
    assert _section_get(None, "anything", default=9) == 9

    wrapped_none = {"__cfg__": None}
    assert _unwrap_cfg(wrapped_none) is wrapped_none

    empty = _empty_run_full_result()
    assert set(empty) == {
        "out_sample_stats",
        "in_sample_stats",
        "benchmark_ir",
        "risk_diagnostics",
        "fund_weights",
    }

    nested_mapping = DummyMapping({"value": 10})
    assert _section_get(nested_mapping, "value") == 10
    assert _section_get(nested_mapping, "other", default=7) == 7


def test_preprocessing_summary_includes_missing_info() -> None:
    summary = _preprocessing_summary(
        "D", normalised=True, missing_summary="ffill applied"
    )
    assert "Cadence: Daily" in summary
    assert "monthly" in summary.lower()
    assert "ffill applied" in summary

    summary_month = _preprocessing_summary("M", normalised=False, missing_summary=None)
    assert "(month-end)" in summary_month


def test_preprocessing_summary_monthly_normalised() -> None:
    summary = _preprocessing_summary("M", normalised=True, missing_summary="drop 2")
    assert "month-end" in summary
    assert "drop 2" in summary


def test_preprocessing_summary_monthly_branch_without_normalisation() -> None:
    summary = _preprocessing_summary("M", normalised=False, missing_summary="n/a")
    assert summary.startswith("Cadence: Monthly (month-end)")
    assert "Missing data: n/a" in summary


def test_preprocessing_summary_monthly_without_missing_details() -> None:
    summary = _preprocessing_summary("M", normalised=False, missing_summary=None)
    assert summary == "Cadence: Monthly (month-end)"


def test_build_trend_spec_uses_vol_adjust_defaults() -> None:
    cfg = {
        "signals": {
            "window": "12",
            "min_periods": "4",
            "lag": "0",
            "vol_target": "0.15",
            "zscore": True,
        }
    }
    vol_cfg = {"enabled": True, "target_vol": 0.2}
    spec = _build_trend_spec(cfg, vol_cfg)
    assert spec.window == 12
    assert spec.min_periods == 4
    # Lag coerces to minimum of 1 when invalid.
    assert spec.lag == 1
    # ``vol_adjust`` should fall back to the vol config default when unset.
    assert spec.vol_adjust is True
    assert spec.vol_target == pytest.approx(0.15)
    assert spec.zscore is True


def test_policy_from_config_constructs_composites() -> None:
    policy, limit = _policy_from_config(
        {
            "policy": "drop",
            "per_asset": {"A": "ffill", "B": "zero"},
            "limit": 2,
            "per_asset_limit": {"A": 1},
        }
    )
    assert isinstance(policy, dict)
    assert policy["default"] == "drop"
    assert policy["B"] == "zero"
    assert isinstance(limit, dict)
    assert limit["default"] == 2
    assert limit["A"] == 1


def test_policy_from_config_handles_missing_components() -> None:
    policy, limit = _policy_from_config(
        {"per_asset": {"X": "drop"}, "per_asset_limit": {"X": 3}}
    )
    assert policy == {"X": "drop"}
    assert limit == {"X": 3}


def test_derive_split_ratio_fallback_when_date_split_invalid() -> None:
    periods = pd.period_range("2020-01", periods=6, freq="M")
    result = _derive_split_from_periods(
        periods,
        method="date",
        boundary=pd.Period("2019-12", freq="M"),
        ratio=0.4,
    )
    # Boundary precedes all periods; ratio fallback should apply.
    assert result["in_end"] == "2020-02"
    assert result["out_start"] == "2020-03"

    ratio_edge = _derive_split_from_periods(
        periods[:2], method="ratio", boundary=None, ratio=1.5
    )
    # Even with an extreme ratio the helper leaves at least one out-of-sample period.
    assert ratio_edge["out_start"] == "2020-02"


def test_derive_split_ratio_raises_when_no_out_periods() -> None:
    base = pd.period_range("2020-01", periods=3, freq="M")

    class DegeneratePeriods:
        def __init__(self, data: pd.PeriodIndex):
            self._data = data

        def __len__(self) -> int:  # pragma: no cover - simple forwarding
            return len(self._data)

        def __getitem__(self, key):
            if isinstance(key, slice):
                start = 0 if key.start is None else key.start
                if start >= len(self._data) - 1 and key.step in (None, 1):
                    return pd.PeriodIndex([], freq=self._data.freq)
                return self._data[key]
            return self._data[key]

    periods = DegeneratePeriods(base)
    with pytest.raises(ValueError, match="out-of-sample window"):
        _derive_split_from_periods(periods, method="ratio", boundary=None, ratio=1.5)


def test_resolve_sample_split_derives_missing_fields() -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=6, freq="ME"), "A": np.arange(6)}
    )
    split_cfg = {"method": "date", "date": "2020-03"}
    resolved = _resolve_sample_split(df, split_cfg)
    assert resolved["in_end"] == "2020-03"
    assert resolved["out_start"] == "2020-04"

    ratio_cfg = {"method": "ratio", "ratio": "0.5"}
    resolved_ratio = _resolve_sample_split(df, ratio_cfg)
    assert resolved_ratio["in_start"] == "2020-01"
    assert resolved_ratio["out_start"] == "2020-04"

    with pytest.raises(ValueError, match="'Date'"):
        _resolve_sample_split(pd.DataFrame({"value": [1, 2]}), {})


def test_prepare_input_data_resamples_and_applies_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.0, 0.01, 0.0, 0.02],
        }
    )

    summary = FrequencySummary(
        code="D", label="Daily", resampled=True, target="M", target_label="Monthly"
    )
    fake_result = MissingPolicyResult(
        policy={"A": "drop", "B": "drop"},
        default_policy="drop",
        limit={"A": None, "B": None},
        default_limit=None,
        filled={"A": 2, "B": 1},
        dropped_assets=(),
        summary="ffill 2 rows",
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "detect_frequency", lambda series: summary)

        def fake_apply_missing_policy(data: pd.DataFrame, **kwargs):
            assert kwargs["policy"] == "drop"
            assert kwargs["enforce_completeness"] is True
            return data, fake_result

        mp.setattr(pipeline, "apply_missing_policy", fake_apply_missing_policy)

        monthly, freq_info, missing_meta, normalised = _prepare_input_data(
            df,
            date_col="Date",
            missing_policy="drop",
            missing_limit=None,
            enforce_completeness=True,
        )

    assert normalised is True
    assert isinstance(freq_info, FrequencySummary)
    assert missing_meta is fake_result
    assert list(monthly.columns) == ["Date", "A", "B"]
    # Resampled monthly DataFrame should collapse to the end of the month.
    assert all(monthly["Date"].dt.is_month_end)


class GetterTypeError:
    def get(self, key):
        raise KeyError(key)


class GetterKeyError:
    def get(self, key, default=None):
        raise KeyError(key)


def test_section_get_handles_partial_getters() -> None:
    weird = GetterTypeError()
    assert _section_get(weird, "missing", default=5) == 5

    always = GetterKeyError()
    assert _section_get(always, "missing", default=7) == 7

    attr_only = SimpleNamespace(answer=11)
    assert _section_get(attr_only, "answer", default=0) == 11


def test_build_trend_spec_invalid_values_fallbacks() -> None:
    cfg = {
        "signals": {
            "window": "bad",
            "min_periods": "bad",
            "lag": "bad",
            "vol_adjust": True,
            "vol_target": -1,
            "zscore": "",
        }
    }
    vol_cfg = {"enabled": True, "target_vol": 0.3}
    spec = _build_trend_spec(cfg, vol_cfg)
    assert spec.window == 63
    assert spec.min_periods is None
    assert spec.lag == 1
    assert spec.vol_target is None
    assert spec.zscore is False

    cfg_bad_target = {
        "signals": {
            "window": 5,
            "vol_adjust": True,
            "vol_target": "not-a-number",
        }
    }
    spec_bad = _build_trend_spec(cfg_bad_target, vol_cfg)
    assert spec_bad.vol_target is None

    cfg_missing_target = {"signals": {"vol_adjust": True}}
    spec_missing = _build_trend_spec(cfg_missing_target, {"enabled": True, "target_vol": 0.25})
    assert spec_missing.vol_target == pytest.approx(0.25)


def test_resolve_sample_split_invalid_dates_raise() -> None:
    df = pd.DataFrame({"Date": ["not-a-date", "still bad"], "A": [1, 2]})
    with pytest.raises(ValueError, match="no valid dates"):
        _resolve_sample_split(df, {})


def test_resolve_sample_split_handles_invalid_boundary() -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=6, freq="ME"), "A": np.arange(6)}
    )
    resolved = _resolve_sample_split(df, {"method": "date", "date": "invalid"})
    assert set(resolved) == {"in_start", "in_end", "out_start", "out_end"}


def test_resolve_sample_split_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=4, freq="ME"), "A": np.arange(4)}
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            pipeline,
            "_derive_split_from_periods",
            lambda *args, **kwargs: {"in_start": "2020-01"},
        )
        with pytest.raises(ValueError, match="Unable to derive sample split values"):
            _resolve_sample_split(df, {})


def test_policy_from_config_handles_base_only() -> None:
    policy, limit = _policy_from_config({"policy": "ffill", "limit": 3})
    assert policy == "ffill"
    assert limit == 3

    policy_none, limit_map = _policy_from_config({"per_asset_limit": {"A": 2}})
    assert policy_none is None
    assert isinstance(limit_map, dict)
    assert limit_map["A"] == 2

    policy_map, limit_spec = _policy_from_config(
        {
            "policy": "drop",
            "per_asset": {"A": "ffill"},
            "per_asset_limit": {"A": 1},
            "limit": 2,
        }
    )
    assert policy_map["default"] == "drop"
    assert limit_spec["default"] == 2


def test_derive_split_from_periods_edge_cases() -> None:
    periods = pd.period_range("2020-01", periods=5, freq="M")

    fallback = _derive_split_from_periods(
        periods, method="ratio", boundary=None, ratio="bad"
    )
    assert fallback["in_end"] == "2020-04"
    assert fallback["out_start"] == "2020-05"

    zero_ratio = _derive_split_from_periods(
        periods, method="ratio", boundary=None, ratio=0
    )
    assert zero_ratio["in_end"] == "2020-02"

    tiny_ratio = _derive_split_from_periods(
        periods, method="ratio", boundary=None, ratio=0.01
    )
    assert tiny_ratio["in_start"] == "2020-01"

    with pytest.raises(ValueError, match="without any observations"):
        _derive_split_from_periods(
            pd.PeriodIndex([], freq="M"),
            method="ratio",
            boundary=None,
            ratio=0.5,
        )


def test_derive_split_single_period_returns_all() -> None:
    single = pd.period_range("2020-01", periods=1, freq="M")
    split = _derive_split_from_periods(single, method="ratio", boundary=None, ratio=0.5)
    assert split == {
        "in_start": "2020-01",
        "in_end": "2020-01",
        "out_start": "2020-01",
        "out_end": "2020-01",
    }


def test_prepare_input_data_requires_date_column(monthly_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="'Date' column"):
        _prepare_input_data(
            monthly_frame.rename(columns={"Date": "When"}),
            date_col="Date",
            missing_policy=None,
            missing_limit=None,
        )


def test_prepare_input_data_handles_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=2, freq="M")})
    summary = FrequencySummary(
        code="M", label="Monthly", resampled=False, target="M", target_label="Monthly"
    )
    result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "detect_frequency", lambda series: summary)
        mp.setattr(
            pipeline,
            "apply_missing_policy",
            lambda *args, **kwargs: (pd.DataFrame(), result),
        )
        monthly, freq_info, missing_meta, normalised = _prepare_input_data(
            df,
            date_col="Date",
            missing_policy=None,
            missing_limit=None,
            enforce_completeness=True,
        )

    assert monthly.empty
    assert list(monthly.columns) == ["Date"]
    assert isinstance(freq_info, FrequencySummary)
    assert missing_meta is result
    assert normalised is False


def test_run_analysis_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    def prepare_empty(*args, **kwargs):
        return (
            pd.DataFrame(),
            FrequencySummary("M", "Monthly", False, "M", "Monthly"),
            empty_result,
            False,
        )

    monkeypatch.setattr(pipeline, "_prepare_input_data", prepare_empty)
    assert (
        _run_analysis(
            pd.DataFrame({"Date": []}),
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-04",
            0.1,
            0.0,
        )
        is None
    )

    def prepare_no_values(*args, **kwargs):
        frame = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"])})
        return (
            frame,
            FrequencySummary("M", "Monthly", False, "M", "Monthly"),
            empty_result,
            False,
        )

    monkeypatch.setattr(pipeline, "_prepare_input_data", prepare_no_values)
    assert (
        _run_analysis(
            pd.DataFrame({"Date": ["2020-01-01"]}),
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-04",
            0.1,
            0.0,
        )
        is None
    )


def test_run_analysis_rank_selection_with_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="M")
    prepared = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, -0.01, 0.03, 0.04],
            "FundB": [0.01, 0.02, -0.01, 0.0],
            "RF": [0.0, 0.0, 0.0, 0.0],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    def fake_prepare(*args, **kwargs):
        return (prepared.copy(), freq_summary, missing_result, False)

    def fake_single_period_run(*args, **kwargs):
        frame = pd.DataFrame({"Sharpe": [1.0], "Sortino": [0.5]}, index=["FundA"])
        frame.attrs["insample_len"] = 2
        frame.attrs["period"] = ("2020-01", "2020-02")
        return frame

    def fake_realised_volatility(data, window, periods_per_year=None):
        return pd.DataFrame({col: [0.2, 0.25] for col in data.columns})

    def fake_rank_select_funds(*args, **kwargs):
        return ["FundA", "FundB"]

    def fake_trend_signals(df, spec):
        return df.astype(float)

    def boom_engine(scheme):
        raise RuntimeError("engine fail")

    def boom_weights(*args, **kwargs):
        raise RuntimeError("risk fail")

    def fake_information_ratio(*args, **kwargs):
        return 0.0

    def fake_regime_payload(**kwargs):
        return {
            "table": pd.DataFrame(),
            "labels": pd.Series(dtype="string"),
            "out_labels": pd.Series(dtype="string"),
        }

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", fake_prepare)
        mp.setattr(pipeline, "single_period_run", fake_single_period_run)
        mp.setattr(pipeline, "make_window_key", lambda *args, **kwargs: "key")
        mp.setattr(pipeline, "get_window_metric_bundle", lambda key: {})
        mp.setattr(pipeline, "rank_select_funds", fake_rank_select_funds)
        mp.setattr(pipeline, "compute_trend_signals", fake_trend_signals)
        mp.setattr("trend_analysis.plugins.create_weight_engine", boom_engine)
        mp.setattr(pipeline, "compute_constrained_weights", boom_weights)
        mp.setattr(pipeline, "realised_volatility", fake_realised_volatility)
        mp.setattr(pipeline, "information_ratio", fake_information_ratio)
        mp.setattr(pipeline, "build_regime_payload", fake_regime_payload)

        result = _run_analysis(
            prepared,
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-04",
            target_vol=0.1,
            monthly_cost=0.0,
            floor_vol=-1,
            warmup_periods=2,
            selection_mode="rank",
            random_n=1,
            custom_weights=None,
            rank_kwargs={"top_n": 1},
            manual_funds=None,
            indices_list=None,
            benchmarks={"RF": "RF"},
            seed=1,
            weighting_scheme="erc",
            constraints={"max_weight": "bad"},
            missing_policy="drop",
            missing_limit=1,
            risk_window={"length": "bad", "lambda": "bad"},
            previous_weights=None,
            max_turnover="bad",
            signal_spec=TrendSpec(
                window=2, min_periods=None, lag=1, vol_adjust=False, vol_target=None, zscore=False
            ),
            regime_cfg={},
        )

    assert result is not None
    assert result["weight_engine_fallback"]["engine"] == "erc"
    assert set(result["selected_funds"]) == {"FundA", "FundB"}


def test_run_analysis_zero_weight_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="M")
    prepared = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.0, -0.01, 0.02],
            "RF": [0.0, 0.0, 0.0],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    def fake_prepare(*args, **kwargs):
        return (prepared.copy(), freq_summary, missing_result, False)

    def fake_single_period_run(*args, **kwargs):
        frame = pd.DataFrame({"Sharpe": [0.5, 0.4]}, index=["FundA", "FundB"])
        frame.attrs["insample_len"] = 3
        frame.attrs["period"] = ("2020-01", "2020-02")
        return frame

    class DummyDiag(SimpleNamespace):
        pass

    def fake_weights(base, returns, **kwargs):
        weights = pd.Series({"FundA": 0.0, "FundB": 0.0})
        diag = DummyDiag(
            asset_volatility=pd.DataFrame({"FundA": [0.2], "FundB": [0.3]}),
            portfolio_volatility=pd.Series([0.1], name="portfolio"),
            turnover=pd.Series(dtype=float),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0, "FundB": 1.0}),
        )
        return weights, diag

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", fake_prepare)
        mp.setattr(pipeline, "single_period_run", fake_single_period_run)
        mp.setattr(pipeline, "compute_constrained_weights", fake_weights)
        mp.setattr(pipeline, "build_regime_payload", lambda **kwargs: {})

        result = _run_analysis(
            prepared,
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-03",
            target_vol=0.1,
            monthly_cost=0.0,
            custom_weights={"FundA": 0.0, "FundB": 0.0},
            weighting_scheme="equal",
            selection_mode="all",
            benchmarks=None,
            max_turnover=0.5,
            signal_spec=TrendSpec(window=2, min_periods=None, lag=1, vol_adjust=False, vol_target=None, zscore=False),
            regime_cfg={},
        )

    assert result is not None
    assert set(result["fund_weights"]) == {"FundA", "FundB"}

def test_compute_signal_error_paths(monthly_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        compute_signal(monthly_frame, column="missing")
    with pytest.raises(ValueError):
        compute_signal(monthly_frame, column="A", window=0)
    with pytest.raises(ValueError):
        compute_signal(monthly_frame, column="A", window=2, min_periods=0)


def test_run_uses_nan_policy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="M"),
            "FundA": [0.01, 0.02, 0.0, 0.03],
            "FundB": [0.0, -0.01, 0.02, 0.01],
        }
    )
    captured: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        return df

    stats_result = {
        "FundA": _Stats(0.1, 0.2, 1.0, 0.5, -0.1, 0.3),
        "FundB": _Stats(0.2, 0.3, 0.8, 0.4, -0.2, 0.2),
    }

    def fake_run_analysis(*args, **kwargs):
        captured["analysis_kwargs"] = kwargs
        return {"out_sample_stats": stats_result, "benchmark_ir": {}}

    cfg = SimpleNamespace(
        data={
            "csv_path": "dummy.csv",
            "nan_policy": {"FundA": "ffill"},
            "nan_limit": {"FundA": 1},
        },
        sample_split={},
        metrics={"registry": ["Sharpe"]},
        preprocessing={"missing_data": "skip"},
        vol_adjust={"target_vol": 1.0},
        run={"monthly_cost": 0.0},
        portfolio={},
        benchmarks={},
        seed=123,
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "load_csv", fake_load_csv)
        mp.setattr(
            pipeline,
            "_resolve_sample_split",
            lambda df, cfg: {
                "in_start": "2020-01",
                "in_end": "2020-02",
                "out_start": "2020-03",
                "out_end": "2020-04",
            },
        )
        mp.setattr(pipeline, "_build_trend_spec", lambda cfg, vol: SimpleNamespace())
        mp.setattr(pipeline, "_run_analysis", fake_run_analysis)

        result = pipeline.run(cfg)

    assert captured["missing_policy"] == {"FundA": "ffill"}
    assert captured["missing_limit"] == {"FundA": 1}
    assert "FundA" in result.index
    assert "FundB" in result.index


def test_run_full_passes_through_results(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="M"),
            "FundA": [0.01, 0.02, 0.0, 0.03],
            "FundB": [0.0, -0.01, 0.02, 0.01],
        }
    )

    captured: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        return df

    payload = {"selected_funds": ["FundA"], "regime_summary": "ok"}

    def fake_run_analysis(*args, **kwargs):
        return payload

    cfg = SimpleNamespace(
        data={
            "csv_path": "dummy.csv",
            "nan_policy": "drop",
            "nan_limit": {"FundA": 1},
        },
        sample_split={},
        metrics={"registry": ["Sharpe"]},
        preprocessing={"missing_data": {}},
        vol_adjust={"target_vol": 1.0},
        run={"monthly_cost": 0.0},
        portfolio={"weighting_scheme": "equal"},
        benchmarks={},
        seed=0,
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "load_csv", fake_load_csv)
        mp.setattr(pipeline, "_resolve_sample_split", lambda df, cfg: {
            "in_start": "2020-01",
            "in_end": "2020-02",
            "out_start": "2020-03",
            "out_end": "2020-04",
        })
        mp.setattr(pipeline, "_build_trend_spec", lambda cfg, vol: SimpleNamespace())
        mp.setattr(pipeline, "_run_analysis", fake_run_analysis)

        result = pipeline.run_full(cfg)

    assert result == payload
    assert captured["missing_policy"] == "drop"
    assert captured["missing_limit"] == {"FundA": 1}


def test_single_period_run_basic_metrics() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="M"),
            "FundA": [0.01, 0.02, -0.01, 0.03],
            "FundB": [0.0, 0.01, 0.02, -0.01],
        }
    )
    out = single_period_run(df, "2020-01", "2020-03")
    assert set(out.columns) >= {"Sharpe", "Sortino"}
    assert out.index.tolist() == ["FundA", "FundB"]


def test_single_period_run_requires_date_column() -> None:
    df = pd.DataFrame({"FundA": [0.01, 0.02]})
    with pytest.raises(ValueError, match="'Date'"):
        single_period_run(df, "2020-01", "2020-02")


def test_single_period_run_coerces_string_dates() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.0, -0.01, 0.02],
        }
    )
    out = single_period_run(df, "2020-01", "2020-03")
    assert out.attrs["period"] == ("2020-01", "2020-03")


def test_single_period_run_rejects_empty_metrics() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=2, freq="M"),
            "FundA": [0.01, 0.02],
        }
    )

    class EmptyConfig(SimpleNamespace):
        metrics_to_run: list[str] = []

    with pytest.raises(ValueError, match="must not be empty"):
        single_period_run(df, "2020-01", "2020-02", stats_cfg=EmptyConfig())


def test_compute_signal_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    class RaisingIndex(pd.DatetimeIndex):
        @property
        def freq(self):  # type: ignore[override]
            raise RuntimeError("freq unavailable")

    data = pd.DataFrame(
        {"returns": [0.1, -0.2, 0.05, 0.03]},
        index=RaisingIndex(pd.date_range("2020-01-31", periods=4, freq="M")),
    )

    class DummyCache:
        def __init__(self) -> None:
            self.called = False

        def is_enabled(self) -> bool:
            return True

        def get_or_compute(self, *args):
            self.called = True
            compute_fn = args[-1]
            return compute_fn()

    cache = DummyCache()
    monkeypatch.setattr(pipeline, "get_cache", lambda: cache)

    series = compute_signal(data, column="returns", window=2, min_periods=1)
    assert cache.called is True
    assert series.name == "returns_signal"


def test_compute_signal_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {"returns": [0.1, -0.2, 0.05]},
        index=pd.date_range("2020-01-31", periods=3, freq="M"),
    )

    class DummyCache:
        def is_enabled(self) -> bool:
            return False

    monkeypatch.setattr(pipeline, "get_cache", lambda: DummyCache())
    series = compute_signal(frame, column="returns", window=2, min_periods=1)
    assert isinstance(series, pd.Series)
    assert series.name == "returns_signal"
    assert series.index.equals(frame.index)


def test_position_from_signal_behaviour() -> None:
    signal = pd.Series(
        [np.nan, 0.2, 0.0, -0.5, np.nan],
        index=pd.date_range("2020-01-31", periods=5, freq="M"),
        name="signal",
    )
    positions = pipeline.position_from_signal(
        signal, long_position=2.0, short_position=-1.5, neutral_position=0.0
    )
    assert positions.tolist() == [0.0, 2.0, 2.0, -1.5, -1.5]


def test_pipeline_getattr_unknown() -> None:
    with pytest.raises(AttributeError):
        pipeline.__getattr__("does_not_exist")


def test_module_getattr_invocation() -> None:
    with pytest.raises(AttributeError):
        getattr(pipeline, "missing_attribute")


def test_module_getattr_stats_alias() -> None:
    assert getattr(pipeline, "Stats") is _Stats


def test_run_analysis_random_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="M")
    prepared = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03, 0.04, 0.05],
            "FundB": [0.0, -0.01, 0.02, 0.01, -0.02],
            "RF": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    def fake_prepare(*args, **kwargs):
        return (prepared.copy(), freq_summary, missing_result, False)

    class DummyDiag(SimpleNamespace):
        pass

    def fake_weights(base, returns, **kwargs):
        weights = pd.Series({"FundA": 0.6, "FundB": 0.4})
        diag = DummyDiag(
            asset_volatility=pd.DataFrame({"FundA": [0.2], "FundB": [0.3]}),
            portfolio_volatility=pd.Series([0.15], name="portfolio"),
            turnover=pd.Series(dtype=float),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0, "FundB": 1.0}),
        )
        return weights, diag

    class DummyRng:
        def choice(self, seq, size, replace):
            return np.array(seq[:size])

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", fake_prepare)
        mp.setattr(pipeline, "single_period_run", lambda *args, **kwargs: pd.DataFrame({"Sharpe": [0.4]}, index=["FundA"]))
        mp.setattr(pipeline, "compute_constrained_weights", fake_weights)
        mp.setattr(pipeline, "build_regime_payload", lambda **kwargs: {})
        mp.setattr(pipeline, "realised_volatility", lambda data, window, periods_per_year=None: pd.DataFrame({col: [0.2, 0.2] for col in data.columns}))
        mp.setattr(pipeline, "information_ratio", lambda *args, **kwargs: 0.0)
        mp.setattr(np.random, "default_rng", lambda seed=None: DummyRng())

        result = _run_analysis(
            prepared,
            "2020-01",
            "2020-03",
            "2020-04",
            "2020-05",
            target_vol=0.1,
            monthly_cost=0.0,
            warmup_periods=-1,
            selection_mode="random",
            random_n=1,
            custom_weights=None,
            weighting_scheme="equal",
            constraints="invalid",
            missing_policy="drop",
            missing_limit=None,
            risk_window={"length": 0},
            previous_weights=None,
            max_turnover=None,
            signal_spec=TrendSpec(window=2, min_periods=None, lag=1, vol_adjust=False, vol_target=None, zscore=False),
            regime_cfg={},
        )

    assert result is not None
    assert result["selected_funds"]


def test_run_analysis_returns_none_when_copy_becomes_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PretendNonEmpty(pd.DataFrame):
        @property
        def _constructor(self):  # pragma: no cover - pandas protocol
            return PretendNonEmpty

        @property
        def empty(self) -> bool:  # type: ignore[override]
            return False

        def copy(self, deep: bool = True):  # type: ignore[override]
            base = pd.DataFrame(self)
            return base.iloc[0:0].copy()

    pretend = PretendNonEmpty({"Date": pd.to_datetime([]), "FundA": []})
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    with monkeypatch.context() as mp:
        mp.setattr(
            pipeline,
            "_prepare_input_data",
            lambda *args, **kwargs: (pretend, freq_summary, missing_result, False),
        )
        result = _run_analysis(
            pretend,
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-04",
            target_vol=0.1,
            monthly_cost=0.0,
        )

    assert result is None


def test_run_analysis_returns_none_when_ret_cols_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShrinkingColumnsFrame(pd.DataFrame):
        _metadata = ["_columns_calls", "_true_columns", "_debug_calls"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._columns_calls = 0
            self._true_columns = pd.Index(super().columns)
            self._debug_calls: list[int] = []

        @property
        def _constructor(self):  # pragma: no cover - pandas protocol
            return ShrinkingColumnsFrame

        def copy(self, deep: bool = True):  # type: ignore[override]
            result = ShrinkingColumnsFrame(super().copy(deep=deep))
            result._columns_calls = 0
            result._true_columns = self._true_columns
            result._debug_calls = self._debug_calls
            return result

        @property
        def columns(self):  # type: ignore[override]
            self._columns_calls += 1
            self._debug_calls.append(self._columns_calls)
            if self._columns_calls <= 2:
                return self._true_columns
            return pd.Index(["Date"])

    base = ShrinkingColumnsFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "FundA": [0.0, 0.1, -0.1],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    with monkeypatch.context() as mp:
        mp.setattr(
            pipeline,
            "_prepare_input_data",
            lambda *args, **kwargs: (base, freq_summary, missing_result, False),
        )
        result = _run_analysis(
            base,
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-03",
            target_vol=0.1,
            monthly_cost=0.0,
        )

    assert result is None
    assert base._debug_calls[-3:] == [1, 2, 3]


def test_run_analysis_weight_engine_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="M")
    prepared = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.01, 0.03, 0.04],
            "FundB": [0.01, -0.02, 0.02, 0.00],
            "RF": [0.0, 0.0, 0.0, 0.0],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    class DummyDiag(SimpleNamespace):
        pass

    def fake_weights(base, returns, **kwargs):
        weights = pd.Series({"FundA": 0.7, "FundB": 0.3})
        diag = DummyDiag(
            asset_volatility=pd.DataFrame({"FundA": [0.2], "FundB": [0.25]}),
            portfolio_volatility=pd.Series([0.18], name="portfolio"),
            turnover=pd.Series(dtype=float),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0, "FundB": 1.0}),
        )
        return weights, diag

    class DummyEngine:
        def weight(self, cov: pd.DataFrame) -> pd.Series:
            return pd.Series({"FundA": 0.9, "FundB": 0.1})

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", lambda *a, **k: (prepared.copy(), freq_summary, missing_result, False))
        mp.setattr(pipeline, "single_period_run", lambda *a, **k: pd.DataFrame({"Sharpe": [0.5, 0.4]}, index=["FundA", "FundB"]))
        mp.setattr(pipeline, "compute_constrained_weights", fake_weights)
        mp.setattr("trend_analysis.plugins.create_weight_engine", lambda scheme: DummyEngine())
        mp.setattr(pipeline, "build_regime_payload", lambda **kwargs: {})
        mp.setattr(pipeline, "compute_trend_signals", lambda *args, **kwargs: pd.DataFrame(args[0]))
        mp.setattr(pipeline, "realised_volatility", lambda data, window, periods_per_year=None: pd.DataFrame({col: [0.2, 0.2] for col in data.columns}))
        mp.setattr(pipeline, "information_ratio", lambda *args, **kwargs: 0.0)

        result = _run_analysis(
            prepared,
            "2020-01",
            "2020-03",
            "2020-04",
            "2020-04",
            target_vol=0.1,
            monthly_cost=0.0,
            weighting_scheme="custom",
        )

    assert result is not None
    assert result["weight_engine_fallback"] is None
    assert result["fund_weights"]["FundA"] != result["fund_weights"]["FundB"]


def test_run_analysis_uses_empty_signal_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class SignalFrame(pd.DataFrame):
        _metadata = ["_force_empty_next"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._force_empty_next = False

        @property
        def _constructor(self):  # pragma: no cover - pandas protocol
            return SignalFrame

        def __finalize__(self, other, method=None):  # pragma: no cover - pandas protocol
            self._force_empty_next = False
            return self

        def copy(self, deep: bool = True):  # type: ignore[override]
            result = SignalFrame(super().copy(deep=deep))
            result._force_empty_next = True
            return result

        def set_index(self, keys, drop: bool = True, inplace: bool = False, verify_integrity: bool = False):  # type: ignore[override]
            result = super().set_index(keys, drop=drop, inplace=inplace, verify_integrity=verify_integrity)
            if self._force_empty_next:
                self._force_empty_next = False
                return result.iloc[0:0].copy()
            return SignalFrame(result)

    dates = pd.date_range("2020-01-31", periods=2, freq="M")
    frame = SignalFrame({"Date": dates, "FundA": [0.01, 0.02], "RF": [0.0, 0.0]})
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    class DummyDiag(SimpleNamespace):
        pass

    def fake_weights(base, returns, **kwargs):
        weights = pd.Series({"FundA": 1.0})
        diag = DummyDiag(
            asset_volatility=pd.DataFrame({"FundA": [0.2]}),
            portfolio_volatility=pd.Series([0.18], name="portfolio"),
            turnover=pd.Series(dtype=float),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0}),
        )
        return weights, diag

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", lambda *a, **k: (frame, freq_summary, missing_result, False))
        mp.setattr(pipeline, "single_period_run", lambda *a, **k: pd.DataFrame({"Sharpe": [0.3]}, index=["FundA"]))
        mp.setattr(pipeline, "compute_constrained_weights", fake_weights)
        mp.setattr(pipeline, "build_regime_payload", lambda **kwargs: {})
        mp.setattr(pipeline, "compute_trend_signals", lambda *args, **kwargs: pd.DataFrame(args[0]))
        mp.setattr(pipeline, "realised_volatility", lambda data, window, periods_per_year=None: pd.DataFrame({col: [0.2] for col in data.columns}))
        mp.setattr(pipeline, "information_ratio", lambda *args, **kwargs: 0.0)

        result = _run_analysis(
            frame,
            "2020-01",
            "2020-01",
            "2020-02",
            "2020-02",
            target_vol=0.1,
            monthly_cost=0.0,
        )

    assert result is not None
    signal_frame = result["signal_frame"]
    assert isinstance(signal_frame, pd.DataFrame)
    assert signal_frame.empty


def test_run_analysis_warmup_zeroes_initial_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="M")
    prepared = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.01, 0.03, 0.04, 0.05],
            "FundB": [0.01, -0.02, 0.02, 0.00, 0.01],
            "RF": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    class DummyDiag(SimpleNamespace):
        pass

    def fake_weights(base, returns, **kwargs):
        weights = pd.Series({"FundA": 0.5, "FundB": 0.5})
        diag = DummyDiag(
            asset_volatility=pd.DataFrame({"FundA": [0.2], "FundB": [0.2]}),
            portfolio_volatility=pd.Series([0.18], name="portfolio"),
            turnover=pd.Series(dtype=float),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0, "FundB": 1.0}),
        )
        return weights, diag

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", lambda *a, **k: (prepared.copy(), freq_summary, missing_result, False))
        mp.setattr(pipeline, "single_period_run", lambda *a, **k: pd.DataFrame({"Sharpe": [0.3, 0.2]}, index=["FundA", "FundB"]))
        mp.setattr(pipeline, "compute_constrained_weights", fake_weights)
        mp.setattr(pipeline, "build_regime_payload", lambda **kwargs: {})
        mp.setattr(pipeline, "compute_trend_signals", lambda *args, **kwargs: pd.DataFrame(args[0]))
        mp.setattr(pipeline, "realised_volatility", lambda data, window, periods_per_year=None: pd.DataFrame({col: [0.2, 0.2] for col in data.columns}))
        mp.setattr(pipeline, "information_ratio", lambda *args, **kwargs: 0.0)

        result = _run_analysis(
            prepared,
            "2020-01",
            "2020-03",
            "2020-04",
            "2020-05",
            target_vol=0.1,
            monthly_cost=0.0,
            warmup_periods=2,
        )

    assert result is not None
    in_scaled = result["in_sample_scaled"]
    out_scaled = result["out_sample_scaled"]
    assert (in_scaled.iloc[:2] == 0.0).all().all()
    assert (out_scaled.iloc[:2] == 0.0).all().all()


def test_run_analysis_adds_valid_indices_and_skips_missing_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="M")
    prepared = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.01, 0.03, 0.04],
            "FundB": [0.01, 0.0, 0.02, -0.01],
            "RF": [0.0, 0.0, 0.0, 0.0],
            "Index": [0.005, 0.004, 0.003, 0.002],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    class DummyDiag(SimpleNamespace):
        pass

    def fake_weights(base, returns, **kwargs):
        weights = pd.Series({"FundA": 0.6, "FundB": 0.4})
        diag = DummyDiag(
            asset_volatility=pd.DataFrame({"FundA": [0.2], "FundB": [0.25]}),
            portfolio_volatility=pd.Series([0.18], name="portfolio"),
            turnover=pd.Series(dtype=float),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0, "FundB": 1.0}),
        )
        return weights, diag

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", lambda *a, **k: (prepared.copy(), freq_summary, missing_result, False))
        mp.setattr(pipeline, "single_period_run", lambda *a, **k: pd.DataFrame({"Sharpe": [0.3, 0.2]}, index=["FundA", "FundB"]))
        mp.setattr(pipeline, "compute_constrained_weights", fake_weights)
        mp.setattr(pipeline, "build_regime_payload", lambda **kwargs: {})
        mp.setattr(pipeline, "compute_trend_signals", lambda *args, **kwargs: pd.DataFrame(args[0]))
        mp.setattr(pipeline, "realised_volatility", lambda data, window, periods_per_year=None: pd.DataFrame({col: [0.2, 0.2] for col in data.columns}))
        mp.setattr(pipeline, "information_ratio", lambda *args, **kwargs: 0.0)

        result = _run_analysis(
            prepared,
            "2020-01",
            "2020-03",
            "2020-04",
            "2020-04",
            target_vol=0.1,
            monthly_cost=0.0,
            indices_list=["Index"],
            benchmarks={"Missing": "Missing"},
        )

    assert result is not None
    assert "Index" in result["benchmark_stats"]
    assert "Missing" not in result["benchmark_stats"]


def test_run_analysis_handles_benchmark_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="M")
    prepared = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.01, 0.03],
            "FundB": [0.01, 0.0, 0.02],
            "RF": [0.0, 0.0, 0.0],
            "Index": [0.005, 0.004, 0.003],
        }
    )
    freq_summary = FrequencySummary("M", "Monthly", False, "M", "Monthly")
    missing_result = MissingPolicyResult(
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    class DummyDiag(SimpleNamespace):
        pass

    def fake_weights(base, returns, **kwargs):
        weights = pd.Series({"FundA": 0.6, "FundB": 0.4})
        diag = DummyDiag(
            asset_volatility=pd.DataFrame({"FundA": [0.2], "FundB": [0.25]}),
            portfolio_volatility=pd.Series([0.18], name="portfolio"),
            turnover=pd.Series(dtype=float),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0, "FundB": 1.0}),
        )
        return weights, diag

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "_prepare_input_data", lambda *a, **k: (prepared.copy(), freq_summary, missing_result, False))
        mp.setattr(pipeline, "single_period_run", lambda *a, **k: pd.DataFrame({"Sharpe": [0.3, 0.2]}, index=["FundA", "FundB"]))
        mp.setattr(pipeline, "compute_constrained_weights", fake_weights)
        mp.setattr(pipeline, "build_regime_payload", lambda **kwargs: {})
        mp.setattr(pipeline, "compute_trend_signals", lambda *args, **kwargs: pd.DataFrame(args[0]))
        mp.setattr(pipeline, "realised_volatility", lambda data, window, periods_per_year=None: pd.DataFrame({col: [0.2, 0.2] for col in data.columns}))
        mp.setattr(pipeline, "information_ratio", lambda *args, **kwargs: 0.0)

        result = _run_analysis(
            prepared,
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-03",
            target_vol=0.1,
            monthly_cost=0.0,
            indices_list=["Index"],
            benchmarks={"Index": "Index"},
        )

    assert result is not None
    stats = result["benchmark_stats"]
    assert "Index" in stats


def test_run_missing_policy_and_limit_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="M"),
            "FundA": [0.01, 0.02, 0.0, 0.03],
            "FundB": [0.0, -0.01, 0.02, 0.01],
        }
    )

    captured: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        captured["policy"] = missing_policy
        captured["limit"] = missing_limit
        return df

    def fake_run_analysis(*args, **kwargs):
        return {"out_sample_stats": {"FundA": _Stats(0, 0, 0, 0, 0, 0)}, "benchmark_ir": {}}

    cfg = SimpleNamespace(
        data={"csv_path": "dummy.csv", "nan_policy": "drop", "nan_limit": {"FundA": 1}},
        sample_split={},
        metrics={},
        preprocessing={"missing_data": "skip"},
        vol_adjust={"target_vol": 1.0},
        run={"monthly_cost": 0.0},
        portfolio={},
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "load_csv", fake_load_csv)
        mp.setattr(pipeline, "_resolve_sample_split", lambda df, cfg: {"in_start": "2020-01", "in_end": "2020-02", "out_start": "2020-03", "out_end": "2020-04"})
        mp.setattr(pipeline, "_build_trend_spec", lambda cfg, vol: SimpleNamespace())
        mp.setattr(pipeline, "_run_analysis", fake_run_analysis)

        result = pipeline.run(cfg)

    assert captured["policy"] == "drop"
    assert captured["limit"] == {"FundA": 1}
    assert isinstance(result, pd.DataFrame)


def test_run_respects_explicit_missing_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "FundA": [0.01, 0.02, 0.03],
        }
    )

    captured: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        captured["policy"] = missing_policy
        captured["limit"] = missing_limit
        return df

    def fake_run_analysis(*args, **kwargs):
        return {"out_sample_stats": {"FundA": _Stats(0, 0, 0, 0, 0, 0)}, "benchmark_ir": {}}

    cfg = SimpleNamespace(
        data={"csv_path": "dummy.csv", "missing_policy": "keep", "missing_limit": 2},
        sample_split={},
        metrics={},
        preprocessing={"missing_data": {}},
        vol_adjust={"target_vol": 1.0},
        run={"monthly_cost": 0.0},
        portfolio={},
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "load_csv", fake_load_csv)
        mp.setattr(pipeline, "_resolve_sample_split", lambda df, cfg: {"in_start": "2020-01", "in_end": "2020-01", "out_start": "2020-02", "out_end": "2020-03"})
        mp.setattr(pipeline, "_build_trend_spec", lambda cfg, vol: SimpleNamespace())
        mp.setattr(pipeline, "_run_analysis", fake_run_analysis)

        pipeline.run(cfg)

    assert captured["policy"] == "keep"
    assert captured["limit"] == 2


def test_run_full_requires_csv_path() -> None:
    cfg = SimpleNamespace(data={})
    with pytest.raises(KeyError):
        pipeline.run_full(cfg)


def test_run_full_uses_nan_policy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="M"),
            "FundA": [0.01, 0.02, 0.0, 0.03],
            "FundB": [0.0, -0.01, 0.02, 0.01],
        }
    )
    captured: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        captured["policy"] = missing_policy
        captured["limit"] = missing_limit
        return df

    def fake_run_analysis(*args, **kwargs):
        return {"selected_funds": ["FundA"], "regime_summary": "ok"}

    cfg = SimpleNamespace(
        data={"csv_path": "dummy.csv", "nan_policy": "drop", "nan_limit": {"FundA": 1}},
        sample_split={},
        metrics={},
        preprocessing={"missing_data": {}},
        vol_adjust={"target_vol": 1.0},
        run={"monthly_cost": 0.0},
        portfolio={"weighting_scheme": "equal"},
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "load_csv", fake_load_csv)
        mp.setattr(pipeline, "_resolve_sample_split", lambda df, cfg: {"in_start": "2020-01", "in_end": "2020-02", "out_start": "2020-03", "out_end": "2020-04"})
        mp.setattr(pipeline, "_build_trend_spec", lambda cfg, vol: SimpleNamespace())
        mp.setattr(pipeline, "_run_analysis", fake_run_analysis)

        result = pipeline.run_full(cfg)

    assert captured["policy"] == "drop"
    assert captured["limit"] == {"FundA": 1}
    assert result["selected_funds"] == ["FundA"]


def test_run_full_respects_explicit_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="M"),
            "FundA": [0.01, 0.02, 0.03],
        }
    )
    captured: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        captured["policy"] = missing_policy
        captured["limit"] = missing_limit
        return df

    def fake_run_analysis(*args, **kwargs):
        return {"selected_funds": ["FundA"]}

    cfg = SimpleNamespace(
        data={"csv_path": "dummy.csv", "missing_policy": "keep", "missing_limit": 2},
        sample_split={},
        metrics={},
        preprocessing={"missing_data": {}},
        vol_adjust={"target_vol": 1.0},
        run={"monthly_cost": 0.0},
        portfolio={"weighting_scheme": "equal"},
    )

    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "load_csv", fake_load_csv)
        mp.setattr(pipeline, "_resolve_sample_split", lambda df, cfg: {"in_start": "2020-01", "in_end": "2020-01", "out_start": "2020-02", "out_end": "2020-03"})
        mp.setattr(pipeline, "_build_trend_spec", lambda cfg, vol: SimpleNamespace())
        mp.setattr(pipeline, "_run_analysis", fake_run_analysis)

        pipeline.run_full(cfg)

    assert captured["policy"] == "keep"
    assert captured["limit"] == 2
