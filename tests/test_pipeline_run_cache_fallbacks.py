import types

import numpy as np
import pandas as pd
import pytest

import trend_analysis.pipeline as pipeline


class DummyCache:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def is_enabled(self) -> bool:
        return True

    def get_or_compute(self, *args):
        self.calls.append(args)
        compute_fn = args[-1]
        return compute_fn()


def _stats_payload() -> dict[str, pipeline._Stats]:  # type: ignore[attr-defined]
    return {
        "FundA": pipeline._Stats(  # type: ignore[attr-defined]
            cagr=0.1,
            vol=0.2,
            sharpe=1.5,
            sortino=1.2,
            max_drawdown=-0.05,
            information_ratio=0.9,
        )
    }


def test_run_uses_nan_policy_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        calls["policy"] = missing_policy
        calls["limit"] = missing_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=4, freq="ME"),
                "RF": 0.0,
                "FundA": [0.01, 0.02, 0.0, 0.03],
            }
        )

    def fake_run_analysis(*args, **kwargs):
        return {
            "out_sample_stats": _stats_payload(),
            "benchmark_ir": {
                "Bench": {"FundA": 0.5, "equal_weight": 1.0, "user_weight": 0.7}
            },
        }

    monkeypatch.setattr(pipeline, "load_csv", fake_load_csv)
    monkeypatch.setattr(pipeline, "_run_analysis", fake_run_analysis)

    cfg = {
        "data": {
            "csv_path": "dummy.csv",
            "nan_policy": {"default": "zero"},
            "nan_limit": {"default": 2},
        },
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-02",
            "out_start": "2020-03",
            "out_end": "2020-04",
        },
        "vol_adjust": {"target_vol": 0.1},
        "run": {"monthly_cost": 0.0},
        "portfolio": {},
    }

    result = pipeline.run(cfg)

    assert calls["policy"] == {"default": "zero"}
    assert calls["limit"] == {"default": 2}
    assert result.index.tolist() == ["FundA"]
    assert "ir_Bench" in result.columns
    assert set(result.columns) >= {"cagr", "vol", "sharpe", "ir_Bench"}


def test_run_full_uses_nan_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        calls["policy"] = missing_policy
        calls["limit"] = missing_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=4, freq="ME"),
                "RF": 0.0,
                "FundA": [0.01, 0.02, 0.0, 0.03],
            }
        )

    payload = {
        "out_sample_stats": _stats_payload(),
        "benchmark_ir": {"Bench": {"FundA": 0.2}},
        "extra": 1,
    }

    monkeypatch.setattr(pipeline, "load_csv", fake_load_csv)
    monkeypatch.setattr(pipeline, "_run_analysis", lambda *a, **k: payload)

    cfg = {
        "data": {
            "csv_path": "dummy.csv",
            "nan_policy": {"default": "ffill"},
            "nan_limit": {"default": 1},
        },
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-02",
            "out_start": "2020-03",
            "out_end": "2020-04",
        },
        "vol_adjust": {"target_vol": 0.1},
        "run": {"monthly_cost": 0.0},
        "portfolio": {},
    }

    result = pipeline.run_full(cfg)

    assert calls["policy"] == {"default": "ffill"}
    assert calls["limit"] == {"default": 1}
    assert result["out_sample_stats"] == payload["out_sample_stats"]
    assert result["extra"] == 1


def test_run_requires_csv_path() -> None:
    with pytest.raises(KeyError):
        pipeline.run({"data": {}})


def test_run_full_requires_csv_path() -> None:
    with pytest.raises(KeyError):
        pipeline.run_full({"data": {}})


def test_run_respects_explicit_missing_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        observed["policy"] = missing_policy
        observed["limit"] = missing_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
                "RF": 0.0,
                "FundA": [0.01, 0.02, 0.03],
            }
        )

    monkeypatch.setattr(pipeline, "load_csv", fake_load_csv)
    monkeypatch.setattr(pipeline, "_run_analysis", lambda *a, **k: pipeline._empty_run_full_result())

    cfg = {
        "data": {
            "csv_path": "dummy.csv",
            "missing_policy": {"default": "zero"},
            "missing_limit": {"default": 3},
        },
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-01",
            "out_start": "2020-02",
            "out_end": "2020-02",
        },
    }

    result = pipeline.run(cfg)
    assert result.empty
    assert observed["policy"] == {"default": "zero"}
    assert observed["limit"] == {"default": 3}


def test_run_full_handles_missing_data_section(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_load_csv(path: str, *, errors: str, missing_policy, missing_limit):
        observed["policy"] = missing_policy
        observed["limit"] = missing_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
                "RF": 0.0,
                "FundA": [0.01, 0.02, 0.03],
            }
        )

    def fake_run_analysis(*args, **kwargs):
        return {
            "out_sample_stats": _stats_payload(),
            "benchmark_ir": {},
        }

    monkeypatch.setattr(pipeline, "load_csv", fake_load_csv)
    monkeypatch.setattr(pipeline, "_run_analysis", fake_run_analysis)

    cfg = {
        "data": {"csv_path": "dummy.csv"},
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-01",
            "out_start": "2020-02",
            "out_end": "2020-02",
        },
        "metrics": {"registry": ["Sharpe"]},
        "preprocessing": {"missing_data": "unsupported"},
    }

    payload = pipeline.run_full(cfg)
    assert observed["policy"] is None
    assert observed["limit"] is None
    assert payload["benchmark_ir"] == {}

def test_compute_signal_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-01", periods=6, freq="D"), "value": np.arange(6)}
    ).set_index("Date")

    cache = DummyCache()
    monkeypatch.setattr(pipeline, "get_cache", lambda: cache)

    series = pipeline.compute_signal(df, column="value", window=3, min_periods=2)

    assert cache.calls, "Expected cache to be used when enabled"
    dataset_hash, window_arg, freq_tag, method_tag, _ = cache.calls[-1]
    assert window_arg == 3
    assert method_tag.endswith("min2")
    assert series.name == "value_signal"
    assert series.index.equals(df.index)


def test_preprocessing_summary_monthly_branch() -> None:
    summary = pipeline._preprocessing_summary("M", normalised=False, missing_summary="ok")
    assert "month-end" in summary
    assert "Missing data" in summary
    daily = pipeline._preprocessing_summary("D", normalised=True, missing_summary=None)
    assert "monthly" in daily.lower()


def test_cfg_section_and_section_get_defaults() -> None:
    cfg = {"present": {"value": 1}}
    assert pipeline._cfg_section(cfg, "missing") == {}
    ns = types.SimpleNamespace(answer=42)
    assert pipeline._section_get(ns, "answer", default=None) == 42
    assert pipeline._section_get(None, "missing", default="fallback") == "fallback"
    assert pipeline._unwrap_cfg({"__cfg__": {"__cfg__": None}}) == {"__cfg__": None}
    empty = pipeline._empty_run_full_result()
    assert set(empty) == {"out_sample_stats", "in_sample_stats", "benchmark_ir", "risk_diagnostics", "fund_weights"}


def test_derive_split_edge_cases() -> None:
    with pytest.raises(ValueError):
        pipeline._derive_split_from_periods(
            pd.PeriodIndex([]), method="date", boundary=None, ratio=0.5
        )

    single = pd.period_range("2020-01", periods=1, freq="M")
    result = pipeline._derive_split_from_periods(
        single, method="date", boundary=None, ratio=0.5
    )
    assert result["in_start"] == "2020-01"
    assert result["out_end"] == "2020-01"


def test_position_from_signal_tracks_state() -> None:
    data = pd.Series(
        [0.0, 1.0, np.nan, -1.0, 0.0], index=pd.date_range("2020-01-01", periods=5)
    )
    positions = pipeline.position_from_signal(
        data, long_position=2.0, short_position=-2.0, neutral_position=0.5
    )
    assert positions.iloc[0] == pytest.approx(0.5)
    assert positions.iloc[1] == pytest.approx(2.0)
    assert positions.iloc[2] == pytest.approx(2.0)
    assert positions.iloc[3] == pytest.approx(-2.0)
    assert positions.iloc[4] == pytest.approx(-2.0)

def test_policy_from_config_scalar_defaults() -> None:
    policy, limit = pipeline._policy_from_config({})
    assert policy is None
    assert limit is None

    policy_only, limit_only = pipeline._policy_from_config({"policy": "drop"})
    assert policy_only == "drop"
    assert limit_only is None

    none_map, none_limit = pipeline._policy_from_config({"per_asset": []})
    assert none_map is None
    assert none_limit is None


def test_build_trend_spec_uses_vol_adjust_default() -> None:
    cfg = {"signals": {"vol_adjust": True}}
    vol_cfg = {"enabled": True, "target_vol": "not-a-number"}
    spec = pipeline._build_trend_spec(cfg, vol_cfg)
    assert spec.vol_adjust is True
    assert spec.vol_target is None

    mixed = {
        "policy": "ffill",
        "per_asset": {"A": "zero"},
        "limit": 5,
        "per_asset_limit": {"A": 1},
    }
    policy_map, limit_map = pipeline._policy_from_config(mixed)
    assert policy_map == {"default": "ffill", "A": "zero"}
    assert limit_map == {"default": 5, "A": 1}


def test_resolve_sample_split_ratio_extremes() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "Fund": [0.1, 0.2, 0.0, 0.3, 0.1],
        }
    )
    cfg = {"ratio": -1.0}
    split = pipeline._resolve_sample_split(df, cfg)
    assert split["in_start"] == "2020-01"
    assert split["out_start"] > split["in_end"]

    cfg_high = {"ratio": 2.0}
    split_high = pipeline._resolve_sample_split(df, cfg_high)
    assert split_high["in_end"] < split_high["out_end"]
    assert split_high["out_start"] == "2020-05"


def test_single_period_run_validation() -> None:
    df = pd.DataFrame({"Value": [0.1, 0.2]})
    with pytest.raises(ValueError):
        pipeline.single_period_run(df, "2020-01", "2020-02")

    monthly = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=2, freq="ME"),
            "Fund": [0.1, 0.2],
        }
    )
    cfg = pipeline.RiskStatsConfig(metrics_to_run=())  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        pipeline.single_period_run(monthly, "2020-01", "2020-02", stats_cfg=cfg)


class _FailingFreqIndex(pd.DatetimeIndex):
    @property  # type: ignore[override]
    def freq(self):
        raise RuntimeError("freq failure")


def test_compute_signal_handles_freq_attribute_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    base_index = _FailingFreqIndex(pd.date_range("2020-01-01", periods=5, freq="D"))
    df = pd.DataFrame({"value": np.arange(5)}, index=base_index)

    class OffCache(DummyCache):
        def is_enabled(self) -> bool:
            return True

    cache = OffCache()
    monkeypatch.setattr(pipeline, "get_cache", lambda: cache)

    pipeline.compute_signal(df, column="value", window=2, min_periods=1)
    assert cache.calls, "Expected caching path despite freq attribute failure"


def test_compute_signal_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-01", periods=4, freq="D"), "value": [1.0, 2.0, 3.0, 4.0]}
    ).set_index("Date")

    class DisabledCache(DummyCache):
        def is_enabled(self) -> bool:
            return False

    monkeypatch.setattr(pipeline, "get_cache", lambda: DisabledCache())

    series = pipeline.compute_signal(df, column="value", window=2, min_periods=1)
    assert series.iloc[-1] != 0.0

def test_run_analysis_rank_branch_with_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": np.zeros(len(dates)),
            "FundA": [0.02, 0.01, 0.0, 0.03, 0.01, 0.02],
            "FundB": [0.01, 0.03, 0.02, 0.01, 0.0, 0.02],
        }
    )

    freq_summary = pipeline.FrequencySummary("M", "Monthly", False, "M", "Monthly")  # type: ignore[attr-defined]
    missing_summary = pipeline.MissingPolicyResult(  # type: ignore[attr-defined]
        policy={"FundA": "drop", "FundB": "drop"},
        default_policy="drop",
        limit={"FundA": None, "FundB": None},
        default_limit=None,
        filled={"FundA": 0, "FundB": 0},
        dropped_assets=(),
        summary="none",
    )

    def fake_prepare(*_args, **_kwargs):
        return df, freq_summary, missing_summary, False

    monkeypatch.setattr(pipeline, "_prepare_input_data", fake_prepare)

    captured: dict[str, list[str]] = {"funds": []}

    def fake_rank_select(sub, stats_cfg, **kwargs):  # noqa: ANN001
        captured["funds"].append(sorted(sub.columns.tolist()))
        return ["FundB"]

    monkeypatch.setattr(pipeline, "rank_select_funds", fake_rank_select)
    monkeypatch.setattr(pipeline, "get_window_metric_bundle", lambda key: {"key": key})
    monkeypatch.setattr(pipeline, "make_window_key", lambda *a, **k: "window-key")

    from trend_analysis.risk import RiskDiagnostics  # lazy import to avoid cycles

    def fake_compute_constrained_weights(*_args, **_kwargs):  # noqa: ANN001
        weights = pd.Series({"FundB": 0.0})
        diag = RiskDiagnostics(
            asset_volatility=pd.DataFrame(
                {"FundB": [0.1, 0.1]}, index=pd.Index([dates[0], dates[1]], name="Date")
            ),
            portfolio_volatility=pd.Series([0.1, 0.1], index=[dates[0], dates[1]]),
            turnover=pd.Series(dtype=float, name="turnover"),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundB": 1.0}),
        )
        return weights, diag

    monkeypatch.setattr(pipeline, "compute_constrained_weights", fake_compute_constrained_weights)
    monkeypatch.setattr(pipeline, "compute_trend_signals", lambda *_a, **_k: pd.DataFrame({"FundB": [0.0, 0.0]}))

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        selection_mode="rank",
        custom_weights={"FundB": 0.0},
        constraints="not-a-mapping",
        risk_window={"length": "bad", "lambda": "oops"},
        max_turnover="oops",
        warmup_periods=2,
    )

    assert result is not None
    assert result["selected_funds"] == ["FundB"]
    assert captured["funds"]
    warmup_slice = result["in_sample_scaled"].iloc[0:2]
    assert (warmup_slice == 0.0).all().all()


def test_run_analysis_risk_window_zero_length(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.02, 0.01, 0.02],
        }
    )

    freq_summary = pipeline.FrequencySummary("M", "Monthly", False, "M", "Monthly")  # type: ignore[attr-defined]
    missing_summary = pipeline.MissingPolicyResult(  # type: ignore[attr-defined]
        policy={"FundA": "drop", "FundB": "drop"},
        default_policy="drop",
        limit={"FundA": None, "FundB": None},
        default_limit=None,
        filled={"FundA": 0, "FundB": 0},
        dropped_assets=(),
        summary="none",
    )

    def fake_prepare(*_args, **_kwargs):
        return df, freq_summary, missing_summary, False

    monkeypatch.setattr(pipeline, "_prepare_input_data", fake_prepare)

    from trend_analysis.risk import RiskDiagnostics

    def fake_compute_constrained_weights(*_args, **_kwargs):
        weights = pd.Series({"FundA": 0.5, "FundB": 0.5})
        diag = RiskDiagnostics(
            asset_volatility=pd.DataFrame({"FundA": [0.1], "FundB": [0.1]}),
            portfolio_volatility=pd.Series([0.1]),
            turnover=pd.Series(dtype=float, name="turnover"),
            turnover_value=0.0,
            scale_factors=pd.Series({"FundA": 1.0, "FundB": 1.0}),
        )
        return weights, diag

    monkeypatch.setattr(pipeline, "compute_constrained_weights", fake_compute_constrained_weights)
    monkeypatch.setattr(pipeline, "compute_trend_signals", lambda *_a, **_k: pd.DataFrame({"FundA": [0.0], "FundB": [0.0]}))

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-03",
        0.1,
        0.0,
        risk_window={"length": 0},
        constraints={"max_weight": "oops"},
    )

    assert result is not None

def test_prepare_input_data_without_value_columns() -> None:
    df = pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=2, freq="ME")})
    monthly, summary, missing, normalised = pipeline._prepare_input_data(
        df, date_col="Date", missing_policy=None, missing_limit=None
    )
    assert monthly.columns.tolist() == ["Date"]
    assert summary.code == "M"
    assert missing.policy == {}
    assert normalised is False


def test_derive_split_ratio_nan_and_full_window() -> None:
    periods = pd.period_range("2020-01", periods=4, freq="M")
    result = pipeline._derive_split_from_periods(periods, method="ratio", boundary=None, ratio=float("nan"))
    assert result["in_start"] == "2020-01"
    assert result["out_start"] == "2020-03"

    result_high = pipeline._derive_split_from_periods(periods, method="ratio", boundary=None, ratio=1.0)
    assert result_high["out_start"] == "2020-04"


def test_run_analysis_returns_none_when_no_value_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    freq_summary = pipeline.FrequencySummary("M", "Monthly", False, "M", "Monthly")  # type: ignore[attr-defined]
    empty_missing = pipeline.MissingPolicyResult(  # type: ignore[attr-defined]
        policy={},
        default_policy="drop",
        limit={},
        default_limit=None,
        filled={},
        dropped_assets=(),
        summary="none",
    )

    def fake_prepare(*_args, **_kwargs):
        frame = pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=2, freq="ME")})
        return frame, freq_summary, empty_missing, False

    monkeypatch.setattr(pipeline, "_prepare_input_data", fake_prepare)

    result = pipeline._run_analysis(
        pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=2, freq="ME")}),
        "2020-01",
        "2020-01",
        "2020-02",
        "2020-02",
        0.1,
        0.0,
    )

    assert result is None


def test_single_period_run_converts_non_datetime_dates() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "Fund": [0.1, 0.2, 0.3],
        }
    )
    res = pipeline.single_period_run(df, "2020-01", "2020-03")
    assert isinstance(res, pd.DataFrame)
    assert res.attrs["period"] == ("2020-01", "2020-03")
