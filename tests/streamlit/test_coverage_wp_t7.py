"""Coverage tests for work package t7 (Data page and analysis runner)."""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pandas as pd
import pytest

from streamlit_app.components.csv_validation import DateCorrectionNeeded
from streamlit_app.components.upload_guard import GuardedUpload, hash_bytes
from tests.support.dummy_streamlit import DummyStreamlit
from trend_analysis.io.date_correction import DateCorrection


class DummyUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data

    def seek(self, _pos: int) -> None:
        return None


def _reset_stub_recordings(stub: DummyStreamlit) -> None:
    """Drop side effects from ``1_Data`` module-level ``render_data_page()`` on import."""
    stub.write_calls.clear()
    stub.info_messages.clear()
    stub.success_messages.clear()
    stub.error_messages.clear()
    stub.warning_messages.clear()
    stub.captions.clear()
    stub.code_blocks.clear()
    stub.dataframes.clear()
    stub.rerun_called = False


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    stub = SimpleNamespace()
    stub.session_state = {}
    stub.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda fn: fn))
    stub.cache_resource = stub.cache_data
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    from streamlit_app.components import analysis_runner

    importlib.reload(analysis_runner)
    stub.session_state.clear()
    return analysis_runner


@pytest.fixture
def data_page(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, DummyStreamlit]:
    monkeypatch.setenv("TREND_DEMO_PROFILE", "public_llm_demo")
    stub = DummyStreamlit()
    module = ModuleType("streamlit")

    def bind(name: str) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return getattr(stub, name)(*args, **kwargs)

        return wrapper

    for attr in [
        "title",
        "write",
        "radio",
        "selectbox",
        "file_uploader",
        "button",
        "info",
        "success",
        "error",
        "warning",
        "caption",
        "code",
        "dataframe",
        "markdown",
        "expander",
        "container",
        "checkbox",
        "json",
        "cache_data",
        "subheader",
        "columns",
        "data_editor",
        "rerun",
        "metric",
    ]:
        setattr(module, attr, bind(attr))
    module.column_config = stub.column_config
    module.session_state = stub.session_state
    monkeypatch.setitem(sys.modules, "streamlit", module)
    stub.clear_calls = 0  # type: ignore[attr-defined]

    def mark_clear() -> None:
        stub.clear_calls += 1  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "streamlit_app.components.analysis_runner.clear_cached_analysis", mark_clear
    )
    from streamlit_app import state as app_state

    monkeypatch.setattr(app_state, "st", module)

    def fake_guard(uploaded: DummyUpload) -> GuardedUpload:
        data = uploaded.getvalue()
        return GuardedUpload(
            original_name=uploaded.name,
            stored_path=Path("/tmp") / uploaded.name,
            data=data,
            content_hash=hash_bytes(data),
            size=len(data),
        )

    page = importlib.reload(importlib.import_module("streamlit_app.pages.1_Data"))
    monkeypatch.setattr(page, "guard_and_buffer_upload", fake_guard)
    monkeypatch.setattr(page, "hash_path", lambda _p: "samplehash")
    monkeypatch.setattr(page, "validate_uploaded_csv", lambda *a, **k: None)
    _reset_stub_recordings(stub)
    return page, stub


def _idx() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=36, freq="ME")


# --- analysis_runner: coercion / month end / sample split -------------------


def test_sample_split_empty_raises(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="Dataset is empty") as excinfo:
        runner._build_sample_split(pd.DatetimeIndex([]), {})
    assert excinfo.value.args == ("Dataset is empty",)


def test_sample_split_relative_annual_clamps_to_first(runner: ModuleType) -> None:
    result = runner._build_sample_split(_idx(), {})
    assert result == {
        "in_start": "2020-01",
        "in_end": "2021-12",
        "out_start": "2022-01",
        "out_end": "2022-12",
    }


def test_sample_split_relative_quarterly(runner: ModuleType) -> None:
    result = runner._build_sample_split(
        _idx(), {"multi_period_frequency": "Q", "lookback_periods": 4, "evaluation_periods": 2}
    )
    # lookback 12m, evaluation 6m: out 2022-07..2022-12, in 2021-07..2022-06
    assert result == {
        "in_start": "2021-07",
        "in_end": "2022-06",
        "out_start": "2022-07",
        "out_end": "2022-12",
    }


def test_sample_split_relative_short_index_clamps_windows(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner._coerce_positive_int("abc", default=7, minimum=3) == 7
    assert runner._coerce_positive_int(None, default=9, minimum=3) == 9
    assert runner._coerce_positive_int(0, default=99, minimum=4) == 4
    assert runner._coerce_positive_int("4", default=5, minimum=1) == 4
    jan_end = runner._month_end(pd.Timestamp("2020-01-15"))
    assert str(jan_end) == "2020-01-31 23:59:59.999999"
    freq_calls: list[str] = []
    original_to_period = pd.Timestamp.to_period

    def track_to_period(self: pd.Timestamp, freq: str) -> pd.Period:
        freq_calls.append(freq)
        return original_to_period(self, freq)

    monkeypatch.setattr(pd.Timestamp, "to_period", track_to_period)
    short = pd.date_range("2020-01-31", periods=3, freq="ME")
    result = runner._build_sample_split(short, {})
    assert freq_calls
    assert result == {
        "in_start": "2020-01",
        "in_end": "2020-01",
        "out_start": "2020-01",
        "out_end": "2020-03",
    }


def test_sample_split_explicit_exact(runner: ModuleType) -> None:
    result = runner._build_sample_split(
        _idx(),
        {
            "date_mode": "explicit",
            "start_date": "2021-06-15",
            "end_date": "2022-06-20",
            "multi_period_frequency": "A",
            "lookback_periods": 1,
        },
    )
    assert result == {
        "in_start": "2020-06",
        "in_end": "2021-05",
        "out_start": "2021-06",
        "out_end": "2022-06",
    }


def test_sample_split_explicit_clamps_and_in_start_floor(runner: ModuleType) -> None:
    result = runner._build_sample_split(
        _idx(),
        {
            "date_mode": "explicit",
            "start_date": "2019-01-01",
            "end_date": "2025-01-01",
            "multi_period_frequency": "M",
            "lookback_periods": 600,
        },
    )
    assert result == {
        "in_start": "2020-01",
        "in_end": "2020-01",
        "out_start": "2020-01",
        "out_end": "2022-12",
    }


def test_sample_split_explicit_missing_dates_raises(runner: ModuleType) -> None:
    with pytest.raises(
        ValueError,
        match="Explicit date mode requires both start_date and end_date to be specified",
    ):
        runner._build_sample_split(_idx(), {"date_mode": "explicit", "start_date": "2021-01-01"})
    with pytest.raises(
        ValueError,
        match="Explicit date mode requires both start_date and end_date to be specified",
    ):
        runner._build_sample_split(_idx(), {"date_mode": "explicit", "end_date": "2021-01-01"})


def test_sample_split_explicit_bad_date_falls_back_to_relative(runner: ModuleType) -> None:
    result = runner._build_sample_split(
        _idx(), {"date_mode": "explicit", "start_date": "not-a-date", "end_date": "2022-06-20"}
    )
    assert result == {
        "in_start": "2020-01",
        "in_end": "2021-12",
        "out_start": "2022-01",
        "out_end": "2022-12",
    }


# --- analysis_runner: signals / weights / portfolio -------------------------


def test_signals_config_defaults_and_overrides(runner: ModuleType) -> None:
    assert runner._build_signals_config({}) == {
        "kind": "tsmom",
        "window": 63,
        "lag": 1,
        "vol_adjust": False,
        "zscore": False,
    }
    result = runner._build_signals_config(
        {
            "window": 10,
            "lag": 2,
            "min_periods": 4,
            "vol_adjust": True,
            "vol_target": 0.15,
            "zscore": True,
        }
    )
    assert result == {
        "kind": "tsmom",
        "window": 10,
        "lag": 2,
        "vol_adjust": True,
        "zscore": True,
        "min_periods": 4,
        "vol_target": 0.15,
    }


def test_signals_config_min_periods_edges(runner: ModuleType) -> None:
    assert "min_periods" not in runner._build_signals_config({"min_periods": "x"})
    assert "min_periods" not in runner._build_signals_config({"min_periods": 0})
    assert "min_periods" not in runner._build_signals_config({"min_periods": ""})
    assert runner._build_signals_config({"window": 5, "min_periods": 99})["min_periods"] == 5


def test_signals_config_vol_target_edges(runner: ModuleType) -> None:
    assert runner._build_signals_config({"vol_adjust": True, "vol_target": "x"}) == {
        "kind": "tsmom",
        "window": 63,
        "lag": 1,
        "vol_adjust": True,
        "zscore": False,
    }
    assert runner._build_signals_config({"vol_adjust": True, "vol_target": 0}) == {
        "kind": "tsmom",
        "window": 63,
        "lag": 1,
        "vol_adjust": True,
        "zscore": False,
    }
    assert runner._build_signals_config({"vol_adjust": False, "vol_target": 0.2}) == {
        "kind": "tsmom",
        "window": 63,
        "lag": 1,
        "vol_adjust": False,
        "zscore": False,
    }
    assert runner._build_signals_config({"vol_adjust": True, "vol_target": "0.25"}) == {
        "kind": "tsmom",
        "window": 63,
        "lag": 1,
        "vol_adjust": True,
        "zscore": False,
        "vol_target": 0.25,
    }


def test_normalise_metric_weights_filters_and_normalises(runner: ModuleType) -> None:
    result = runner._normalise_metric_weights(
        {"sharpe": 2.0, "return_ann": 1.0, "unknown_metric": 5.0, "vol": "bad", "sortino": 0}
    )
    assert result == {"sharpe": 2.0 / 3.0, "return_ann": 1.0 / 3.0}


def test_normalise_metric_weights_empty_returns_thirds(runner: ModuleType) -> None:
    expected = {"sharpe": 1.0 / 3, "return_ann": 1.0 / 3, "drawdown": 1.0 / 3}
    assert runner._normalise_metric_weights({}) == expected
    assert runner._normalise_metric_weights({"nope": 3.0, "sharpe": -1.0}) == expected


def test_portfolio_config_buy_and_hold_random_and_limits(runner: ModuleType) -> None:
    cfg = runner._build_portfolio_config(
        {
            "selection_approach": "buy_and_hold",
            "buy_hold_initial": "random",
            "bottom_k": 3,
            "min_tenure_n": 2,
            "max_changes_per_period": 4,
            "max_active_positions": 7,
            "selection_count": 5,
        },
        {"sharpe": 1.0},
    )
    assert cfg["selection_mode"] == "buy_and_hold"
    assert cfg["rank"]["bottom_k"] == 3
    assert cfg["min_tenure_n"] == 2
    assert cfg["turnover_budget_max_changes"] == 4
    assert cfg["constraints"]["max_active_positions"] == 7
    assert cfg["rank"]["n"] == 5


def test_portfolio_config_defaults_have_no_optional_keys(runner: ModuleType) -> None:
    cfg = runner._build_portfolio_config({}, {"sharpe": 1.0})
    assert cfg["selection_mode"] == "rank"
    assert "bottom_k" not in cfg["rank"]
    assert "min_tenure_n" not in cfg
    assert "turnover_budget_max_changes" not in cfg
    assert "max_active_positions" not in cfg["constraints"]


# --- analysis_runner: feasibility / csv path / frequency --------------------


def test_assert_config_feasible_raises_exact(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def fake(cfg: dict[str, Any], base_path: Any = None) -> list[Any]:
        seen.append(cfg)
        return [SimpleNamespace(message="too many funds.", suggestion="Reduce n.")]

    monkeypatch.setattr("trend_analysis.config.validation.collect_feasibility_errors", fake)
    cfg = SimpleNamespace(portfolio={"a": 1}, multi_period={}, vol_adjust={}, data={})
    with pytest.raises(
        ValueError,
        match=r"Infeasible configuration; the simulation was not run:\n- too many funds\. Reduce n\.",
    ):
        runner._assert_config_feasible(cfg)
    assert seen == [{"portfolio": {"a": 1}, "multi_period": {}, "vol_adjust": {}, "data": {}}]


def test_assert_config_feasible_ok(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def collect(cfg: dict[str, Any], base_path: Any = None) -> list[Any]:
        seen.append(cfg)
        return []

    monkeypatch.setattr("trend_analysis.config.validation.collect_feasibility_errors", collect)
    cfg = SimpleNamespace(portfolio={"p": 2}, multi_period={"m": 1}, vol_adjust={}, data={"d": 3})
    assert runner._assert_config_feasible(cfg) is None
    assert seen == [
        {"portfolio": {"p": 2}, "multi_period": {"m": 1}, "vol_adjust": {}, "data": {"d": 3}}
    ]


def test_ensure_validation_csv_path_existing_csv(runner: ModuleType, tmp_path: Path) -> None:
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    runner.st.session_state["data_saved_path"] = str(csv_file)
    assert runner._ensure_validation_csv_path(pd.DataFrame({"A": [0.1]})) == str(csv_file)


def test_ensure_validation_csv_path_exception_returns_none(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.st.session_state.clear()
    calls: list[pd.DataFrame] = []

    def boom(frame: pd.DataFrame) -> str:
        calls.append(frame)
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "cache_key_for_frame", boom)
    df = pd.DataFrame({"A": [0.1]}, index=pd.date_range("2020-01-31", periods=1, freq="ME"))
    assert runner._ensure_validation_csv_path(df) is None
    assert len(calls) == 1
    assert calls[0].iloc[0, 0] == 0.1


def test_resolve_frequency_from_session_meta(runner: ModuleType) -> None:
    runner.st.session_state["schema_meta"] = {"frequency_code": " m "}
    df = pd.DataFrame({"A": [0.1]}, index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    assert runner._resolve_frequency(df) == "M"
    runner.st.session_state["schema_meta"] = {"frequency": "q"}
    assert runner._resolve_frequency(df) == "Q"


# --- analysis_runner: variants / cached / hash ------------------------------


def test_hashable_model_state_sorted(runner: ModuleType) -> None:
    assert runner._hashable_model_state({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


def test_apply_variant_patch_validation_wrap(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trend_analysis.config.patch import ConfigPatch

    monkeypatch.setattr(
        runner,
        "_validate_streamlit_payload",
        lambda _p: (_ for _ in ()).throw(ValueError("badcfg")),
    )
    returns = pd.DataFrame({"FundA": [0.01]}, index=pd.to_datetime(["2020-01-31"]))
    with pytest.raises(runner.VariantRunError) as exc:
        runner.apply_variant_patch(
            {"lookback_periods": 3},
            ConfigPatch(operations=[], summary="x"),
            returns=returns,
            benchmark=None,
            label="lab",
        )
    assert exc.value.label == "lab"
    assert exc.value.stage == "config validation"
    assert exc.value.detail == "badcfg"


def test_run_variant_analysis_count_and_duplicate_labels(runner: ModuleType) -> None:
    from trend_analysis.config.patch import ConfigPatch

    returns = pd.DataFrame({"FundA": [0.01]}, index=pd.to_datetime(["2020-01-31"]))
    one = [SimpleNamespace(label="a", patch=ConfigPatch(operations=[], summary="a"))]
    with pytest.raises(ValueError, match="Expected exactly three variants for analysis."):
        runner.run_variant_analysis(returns, {}, None, one)
    dup = [
        SimpleNamespace(label="Base", patch=ConfigPatch(operations=[], summary="1")),
        SimpleNamespace(label="base", patch=ConfigPatch(operations=[], summary="2")),
        SimpleNamespace(label="other", patch=ConfigPatch(operations=[], summary="3")),
    ]
    with pytest.raises(ValueError, match="Variant labels must be unique."):
        runner.run_variant_analysis(returns, {}, None, dup)


def test_run_variant_analysis_progress_and_run_failure(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trend_analysis.config.patch import ConfigPatch

    returns = pd.DataFrame({"FundA": [0.01]}, index=pd.to_datetime(["2020-01-31"]))
    variants = [
        SimpleNamespace(label=n, patch=ConfigPatch(operations=[], summary=n))
        for n in ("a", "b", "c")
    ]
    monkeypatch.setattr(runner, "apply_variant_patch", lambda *a, **k: {"patched": True})
    monkeypatch.setattr(runner, "run_analysis", lambda *a, **k: {"ok": True})
    seen: list[tuple[str, int, int]] = []
    results = runner.run_variant_analysis(
        returns, {}, None, variants, progress=lambda label, i, t: seen.append((label, i, t))
    )
    assert seen == [("a", 1, 3), ("b", 2, 3), ("c", 3, 3)]
    assert list(results.keys()) == ["a", "b", "c"]
    monkeypatch.setattr(
        runner, "run_analysis", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom"))
    )
    with pytest.raises(runner.VariantRunError) as exc:
        runner.run_variant_analysis(returns, {}, None, variants)
    assert exc.value.stage == "analysis run"
    assert exc.value.detail == "kaboom"
    assert exc.value.label == "a"


def test_run_cached_and_run_analysis(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_exec(payload: Any) -> str:
        captured["payload"] = payload
        return "EXEC"

    monkeypatch.setattr(runner, "_execute_analysis", fake_exec)
    df = pd.DataFrame({"A": [0.1]}, index=pd.date_range("2020-01-31", periods=1, freq="ME"))
    assert runner.run_cached_analysis(df, '{"x": 2}', "BM", "h1", "fp") == "EXEC"
    assert captured["payload"].model_state == {"x": 2}
    assert captured["payload"].benchmark == "BM"
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(runner, "run_cached_analysis", lambda *a: calls.append(a) or "CACHED")
    monkeypatch.setattr(runner, "cache_key_for_frame", lambda _df: "hashX")
    monkeypatch.setattr(runner, "membership_cache_fingerprint", lambda: "mfp")
    assert runner.run_analysis(df, {"b": 1}, None) == "CACHED"
    assert calls[0][1] == '{"b": 1}'
    assert calls[0][3] == "hashX"
    assert calls[0][4] == "mfp"
    calls.clear()
    assert runner.run_analysis(df, {"b": 1}, "BM", data_hash="given") == "CACHED"
    assert calls[0][3] == "given"


def test_mask_returns_indexed_branch(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis.universe as uni

    sentinel = pd.DataFrame({"A": [9.9]}, index=pd.to_datetime(["2020-01-31"]))
    loaded: list[str] = []

    def load_membership(path: str) -> str:
        loaded.append(path)
        return "MEMB"

    monkeypatch.setattr(uni, "load_universe_membership", load_membership)
    seen: dict[str, Any] = {}

    def fake_apply(df: pd.DataFrame, memb: Any) -> pd.DataFrame:
        seen["df"] = df
        seen["memb"] = memb
        return sentinel

    monkeypatch.setattr(uni, "apply_membership_windows", fake_apply)
    returns = pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2020-01-31"]))
    result = runner._mask_returns_for_membership(returns, "m.csv")
    assert result is sentinel
    assert loaded == ["m.csv"]
    assert seen["memb"] == "MEMB"
    assert float(seen["df"].iloc[0, 0]) == 0.1
    assert list(seen["df"].columns) == ["A"]
    dated = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-31"]), "A": [0.2]},
    )
    indexed_sentinel = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-31"]), "A": [8.8]},
    )
    seen.clear()
    loaded.clear()

    def indexed_apply(df: pd.DataFrame, memb: Any) -> pd.DataFrame:
        seen["indexed_df"] = df
        seen["indexed_memb"] = memb
        return indexed_sentinel.set_index("Date")

    monkeypatch.setattr(uni, "apply_membership_windows", indexed_apply)
    indexed_result = runner._mask_returns_for_membership(dated, "other.csv")
    assert loaded == ["other.csv"]
    assert list(indexed_result.columns) == ["Date", "A"]
    assert float(indexed_result.iloc[0, 1]) == 8.8
    assert seen["indexed_memb"] == "MEMB"
    assert list(seen["indexed_df"].columns) == ["A"]


# --- Data page helpers -------------------------------------------------------


def _correction(**kw: Any) -> DateCorrectionNeeded:
    base: dict[str, Any] = {
        "corrections": [],
        "unfixable": [],
        "trailing_empty_rows": [],
        "droppable_empty_rows": [],
        "raw_data": b"Date,FundA\n2020-01-31,0.01\n",
        "original_name": "f.csv",
        "date_column": "Date",
    }
    base.update(kw)
    return DateCorrectionNeeded(**base)


def test_fund_table_state_key_and_ensure_branches(data_page: Any) -> None:
    page, stub = data_page
    assert page._fund_table_state_key("ds1") == "fund_table::ds1"
    key = page._ensure_fund_table_state(
        data_key="ds1", available_funds=["A", "B"], current_selection=["B"]
    )
    assert key == "fund_table::ds1"
    df = stub.session_state[key]
    assert df["Fund Name"].tolist() == ["A", "B"]
    assert df["Include"].tolist() == [False, True]
    # same funds: normalised copy returned unchanged in content
    stub.session_state[key] = pd.DataFrame({"Include": [True, False], "Fund Name": ["A", "B"]})
    page._ensure_fund_table_state(data_key="ds1", available_funds=["A", "B"], current_selection=[])
    assert stub.session_state[key]["Include"].tolist() == [True, False]
    # changed funds: remap by name, new funds default False
    page._ensure_fund_table_state(
        data_key="ds1", available_funds=["B", "C"], current_selection=["C"]
    )
    df2 = stub.session_state[key]
    assert df2["Fund Name"].tolist() == ["B", "C"]
    assert df2["Include"].tolist() == [False, False]


def test_render_validation_branches(data_page: Any) -> None:
    page, stub = data_page
    _reset_stub_recordings(stub)
    page._render_validation({"validation": None})
    assert stub.error_messages == [] and stub.warning_messages == []
    page._render_validation({"validation": {"issues": ["bad1"], "warnings": ["w1"]}})
    assert stub.error_messages == ["Data quality checks flagged issues:"]
    assert stub.warning_messages == ["Warnings:"]
    assert stub.write_calls == ["• bad1", "• w1"]


def test_store_dataset_no_benchmark_candidate(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    monkeypatch.setattr(page, "infer_benchmarks", lambda cols: [])
    monkeypatch.setattr(page, "infer_risk_free_candidates", lambda cols: [])
    df = pd.DataFrame({"FundA": [0.01]}, index=pd.date_range("2020-01-31", periods=1, freq="ME"))
    stub.session_state["selected_benchmark"] = "OLD"
    stub.session_state["selected_risk_free"] = "OLD"
    page._store_dataset(df, {"validation": {}}, "k", data_hash="h")
    assert stub.session_state["selected_benchmark"] is None
    assert stub.session_state["selected_risk_free"] is None
    assert stub.session_state["benchmark_candidates"] == []
    assert stub.session_state["risk_free_candidates"] == []
    assert stub.session_state["data_loaded_key"] == "k"
    assert stub.session_state["data_fingerprint"] == "h"
    assert stub.session_state["selected_fund_columns"] == ["FundA"]


def test_handle_failure_variants(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    _reset_stub_recordings(stub)
    page._handle_failure(page.UploadViolation("too big"))
    assert stub.error_messages[-1] == "too big"
    stub.error_messages.clear()
    stub.write_calls.clear()
    err = page.CSVValidationError("csv bad", issues=["i1"], sample_preview="PREV")
    page._handle_failure(err)
    assert stub.error_messages[-1] == "csv bad"
    assert stub.write_calls == ["• i1"]
    assert stub.code_blocks == [("PREV", "text")]
    stub.error_messages.clear()
    page._handle_failure(page.InputValidationError("inp bad", issues=["j1"]))
    assert stub.error_messages[-1] == "inp bad"
    assert stub.session_state["validation_report"]["issues"] == ["j1"]
    page._handle_failure(page.MarketDataValidationError("mkt bad", issues=["k1"]))
    assert stub.error_messages[-1] == "mkt bad"
    stub.error_messages.clear()
    stub.write_calls.clear()
    stub.captions.clear()
    page._handle_failure(RuntimeError("   "))
    assert stub.error_messages[-1] == (
        "We couldn't process the file. Please confirm the format and try again."
    )
    assert stub.captions == []
    page._handle_failure(RuntimeError("detail only"))
    assert stub.captions == ["detail only"]
    stub.error_messages.clear()
    stub.captions.clear()
    stub.code_blocks.clear()
    err_preview = page.CSVValidationError("csv prev", issues=[], sample_preview="ROW1,ROW2")
    page._handle_failure(err_preview)
    assert stub.code_blocks == [("ROW1,ROW2", "text")]
    assert stub.captions[-2:] == ["Example format:"]


def test_handle_failure_date_correction_delegates(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    _reset_stub_recordings(stub)
    seen: list[Any] = []
    monkeypatch.setattr(page, "_render_date_correction_ui", lambda m, c: seen.append((m, c)))
    corr = _correction()
    page._handle_failure(page.CSVValidationError("needs fix", issues=["x"], date_correction=corr))
    assert seen == [("needs fix", corr)]
    assert stub.error_messages == []


def test_render_date_correction_ui_apply_and_cancel(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    corr = _correction(
        corrections=[DateCorrection(0, "2020-11-31", "2020-11-30", "clamp")],
        unfixable=[(i, f"bad{i}") for i in range(7)],
        trailing_empty_rows=[3],
    )
    applied: list[bool] = []
    monkeypatch.setattr(page, "_apply_pending_date_corrections", lambda: applied.append(True))
    stub.button_responses = [True, False]
    page._render_date_correction_ui("fix dates", corr)
    assert stub.warning_messages == ["fix dates"]
    assert applied == [True]
    assert stub.session_state["pending_date_correction"] is corr
    assert "• Row 1: `bad0`" in stub.write_calls
    assert "• ... and 2 more" in stub.write_calls
    stub.button_responses = [False, True]
    page._render_date_correction_ui("fix dates", corr)
    assert "pending_date_correction" not in stub.session_state
    assert stub.info_messages[-1] == "Upload cancelled. Please fix the dates and try again."


def test_apply_corrections_no_pending(data_page: Any) -> None:
    page, stub = data_page
    stub.session_state.pop("pending_date_correction", None)
    page._apply_pending_date_corrections()
    assert stub.error_messages == ["No pending corrections found."]


def test_apply_corrections_csv_success(data_page: Any, monkeypatch: Any, tmp_path: Path) -> None:
    page, stub = data_page
    corr = _correction(
        corrections=[DateCorrection(0, "2020-11-31", "2020-11-30", "clamp")],
        droppable_empty_rows=[2],
    )
    stub.session_state["pending_date_correction"] = corr
    monkeypatch.setattr(page, "validate_uploaded_csv", lambda *a, **k: None)
    df_final = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2020-01-31", periods=1, freq="ME")
    )
    monkeypatch.setattr(
        page.data_cache, "load_dataset_from_bytes", lambda d, n: (df_final, {"validation": {}})
    )
    guarded = GuardedUpload("f.csv", tmp_path / "f.csv", b"x", "abcdef0123456789zz", 1)
    monkeypatch.setattr(
        "streamlit_app.components.upload_guard.store_buffered_upload", lambda d, n: guarded
    )
    stored: list[Any] = []
    monkeypatch.setattr(page, "_store_dataset", lambda *a, **k: stored.append((a, k)))
    page._apply_pending_date_corrections()
    assert stored and stored[0][0][2] == "corrected::f.csv::abcdef0123456789"
    assert stub.session_state["uploaded_file_path"] == str(tmp_path / "f.csv")
    assert "pending_date_correction" not in stub.session_state
    assert stub.session_state["date_correction_success"] == (
        "✅ Applied 1 date correction(s) and 1 row(s) with empty dates removed. Loaded f.csv."
    )
    assert stub.rerun_called is True


def test_apply_corrections_revalidation_branches(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    corr = _correction()
    stub.session_state["pending_date_correction"] = corr

    def raise_new(*a: Any, **k: Any) -> None:
        raise page.CSVValidationError("other error", issues=["z"])

    monkeypatch.setattr(page, "validate_uploaded_csv", raise_new)
    failures: list[Any] = []
    monkeypatch.setattr(page, "_handle_failure", lambda e: failures.append(e))
    page._apply_pending_date_corrections()
    assert len(failures) == 1 and failures[0].user_message == "other error"
    assert "pending_date_correction" not in stub.session_state
    # still-has-issues branch
    stub.session_state["pending_date_correction"] = corr
    new_corr = _correction()

    def raise_again(*a: Any, **k: Any) -> None:
        raise page.CSVValidationError("again", issues=["y"], date_correction=new_corr)

    monkeypatch.setattr(page, "validate_uploaded_csv", raise_again)
    page._apply_pending_date_corrections()
    assert stub.session_state["pending_date_correction"] is new_corr
    assert stub.warning_messages[-1] == "Some dates still have issues after correction."
    assert stub.rerun_called is True


def test_apply_corrections_generic_failure(data_page: Any) -> None:
    page, stub = data_page
    stub.session_state["pending_date_correction"] = _correction(raw_data=b"")
    page._apply_pending_date_corrections()
    assert "pending_date_correction" not in stub.session_state
    assert stub.error_messages and stub.error_messages[0].startswith("Failed to apply corrections:")


def test_load_uploaded_dataset_branches(data_page: Any, monkeypatch: Any, tmp_path: Path) -> None:
    page, stub = data_page
    _reset_stub_recordings(stub)
    monkeypatch.setattr(
        page,
        "guard_and_buffer_upload",
        lambda u: (_ for _ in ()).throw(page.UploadViolation("guard nope")),
    )
    page._load_uploaded_dataset(DummyUpload("f.csv", b"x"))
    assert stub.error_messages[-1] == "guard nope"
    monkeypatch.setattr(
        page,
        "guard_and_buffer_upload",
        lambda u: GuardedUpload("f.csv", tmp_path / "f.csv", b"d", "h123", 1),
    )
    monkeypatch.setattr(
        page,
        "validate_uploaded_csv",
        lambda *a, **k: (_ for _ in ()).throw(
            page.CSVValidationError("invalid csv", issues=["bad col"])
        ),
    )
    page._load_uploaded_dataset(DummyUpload("f.csv", b"x"))
    assert stub.error_messages[-1] == "invalid csv"
    monkeypatch.setattr(page, "validate_uploaded_csv", lambda *a, **k: None)
    monkeypatch.setattr(
        page.data_cache,
        "load_dataset_from_bytes",
        lambda d, n: (_ for _ in ()).throw(ValueError("parse boom")),
    )
    page._load_uploaded_dataset(DummyUpload("f.csv", b"x"))
    assert stub.error_messages[-1] == (
        "We couldn't process the file. Please confirm the format and try again."
    )
    df = pd.DataFrame(
        {"FundA": [0.01, 0.02]}, index=pd.date_range("2020-01-31", periods=2, freq="ME")
    )
    monkeypatch.setattr(
        page.data_cache, "load_dataset_from_bytes", lambda d, n: (df, {"validation": {}})
    )
    page._load_uploaded_dataset(DummyUpload("f.csv", b"x"))
    assert stub.success_messages[-1] == "Loaded f.csv."
    assert stub.session_state["uploaded_file_path"] == str(tmp_path / "f.csv")
    assert stub.dataframes and stub.dataframes[-1].shape == (2, 1)


def test_maybe_autoload_sample_valid_upload(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    stub.session_state.clear()
    checks: list[str] = []

    def has_upload() -> bool:
        checks.append("checked")
        return True

    monkeypatch.setattr("streamlit_app.state.has_valid_upload", has_upload)
    monkeypatch.setattr(
        page.data_cache,
        "default_sample_dataset",
        lambda: (_ for _ in ()).throw(AssertionError("must not load sample")),
    )
    page._maybe_autoload_sample()
    assert checks == ["checked"]
    assert stub.session_state["_data_loaded_once"] is True
    assert stub.session_state.get("returns_df") is None


def test_universe_membership_controls(data_page: Any, monkeypatch: Any, tmp_path: Path) -> None:
    page, stub = data_page
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    demo_file = demo_dir / "demo_universe_membership.csv"
    demo_file.write_text("fund,effective_date,end_date\nA,2020-01-01,\n", encoding="utf-8")
    monkeypatch.setattr(page, "proj_path", lambda: tmp_path)
    events: list[str] = []
    monkeypatch.setattr(page, "validate_membership_file", lambda p: events.append("validate"))
    monkeypatch.setattr(
        page, "clear_active_universe_membership", lambda s: events.append("clear") or True
    )
    monkeypatch.setattr(
        page, "set_active_universe_membership", lambda p, s, ss: events.append("set") or True
    )
    monkeypatch.setattr(
        page, "summarise_membership", lambda p: {"fund_count": 2, "exited_fund_count": 1}
    )
    monkeypatch.setattr(
        page, "invalidate_analysis_for_membership_change", lambda: events.append("inv")
    )
    monkeypatch.setattr(page, "resolve_universe_membership_path", lambda: str(demo_file))
    stub.button_responses = [True, False]
    page._render_universe_membership_controls()
    assert events == ["validate", "clear", "set", "inv"]
    assert stub.info_messages[-1] == (
        "Membership active for 2 fund(s); 1 fund(s) have exit windows."
    )
    # clear button branch
    events.clear()
    stub.button_responses = [False, True]
    page._render_universe_membership_controls()
    assert events == ["clear", "inv"]
    # upload success + rejection
    monkeypatch.setattr(
        page, "persist_membership_upload", lambda b, upload_dir, filename: str(tmp_path / "m.csv")
    )
    stub.uploaded = DummyUpload("m.csv", b"a,b\n1,2\n")
    stub.button_responses = [False, False]
    page._render_universe_membership_controls()
    assert stub.success_messages[-1] == "Membership file validated and ready."
    monkeypatch.setattr(
        page,
        "persist_membership_upload",
        lambda b, upload_dir, filename: (_ for _ in ()).throw(ValueError("bad memb")),
    )
    stub.session_state.pop(page.UNIVERSE_MEMBERSHIP_UPLOAD_DIGEST_KEY, None)
    stub.uploaded = DummyUpload("m.csv", b"different-bytes")
    stub.button_responses = [False, False]
    page._render_universe_membership_controls()
    assert stub.error_messages[-1] == "Membership file rejected: bad memb"


def _seed_loaded_sample(page: Any, stub: Any, monkeypatch: Any) -> None:
    stub.session_state.clear()
    df = pd.DataFrame(
        {"FundA": [0.01, 0.02], "FundB": [0.03, 0.04], "SPX Index": [0.05, 0.06]},
        index=pd.date_range("2024-01-31", periods=2, freq="ME"),
    )
    sample = page.data_cache.SampleDataset("demo.csv", Path("demo/demo_returns.csv"))
    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: None)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {sample.label: sample})
    monkeypatch.setattr(
        page.data_cache,
        "load_dataset_from_path",
        lambda _p: (df, {"validation": {"issues": [], "warnings": []}}),
    )
    monkeypatch.setattr(page, "infer_benchmarks", lambda cols: ["SPX Index"])
    monkeypatch.setattr(page, "_render_universe_membership_controls", lambda: None)
    stub.selectbox_map["Choose a sample"] = sample.label
    page.render_data_page()


def test_render_pending_correction_and_success_msg(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: None)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {})
    corr = _correction()
    stub.session_state["pending_date_correction"] = corr
    seen: list[Any] = []
    monkeypatch.setattr(page, "_render_date_correction_ui", lambda m, c: seen.append((m, c)))
    page.render_data_page()
    assert seen == [("Some dates have issues that can be automatically corrected.", corr)]
    assert "A file upload needs date corrections before it can be processed." in stub.write_calls


def test_render_fund_buttons_and_sample_fallback(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    stub.session_state["selected_sample_label"] = "MISSING_LABEL"
    _seed_loaded_sample(page, stub, monkeypatch)
    assert stub.session_state["selected_fund_columns"] == ["FundA", "FundB"]
    # Select All
    stub.button_responses = [True, False, False, False, False]
    page.render_data_page()
    assert stub.session_state["selected_fund_columns"] == ["FundA", "FundB"]
    assert stub.rerun_called is True
    # Clear All
    stub.button_responses = [False, True, False, False, False]
    page.render_data_page()
    assert stub.session_state["fund_columns"] == []
    # Invert (from cleared state, include keys are False -> inverted selects all)
    stub.button_responses = [False, False, True, False, False]
    page.render_data_page()
    assert stub.session_state["selected_fund_columns"] == ["FundA", "FundB"]
    # Include range / Exclude range
    stub.button_responses = [False, False, False, True, False]
    page.render_data_page()
    stub.button_responses = [False, False, False, False, True]
    page.render_data_page()
    assert stub.rerun_called is True


def test_render_fund_init_variants_and_callbacks(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    _seed_loaded_sample(page, stub, monkeypatch)
    # restored-selection branch: drop selected, keep fund_columns list
    data_key = stub.session_state["data_loaded_key"]
    include_prefix = f"fund_include::{data_key}::"
    stub.session_state.pop("selected_fund_columns", None)
    stub.session_state["fund_columns"] = ["FundB", "NOPE"]
    stub.session_state[f"{include_prefix}FundA"] = False
    stub.session_state[f"{include_prefix}FundB"] = True
    page.render_data_page()
    assert stub.session_state["selected_fund_columns"] == ["FundB"]
    # fund_columns sync elif branch + analysis clear on change
    stub.clear_calls = 0  # type: ignore[attr-defined]
    stub.session_state["analysis_fund_columns"] = ["OLD"]
    page.render_data_page()
    assert stub.session_state["analysis_fund_columns"] == ["FundB"]
    # debug history non-list guard
    stub.session_state["show_perf_diagnostics"] = True
    stub.session_state["_debug_fund_history"] = "notalist"
    page.render_data_page()
    assert isinstance(stub.session_state["_debug_fund_history"], list)
    assert len(stub.session_state["_debug_fund_history"]) == 1
    # on_change callbacks captured via selectbox kwargs
    callbacks: dict[str, Any] = {}
    orig_selectbox = stub.selectbox

    def capturing(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("on_change") is not None:
            callbacks[kwargs.get("key", "")] = kwargs["on_change"]
        return orig_selectbox(*args, **kwargs)

    monkeypatch.setattr(page.st, "selectbox", capturing)
    page.render_data_page()
    stub.session_state["_widget_rf"] = "(None)"
    callbacks["_widget_rf"]()
    assert stub.session_state["selected_risk_free"] is None
    stub.session_state["_widget_rf"] = "FundA"
    callbacks["_widget_rf"]()
    assert stub.session_state["selected_risk_free"] == "FundA"
    stub.session_state["_widget_bench"] = "(None)"
    callbacks["_widget_bench"]()
    assert stub.session_state["selected_benchmark"] is None
    stub.session_state["_widget_bench"] = "SPX Index"
    callbacks["_widget_bench"]()
    assert stub.session_state["selected_benchmark"] == "SPX Index"


def test_apply_corrections_excel_roundtrip(
    data_page: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    page, stub = data_page
    raw_df = pd.DataFrame({"Date": ["2020-11-31"], "FundA": [0.01]})
    raw_buf = io.BytesIO()
    raw_df.to_excel(raw_buf, index=False)
    corr = _correction(
        corrections=[DateCorrection(0, "2020-11-31", "2020-11-30", "clamp")],
        raw_data=raw_buf.getvalue(),
        original_name="upload.xlsx",
    )
    stub.session_state["pending_date_correction"] = corr
    monkeypatch.setattr(page, "validate_uploaded_csv", lambda *a, **k: None)
    df_final = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2020-01-31", periods=1, freq="ME")
    )
    monkeypatch.setattr(
        page.data_cache, "load_dataset_from_bytes", lambda d, n: (df_final, {"validation": {}})
    )
    guarded = GuardedUpload("upload.xlsx", tmp_path / "upload.xlsx", b"x", "excelhash0123456", 1)
    monkeypatch.setattr(
        "streamlit_app.components.upload_guard.store_buffered_upload", lambda d, n: guarded
    )
    stored: list[Any] = []
    monkeypatch.setattr(page, "_store_dataset", lambda *a, **k: stored.append((a, k)))
    page._apply_pending_date_corrections()
    assert stored and stored[0][0][2].startswith("corrected::upload.xlsx::")
    assert "pending_date_correction" not in stub.session_state


def test_render_correction_success_and_sample_label_fallback(
    data_page: Any, monkeypatch: Any
) -> None:
    page, stub = data_page
    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: None)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {})
    stub.session_state["date_correction_success"] = "✅ Dates corrected."
    page.render_data_page()
    assert stub.success_messages == ["✅ Dates corrected."]
    df = pd.DataFrame(
        {"FundA": [0.01], "SPX Index": [0.02]},
        index=pd.date_range("2024-01-31", periods=1, freq="ME"),
    )
    sample = page.data_cache.SampleDataset("demo.csv", Path("demo/demo_returns.csv"))
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {sample.label: sample})
    monkeypatch.setattr(
        page.data_cache,
        "load_dataset_from_path",
        lambda _p: (df, {"validation": {}}),
    )
    monkeypatch.setattr(page, "_render_universe_membership_controls", lambda: None)
    stub.session_state.clear()
    stub.session_state["data_source"] = "Sample dataset"
    stub.session_state["selected_sample_label"] = "__missing_label__"
    stub.session_state["data_loaded_key"] = f"sample::{sample.path.resolve()}"
    page.app_state.store_validated_data(df, {"validation": {}}, data_hash="h1")
    stub.session_state["selected_benchmark"] = "__not_a_benchmark__"
    stub.selectbox_map["Choose a sample"] = sample.label
    page.render_data_page()
    assert stub.session_state["selected_sample_label"] == sample.label


def test_render_fund_defaults_when_selection_empty(data_page: Any, monkeypatch: Any) -> None:
    page, stub = data_page
    _seed_loaded_sample(page, stub, monkeypatch)
    data_key = stub.session_state["data_loaded_key"]
    init_key = f"fund_selection_initialized::{data_key}"
    include_prefix = f"fund_include::{data_key}::"
    stub.session_state.pop("selected_fund_columns", None)
    stub.session_state["fund_columns"] = []
    stub.session_state.pop(init_key, None)
    for key in list(stub.session_state.keys()):
        if str(key).startswith(include_prefix):
            stub.session_state.pop(key, None)
    page.render_data_page()
    assert stub.session_state["selected_fund_columns"] == ["FundA", "FundB"]
    assert stub.session_state[init_key] is True
    stub.session_state["selected_benchmark"] = None
    stub.session_state.pop("selected_fund_columns", None)
    stub.session_state.pop("fund_columns", None)
    stub.session_state.pop(init_key, None)
    page.render_data_page()
    assert stub.session_state["selected_fund_columns"] == ["FundA", "FundB", "SPX Index"]
    assert stub.session_state["fund_columns"] == ["FundA", "FundB", "SPX Index"]
