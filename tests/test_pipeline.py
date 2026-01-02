import logging

import numpy as np
import pandas as pd
import pytest

from trend_analysis import config, pipeline
from trend_analysis.config import Config
from trend_analysis.core.rank_selection import RiskStatsConfig, canonical_metric_list
from trend_analysis.diagnostics import PipelineReasonCode

pytestmark = pytest.mark.runtime


pytestmark = pytest.mark.runtime


def make_cfg(tmp_path, df):
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg_dict = {
        "version": "1",
        "data": {
            "csv_path": str(csv),
            "date_column": "Date",
            "frequency": "M",
            "risk_free_column": "RF",
            "allow_risk_free_fallback": False,
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 1.0},
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-06",
        },
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }
    return Config(**cfg_dict)


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": [0.02, 0.01, 0.03, 0.04, 0.02, 0.01],
        }
    )


RUN_ANALYSIS_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


def test_run_returns_dataframe(tmp_path):
    cfg = make_cfg(tmp_path, make_df())
    out = pipeline.run(cfg)
    assert not out.empty
    columns = set(out.columns)
    expected = {
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "information_ratio",
        "max_drawdown",
    }
    assert expected.issubset(columns)
    # Some pipeline configurations now emit in/out-of-sample average correlation
    # diagnostics.  If present they should be the only additional columns.
    extra = columns - expected
    assert extra <= {"is_avg_corr", "os_avg_corr"}


def test_run_with_benchmarks(tmp_path):
    df = make_df()
    df["SPX"] = 0.01
    cfg = make_cfg(tmp_path, df)
    cfg.benchmarks = {"spx": "SPX"}
    out = pipeline.run(cfg)
    assert "ir_spx" in out.columns


def test_run_errors_when_risk_free_column_missing(tmp_path):
    cfg = make_cfg(tmp_path, make_df())
    cfg.data["risk_free_column"] = "MissingRF"
    cfg.data["allow_risk_free_fallback"] = False
    with pytest.raises(ValueError, match="Configured risk-free column 'MissingRF'"):
        pipeline.run(cfg)


def test_risk_free_series_used_for_score_frame():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    rf = pd.Series([0.001, 0.001, 0.002, 0.002], index=dates)
    a = pd.Series([0.011, 0.012, 0.01, 0.009], index=dates)
    df = pd.DataFrame({"Date": dates, "RF": rf, "A": a, "B": 0.0})
    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        1.0,
        0.0,
        risk_free_column="RF",
        allow_risk_free_fallback=False,
    )

    assert res is not None
    expected_sharpe = pipeline.sharpe_ratio(
        a.iloc[:2], risk_free=rf.iloc[:2], periods_per_year=12
    )
    assert res["score_frame"].loc["A", "Sharpe"] == pytest.approx(expected_sharpe)


def test_risk_free_fallback_logs_choice(caplog):
    caplog.set_level(logging.INFO, logger="trend_analysis.pipeline")
    df = _make_two_fund_df()

    res = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        allow_risk_free_fallback=True,
        risk_free_column=None,
    )

    assert res is not None
    assert "fallback enabled" in caplog.text


def test_run_returns_empty_when_no_funds(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=2, freq="ME"), "RF": 0.0}
    )
    cfg = make_cfg(tmp_path, df)
    result = pipeline.run(cfg)
    assert result.empty
    diagnostic = result.attrs.get("diagnostic")
    assert diagnostic is not None
    assert diagnostic.reason_code == PipelineReasonCode.SAMPLE_WINDOW_EMPTY.value


def test_run_file_missing(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, make_df())
    cfg.data["csv_path"] = str(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        pipeline.run(cfg)


def test_env_override(tmp_path, monkeypatch):
    df = make_df()
    cfg = make_cfg(tmp_path, df)
    cfg_yaml = tmp_path / "c.yml"
    cfg_yaml.write_text(cfg.model_dump_json())
    monkeypatch.setenv("TREND_CFG", str(cfg_yaml))
    loaded_env = config.load()
    assert loaded_env.version == cfg.version
    monkeypatch.delenv("TREND_CFG", raising=False)


def test_run_analysis_none():
    res = pipeline.run_analysis(
        None, "2020-01", "2020-03", "2020-04", "2020-06", 1.0, 0.0
    )
    assert res.unwrap() is None


def test_run_analysis_missing_date():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(ValueError):
        pipeline.run_analysis(df, "2020-01", "2020-03", "2020-04", "2020-06", 1.0, 0.0)


def test_run_analysis_string_dates():
    df = make_df()
    df["Date"] = df["Date"].astype(str)
    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        **RUN_ANALYSIS_KWARGS,
    )
    assert res is not None


def test_run_analysis_no_funds():
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=3, freq="ME"), "RF": 0.0}
    )
    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-03",
        1.0,
        0.0,
        **RUN_ANALYSIS_KWARGS,
    )
    assert res.unwrap() is None


def test_run_analysis_returns_none_when_window_missing():
    df = make_df()

    # In-sample window entirely before available data.
    res_in_empty = pipeline.run_analysis(
        df,
        "2019-01",
        "2019-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        **RUN_ANALYSIS_KWARGS,
    )
    assert res_in_empty.unwrap() is None

    # Out-of-sample window after available data.
    res_out_empty = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2021-01",
        "2021-03",
        1.0,
        0.0,
        **RUN_ANALYSIS_KWARGS,
    )
    assert res_out_empty.unwrap() is None


def test_run_missing_csv_key(tmp_path):
    cfg = Config(
        version="1",
        data={},
        preprocessing={},
        vol_adjust={},
        sample_split={},
        portfolio={},
        metrics={},
        export={},
        run={},
    )
    with pytest.raises(KeyError):
        pipeline.run(cfg)


def test_run_analysis_custom_weights():
    df = make_df()
    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        custom_weights={"A": 100},
        **RUN_ANALYSIS_KWARGS,
    )
    assert res["fund_weights"]["A"] == 1.0


def test_run_full_includes_risk_diagnostics(tmp_path):
    cfg = make_cfg(tmp_path, _make_two_fund_df())
    res = pipeline.run_full(cfg)
    assert "risk_diagnostics" in res
    diag = res["risk_diagnostics"]
    assert "asset_volatility" in diag
    assert "turnover_value" in diag
    asset_vol = diag["asset_volatility"]
    assert isinstance(asset_vol, pd.DataFrame)
    assert not asset_vol.empty


def test_run_full_robustness_settings_affect_weights(tmp_path):
    cfg = make_cfg(tmp_path, _make_three_fund_df())
    cfg.portfolio["weighting_scheme"] = "robust_mv"
    cfg.portfolio["robustness"] = {
        "shrinkage": {"enabled": False},
        "condition_check": {
            "enabled": True,
            "threshold": 1.0,
            "safe_mode": "risk_parity",
            "diagonal_loading_factor": 1.0e-6,
        },
    }

    res_rp = pipeline.run_full(cfg)
    diag_rp = res_rp["weight_engine_diagnostics"]
    assert diag_rp["used_safe_mode"] is True
    assert diag_rp["safe_mode"] == "risk_parity"
    weights_rp = res_rp["fund_weights"]

    cfg.portfolio["robustness"]["condition_check"]["safe_mode"] = "diagonal_mv"
    res_diag = pipeline.run_full(cfg)
    diag_diag = res_diag["weight_engine_diagnostics"]
    assert diag_diag["used_safe_mode"] is True
    assert diag_diag["safe_mode"] == "diagonal_mv"
    weights_diag = res_diag["fund_weights"]

    rp_values = np.array([weights_rp["A"], weights_rp["B"], weights_rp["C"]])
    diag_values = np.array([weights_diag["A"], weights_diag["B"], weights_diag["C"]])
    assert not np.allclose(rp_values, diag_values, rtol=1e-3, atol=1e-4)


def test_run_full_robustness_condition_threshold_uses_cov_condition(tmp_path):
    df = _make_ill_conditioned_df()
    cfg = make_cfg(tmp_path, df)
    cfg.portfolio["weighting_scheme"] = "robust_mv"
    cfg.portfolio["robustness"] = {
        "shrinkage": {"enabled": False},
        "condition_check": {
            "enabled": True,
            "threshold": 1.0,
            "safe_mode": "risk_parity",
            "diagonal_loading_factor": 1.0e-6,
        },
    }
    cfg.sample_split.update(
        {
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2020-08",
        }
    )

    df_from_csv = pd.read_csv(cfg.data["csv_path"])
    preprocess = pipeline._prepare_preprocess_stage(
        df_from_csv,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=None,
        periods_per_year_override=None,
        allow_risk_free_fallback=False,
    )
    assert not isinstance(preprocess, pipeline.PipelineResult)
    window = pipeline._build_sample_windows(
        preprocess,
        in_start="2020-01",
        in_end="2020-06",
        out_start="2020-07",
        out_end="2020-08",
    )
    assert not isinstance(window, pipeline.PipelineResult)
    cov = window.in_df[["A", "B", "C"]].cov()
    raw_condition = float(np.linalg.cond(cov.values))
    cfg.portfolio["robustness"]["condition_check"]["threshold"] = raw_condition / 2.0

    res = pipeline.run_full(cfg)
    diag = res["weight_engine_diagnostics"]
    assert diag["condition_source"] == "raw_cov"
    assert diag["condition_number"] == pytest.approx(raw_condition)
    assert diag["used_safe_mode"] is True
    fallback = res["weight_engine_fallback"]
    assert fallback["reason"] == "condition_threshold_exceeded"
    assert fallback["safe_mode"] == "risk_parity"
    assert fallback["raw_condition_number"] == pytest.approx(raw_condition)
    assert fallback["shrunk_condition_number"] == pytest.approx(raw_condition)
    assert fallback["shrinkage"]["method"] == "none"


def _make_two_fund_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": [0.02, 0.01, 0.03, 0.04, 0.02, 0.01],
            "B": [0.01, 0.02, 0.02, 0.03, 0.01, 0.02],
        }
    )


def _make_three_fund_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": [0.02, 0.01, 0.03, 0.04, 0.02, 0.01],
            "B": [0.01, 0.008, 0.012, 0.009, 0.011, 0.01],
            "C": [0.002, 0.001, 0.003, 0.0025, 0.0015, 0.002],
        }
    )


def _make_ill_conditioned_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    base = np.array([0.01, 0.011, 0.009, 0.012, 0.0105, 0.0115, 0.0095, 0.0108])
    jitter = np.array([1e-4, -1e-4, 5e-5, -5e-5, 8e-5, -8e-5, 6e-5, -6e-5])
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": base,
            "B": base + jitter,
            "C": base * 0.2 + 0.001,
        }
    )


def test_run_analysis_applies_constraints(monkeypatch):
    df = _make_two_fund_df()
    captured: dict[str, object] = {}

    def fake_apply_constraints(
        weights: pd.Series, cons: dict[str, object]
    ) -> pd.Series:
        captured["weights_before"] = weights.copy()
        captured["constraints"] = cons.copy()
        return pd.Series({"A": 0.6, "B": 0.4})

    monkeypatch.setattr(
        "trend_analysis.engine.optimizer.apply_constraints",
        fake_apply_constraints,
    )

    constraints_cfg = {
        "long_only": True,
        "max_weight": 0.55,
        "group_caps": {"grp": 0.8},
        "groups": {"A": "grp", "B": "grp"},
    }

    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        constraints=constraints_cfg,
        **RUN_ANALYSIS_KWARGS,
    )

    assert res is not None
    assert captured["constraints"] == constraints_cfg
    # The pipeline should adopt the constrained weights returned by the helper.
    assert res["fund_weights"] == {"A": pytest.approx(0.6), "B": pytest.approx(0.4)}


def test_run_analysis_constraint_failure_falls_back(monkeypatch):
    df = _make_two_fund_df()
    calls: list[tuple[pd.Series, dict[str, object]]] = []

    def boom(*args, **kwargs):
        weights = args[0]
        cons = args[1]
        calls.append((weights.copy(), cons.copy()))
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "trend_analysis.engine.optimizer.apply_constraints",
        boom,
    )

    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        constraints={"max_weight": 0.6},
        **RUN_ANALYSIS_KWARGS,
    )

    assert calls, "Expected apply_constraints to be invoked"
    assert res is not None
    # When constraints processing fails the pipeline should keep the equal weights.
    assert res["fund_weights"] == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}


def test_run_analysis_injects_avg_corr_metric():
    df = _make_two_fund_df()
    stats_cfg = RiskStatsConfig(
        metrics_to_run=canonical_metric_list(["annual_return", "volatility"]),
        risk_free=0.0,
    )
    setattr(stats_cfg, "extra_metrics", ["AvgCorr"])

    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        stats_cfg=stats_cfg,
        **RUN_ANALYSIS_KWARGS,
    )

    assert res is not None
    score_frame = res["score_frame"]
    assert "AvgCorr" in score_frame.columns
    # The optional injection should provide finite correlation diagnostics when
    # more than one fund is selected.
    assert score_frame["AvgCorr"].notna().all()


# Tests that monkeypatch pipeline functions must run serially.
@pytest.mark.serial
def test_run_analysis_benchmark_ir_fallback(monkeypatch):
    df = _make_two_fund_df()
    df["SPX"] = 0.01

    original_ir = pipeline.information_ratio
    raised = {"flag": False}

    def selective_boom(series_a, series_b):
        target_name = getattr(series_b, "name", "")
        if target_name == "SPX" and getattr(series_a, "ndim", 1) == 1:
            raised["flag"] = True
            raise ZeroDivisionError("bad benchmark")
        return original_ir(series_a, series_b)

    # Patch the module-level binding in pipeline.py so run_analysis sees our stub
    monkeypatch.setattr(pipeline, "information_ratio", selective_boom)

    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        benchmarks={"spx": "SPX"},
        **RUN_ANALYSIS_KWARGS,
    )

    assert res is not None
    assert raised["flag"] is True
