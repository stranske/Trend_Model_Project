from types import SimpleNamespace

import numpy as np
import pandas as pd

import trend_analysis.core.rank_selection as rs
import trend_analysis.perf.cache as perf_cache


def test_call_metric_series_forwards_risk_free_override(monkeypatch):
    df = pd.DataFrame({"FundA": [0.01, 0.02]})
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    captured: dict[str, float | None] = {"value": None}

    def with_override(in_sample_df, metric_name, stats_cfg, *, risk_free_override=None):
        captured["value"] = risk_free_override
        return pd.Series([1.0], index=in_sample_df.columns)

    monkeypatch.setattr(rs, "_compute_metric_series", with_override)

    result = rs._call_metric_series(
        df, "AnnualReturn", cfg, risk_free_override=0.25
    )
    assert captured["value"] == 0.25
    assert result.index.tolist() == ["FundA"]


def test_call_metric_series_skips_risk_free_override_when_unsupported(monkeypatch):
    df = pd.DataFrame({"FundA": [0.01, 0.02]})
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    calls = {"count": 0}

    def no_override(in_sample_df, metric_name, stats_cfg):
        calls["count"] += 1
        return pd.Series([2.0], index=in_sample_df.columns)

    monkeypatch.setattr(rs, "_compute_metric_series", no_override)

    result = rs._call_metric_series(
        df, "AnnualReturn", cfg, risk_free_override=0.25
    )
    assert calls["count"] == 1
    assert result.iloc[0] == 2.0


def _make_bundle(df: pd.DataFrame) -> rs.WindowMetricBundle:
    cfg = rs.RiskStatsConfig()
    return rs.WindowMetricBundle(
        key=None,
        start="2020-01",
        end="2020-02",
        freq="ME",
        stats_cfg_hash=rs._stats_cfg_hash(cfg),
        universe=tuple(df.columns),
        in_sample_df=df,
        _metrics=pd.DataFrame(index=df.columns, dtype=float),
    )


def test_ensure_cov_payload_populates_bundle(monkeypatch):
    df = pd.DataFrame({"FundA": [0.01, 0.02]})
    bundle = _make_bundle(df)
    payload = SimpleNamespace(cov=np.array([[1.0]]))

    def fake_compute(frame):
        assert frame is df
        return payload

    monkeypatch.setattr(perf_cache, "compute_cov_payload", fake_compute)

    result = rs._ensure_cov_payload(df, bundle)
    assert result is payload
    assert bundle.cov_payload is payload


def test_ensure_cov_payload_reuses_existing_payload(monkeypatch):
    df = pd.DataFrame({"FundA": [0.01, 0.02]})
    payload = SimpleNamespace(cov=np.array([[2.0]]))
    bundle = _make_bundle(df)
    bundle.cov_payload = payload

    def boom(_frame):
        raise AssertionError("compute_cov_payload should not be called")

    monkeypatch.setattr(perf_cache, "compute_cov_payload", boom)

    result = rs._ensure_cov_payload(df, bundle)
    assert result is payload


def test_metric_from_cov_payload_handles_variants():
    df = pd.DataFrame({"FundA": [0.01, 0.02], "FundB": [0.02, 0.03]})
    payload = SimpleNamespace(cov=np.array([[4.0, 0.0], [0.0, 9.0]]))

    cov_var = rs._metric_from_cov_payload("__COV_VAR__", df, payload)
    assert cov_var.name == "CovVar"
    assert cov_var.to_dict() == {"FundA": 4.0, "FundB": 9.0}

    avg_corr = rs._metric_from_cov_payload("AvgCorr", df, payload)
    assert avg_corr.name == "AvgCorr"
    assert avg_corr.to_dict() == {"FundA": 0.0, "FundB": 0.0}

    single_payload = SimpleNamespace(cov=np.array([[1.0]]))
    single_avg = rs._metric_from_cov_payload("AvgCorr", df[["FundA"]], single_payload)
    assert single_avg.to_dict() == {"FundA": 0.0}


def test_cov_metric_from_payload_supports_cov_var_and_singleton_avgcorr():
    payload = SimpleNamespace(cov=np.array([[1.0, 0.0], [0.0, 2.0]]))
    cov_var = rs._cov_metric_from_payload("__COV_VAR__", payload, ["A", "B"])
    assert cov_var.name == "CovVar"
    assert cov_var.to_dict() == {"A": 1.0, "B": 2.0}

    singleton = SimpleNamespace(cov=np.array([[1.0]]))
    avg_corr = rs._cov_metric_from_payload("AvgCorr", singleton, ["Solo"])
    assert avg_corr.to_dict() == {"Solo": 0.0}
