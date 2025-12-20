import json
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from streamlit_app.components import comparison


def _result_with_metrics(sharpe: float, cagr: float) -> SimpleNamespace:
    metrics = pd.DataFrame([{"Sharpe": sharpe, "CAGR": cagr}])
    return SimpleNamespace(metrics=metrics, details={})


def _result_with_periods(turnover: float, txn_cost: float) -> SimpleNamespace:
    return SimpleNamespace(
        metrics=pd.DataFrame(),
        details={
            "period_results": [
                {
                    "period": ("", "", "2023-01", "2023-03"),
                    "selected_funds": ["A", "B"],
                    "turnover": turnover,
                    "transaction_cost": txn_cost,
                }
            ]
        },
    )


def _result_with_changes(reason: str, count: int) -> SimpleNamespace:
    changes = [{"reason": reason, "manager": f"M{i}"} for i in range(count)]
    return SimpleNamespace(
        metrics=pd.DataFrame(),
        details={"period_results": [{"manager_changes": changes}]},
    )


def test_comparison_run_key_changes_when_model_or_data_differs() -> None:
    model_a = {"lookback_periods": 3}
    model_b = {"lookback_periods": 6}
    funds = ["FundA", "FundB"]
    key_a = comparison.comparison_run_key(
        model_a, benchmark="SPX", funds=funds, data_fingerprint="abc", risk_free=None
    )
    key_b = comparison.comparison_run_key(
        model_b, benchmark="SPX", funds=funds, data_fingerprint="abc", risk_free=None
    )
    key_c = comparison.comparison_run_key(
        model_a,
        benchmark="SPX",
        funds=funds,
        data_fingerprint="changed",
        risk_free=None,
    )
    assert key_a != key_b
    assert key_a != key_c


def test_metric_delta_frame_computes_deltas() -> None:
    res_a = _result_with_metrics(sharpe=1.0, cagr=0.10)
    res_b = _result_with_metrics(sharpe=1.4, cagr=0.12)
    delta = comparison.metric_delta_frame(res_a, res_b, label_a="A", label_b="B")
    sharpe_row = delta.loc[delta["Metric"] == "Sharpe"].iloc[0]
    cagr_row = delta.loc[delta["Metric"] == "CAGR"].iloc[0]
    assert sharpe_row["Delta (B - A)"] == pytest.approx(0.4)
    assert cagr_row["Delta (B - A)"] == pytest.approx(0.02)


def test_period_delta_aligns_by_period_and_calculates_changes() -> None:
    res_a = _result_with_periods(turnover=0.20, txn_cost=0.01)
    res_b = _result_with_periods(turnover=0.25, txn_cost=0.015)
    delta = comparison.period_delta(res_a, res_b, label_a="A", label_b="B")
    assert delta["Turnover Δ (B - A)"].iloc[0] == pytest.approx(0.05)
    assert delta["Transaction Cost Δ (B - A)"].iloc[0] == pytest.approx(0.005)


def test_manager_change_delta_counts_reasons() -> None:
    res_a = _result_with_changes("z_exit", 2)
    res_b = _result_with_changes("z_exit", 3)
    delta = comparison.manager_change_delta(res_a, res_b, label_a="A", label_b="B")
    assert delta.loc[0, "Count Δ (B - A)"] == 1


def test_comparison_bundle_includes_configs_and_diff_text(tmp_path) -> None:
    metrics = pd.DataFrame(
        [{"Metric": "Sharpe", "A": 1.0, "B": 1.2, "Delta (B - A)": 0.2}]
    )
    periods = pd.DataFrame(
        [
            {
                "Period": "2023-01 → 2023-03",
                "A Selected Funds": 2,
                "B Selected Funds": 3,
                "Selected Funds Δ (B - A)": 1,
            }
        ]
    )
    bundle = comparison.build_comparison_bundle(
        config_a={"lookback_periods": 3},
        config_b={"lookback_periods": 6},
        diff_text="~ lookback_periods: (A) 3 -> (B) 6",
        metrics=metrics,
        periods=periods,
        manager_changes=None,
    )
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(bundle)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        names = set(zf.namelist())
        assert "config_A.json" in names
        assert "config_B.json" in names
        assert "config_diff.txt" in names
        with zf.open("config_diff.txt") as diff_file:
            content = diff_file.read().decode("utf-8")
            assert "lookback_periods" in content
        config_a_payload = json.loads(zf.read("config_A.json"))
        assert config_a_payload["lookback_periods"] == 3


def test_results_page_comparison_uses_cache(monkeypatch) -> None:
    import importlib

    results_page = importlib.import_module("streamlit_app.pages.3_Results")

    calls = {"n": 0}

    def _fake_run_analysis(df_for_analysis, model_state, benchmark, *, data_hash=None):
        calls["n"] += 1
        return SimpleNamespace(metrics=pd.DataFrame(), details={})

    # Minimal session_state for run_key computation + cache.
    monkeypatch.setattr(
        results_page.st,
        "session_state",
        {
            "data_fingerprint": "abc",
            "comparison_results_cache": {},
        },
        raising=False,
    )
    monkeypatch.setattr(
        results_page.analysis_runner, "run_analysis", _fake_run_analysis
    )

    df_for_analysis = pd.DataFrame({"FundA": [0.01, 0.02], "FundB": [0.0, -0.01]})
    model_state = {"lookback_periods": 3}
    funds = ["FundA", "FundB"]

    res1, key1 = results_page._run_comparison_analysis(
        config_name="A",
        model_state=model_state,
        df_for_analysis=df_for_analysis,
        benchmark="SPX",
        data_hash="abc:cols",
        funds=funds,
        selected_rf=None,
    )
    res2, key2 = results_page._run_comparison_analysis(
        config_name="A",
        model_state=model_state,
        df_for_analysis=df_for_analysis,
        benchmark="SPX",
        data_hash="abc:cols",
        funds=funds,
        selected_rf=None,
    )

    assert key1 == key2
    assert calls["n"] == 1
    assert res2 is res1
