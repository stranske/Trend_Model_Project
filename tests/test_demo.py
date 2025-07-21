import pandas as pd
import scripts.demo as demo
from trend_analysis.core import rank_selection as rs


def test_demo_runs(tmp_path, capsys):
    df = pd.read_csv("hedge_fund_returns_with_indexes.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    mask = df["Date"].between(
        pd.Period("2012-01", "M").to_timestamp("M"),
        pd.Period("2012-06", "M").to_timestamp("M"),
    )
    funds = [c for c in df.columns if c not in {"Date", "Risk-Free Rate"}]
    expected_rank = rs.rank_select_funds(
        df.loc[mask, funds],
        rs.RiskStatsConfig(risk_free=0.0),
        inclusion_approach="top_n",
        n=1,
        score_by="AnnualReturn",
    )
    res = demo.main(out_dir=tmp_path)
    captured = capsys.readouterr().out
    assert "Vol-Adj Trend Analysis" in captured
    assert "Generated periods:" in captured
    assert "Multi-period run count:" in captured
    assert "Rebalanced weights:" in captured
    assert "Multi-period final weights:" in captured
    assert "Multi-period weight history:" in captured
    assert "Analysis selected:" in captured
    assert "Top fund by ranking:" in captured
    assert "Multi-period selections:" in captured
    assert (tmp_path / "analysis.xlsx").exists()
    assert (tmp_path / "analysis_metrics.csv").exists()
    assert (tmp_path / "analysis_metrics.json").exists()
    assert (tmp_path / "analysis_history.csv").exists()
    assert (tmp_path / "analysis_history.json").exists()
    assert res["cli_rc"] == 0
    assert res["cli_json_rc"] == 0
    assert res["run_rc"] == 0
    assert res["detailed_rc"] == 0
    assert not res["metrics_df"].empty
    assert isinstance(res["score_frame"], pd.DataFrame)
    assert res["periods"]
    assert len(res["periods"]) == 10
    first = res["periods"][0]
    assert first.in_start == "2012-01-01"
    assert first.out_end == "2012-03-31"
    assert res["mp_res"]["n_periods"] == len(res["periods"])
    assert res["mp_res"]["periods"] == res["periods"]
    assert res["rf_col"] == "Risk-Free Rate"
    assert "Vol-Adj Trend Analysis" in res["summary_text"]
    assert "annual_return" in res["available"]
    assert res["loaded_version"] == "1"
    expected_wts = {
        c: 1 / len(res["score_frame"].columns) for c in res["score_frame"].columns
    }
    assert res["rb_weights"] == expected_wts
    assert res["rb_cfg"] == {"triggers": {"sigma1": {"sigma": 1, "periods": 2}}}
    assert len(res["mp_history"]) == len(res["periods"])
    assert res["mp_history"][-1] == expected_wts
    for w in res["mp_history"]:
        assert w == expected_wts
    assert res["mp_index"] == [
        f"{p.in_start[:7]}_{p.out_end[:7]}" for p in res["periods"]
    ]
    assert res["nb_clean"] is True
    assert isinstance(res["mp_history_df"], pd.DataFrame)
    assert res["mp_history_df"].shape[0] == len(res["periods"])
    assert res["mp_history_df"].index.tolist() == res["mp_index"]
    assert res["mp_history_df"].to_dict("records") == res["mp_history"]
    assert res["mp_history_df"].columns.tolist() == res["score_frame"].columns.tolist()
    assert res["mp_weights"] == expected_wts
    assert res["mp_res"] == {
        "periods": res["periods"],
        "n_periods": len(res["periods"]),
    }
    assert isinstance(res["analysis_res"], dict)
    assert isinstance(res["analysis_res"].get("score_frame"), pd.DataFrame)
    assert res["analysis_res"]["selected_funds"]
    assert res["ranked"]
    assert res["mp_selected"]
    assert len(res["mp_selected"]) == len(res["periods"])
    assert all(res["mp_selected"])
    assert res["ranked"] == expected_rank
    assert res["score_frame"].attrs["insample_len"] == 6
    assert res["score_frame"].attrs["period"] == ("2012-01", "2012-06")
    assert "information_ratio" in res["metrics_df"].columns
    assert "ir_eq60" in res["metrics_df"].columns
    df_csv = pd.read_csv(tmp_path / "analysis_metrics.csv", index_col=0)
    assert df_csv.columns.tolist() == res["metrics_df"].columns.tolist()
    import openpyxl

    wb = openpyxl.load_workbook(tmp_path / "analysis.xlsx")
    assert set(wb.sheetnames) == {"metrics", "summary", "history"}
    assert "eq60" in res["full_res"]["benchmark_ir"]
    assert "equal_weight" in res["full_res"]["benchmark_ir"]["eq60"]
