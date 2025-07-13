import pandas as pd
import scripts.demo as demo


def test_demo_runs(tmp_path, capsys):
    res = demo.main(out_dir=tmp_path)
    captured = capsys.readouterr().out
    assert "Vol-Adj Trend Analysis" in captured
    assert (tmp_path / "analysis.xlsx").exists()
    assert (tmp_path / "analysis_metrics.csv").exists()
    assert (tmp_path / "analysis_metrics.json").exists()
    assert res["cli_rc"] == 0
    assert res["run_rc"] == 0
    assert not res["metrics_df"].empty
    assert isinstance(res["score_frame"], pd.DataFrame)
    assert res["periods"]
    assert res["mp_res"] == {}
