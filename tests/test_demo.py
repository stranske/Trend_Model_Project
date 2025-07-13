import scripts.demo as demo


def test_demo_runs(tmp_path, capsys):
    demo.main(out_dir=tmp_path)
    captured = capsys.readouterr().out
    assert "Vol-Adj Trend Analysis" in captured
    assert (tmp_path / "analysis.xlsx").exists()
    assert (tmp_path / "analysis_metrics.csv").exists()
    assert (tmp_path / "analysis_metrics.json").exists()
