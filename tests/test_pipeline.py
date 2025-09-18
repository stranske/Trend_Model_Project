import pandas as pd
import pytest

from trend_analysis import config, pipeline
from trend_analysis.config import Config


def make_cfg(tmp_path, df):
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg_dict = {
        "version": "1",
        "data": {"csv_path": str(csv)},
        "preprocessing": {},
        "vol_adjust": {"target_vol": 1.0},
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-06",
        },
        "portfolio": {},
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


def test_run_returns_dataframe(tmp_path):
    cfg = make_cfg(tmp_path, make_df())
    out = pipeline.run(cfg)
    assert not out.empty
    assert set(out.columns) == {
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "information_ratio",
        "max_drawdown",
    }


def test_run_with_benchmarks(tmp_path):
    df = make_df()
    df["SPX"] = 0.01
    cfg = make_cfg(tmp_path, df)
    cfg.benchmarks = {"spx": "SPX"}
    out = pipeline.run(cfg)
    assert "ir_spx" in out.columns


def test_run_returns_empty_when_no_funds(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=1, freq="ME"), "RF": 0.0}
    )
    cfg = make_cfg(tmp_path, df)
    result = pipeline.run(cfg)
    assert result.empty


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
    assert res is None


def test_run_analysis_missing_date():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(ValueError):
        pipeline.run_analysis(df, "2020-01", "2020-03", "2020-04", "2020-06", 1.0, 0.0)


def test_run_analysis_string_dates():
    df = make_df()
    df["Date"] = df["Date"].astype(str)
    res = pipeline.run_analysis(
        df, "2020-01", "2020-03", "2020-04", "2020-06", 1.0, 0.0
    )
    assert res is not None


def test_run_analysis_no_funds():
    df = pd.DataFrame(
        {"Date": pd.date_range("2020-01-31", periods=3, freq="ME"), "RF": 0.0}
    )
    res = pipeline.run_analysis(
        df, "2020-01", "2020-02", "2020-03", "2020-03", 1.0, 0.0
    )
    assert res is None


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
    )
    assert res["fund_weights"]["A"] == 1.0


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
    )

    assert calls, "Expected apply_constraints to be invoked"
    assert res is not None
    # When constraints processing fails the pipeline should keep the equal weights.
    assert res["fund_weights"] == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}
