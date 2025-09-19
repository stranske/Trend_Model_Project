"""Additional coverage for ``trend_analysis.multi_period.engine``."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.multi_period import engine


class MinimalCfg:
    """Minimal configuration facade required by ``engine.run``."""

    def __init__(self, *, enable_cache: bool = False) -> None:
        self.data = {"csv_path": "unused.csv"}
        self.portfolio = {
            "policy": "vanilla",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": None,
            "manual_list": None,
            "indices_list": None,
        }
        self.vol_adjust = {"target_vol": 1.0}
        self.performance = {
            "enable_cache": enable_cache,
            "incremental_cov": False,
        }
        self.benchmarks = {}
        self.seed = 11
        self.run = {"monthly_cost": 0.0}
        self._multi_period = {
            "frequency": "M",
            "start": "2020-01",
            "end": "2020-06",
            "in_sample_len": 3,
            "out_sample_len": 1,
        }

    def model_dump(self) -> dict[str, object]:
        return {"multi_period": self._multi_period}


@pytest.fixture
def single_period() -> SimpleNamespace:
    return SimpleNamespace(
        in_start="2020-01-31",
        in_end="2020-03-31",
        out_start="2020-04-30",
        out_end="2020-04-30",
    )


def test_run_rejects_invalid_price_frames() -> None:
    cfg = MinimalCfg()

    with pytest.raises(TypeError):
        engine.run(cfg, price_frames=[])  # type: ignore[arg-type]

    bad = pd.DataFrame({"not_date": [1, 2]})
    with pytest.raises(ValueError):
        engine.run(cfg, price_frames={"2020-01": bad})

    with pytest.raises(ValueError):
        engine.run(cfg, price_frames={})


def test_run_combines_price_frames_and_invokes_analysis(
    monkeypatch: pytest.MonkeyPatch, single_period: SimpleNamespace
) -> None:
    cfg = MinimalCfg()

    frame_one = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.01, 0.02],
            "FundB": [0.03, 0.04],
        }
    )
    frame_two = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-03-31", "2020-04-30"]),
            "FundA": [0.05, 0.06],
            "FundB": [0.07, 0.08],
        }
    )

    captured: list[pd.DataFrame] = []

    monkeypatch.setattr(engine, "generate_periods", lambda _: [single_period])

    def fake_run_analysis(df: pd.DataFrame, *_, **__):
        captured.append(df.copy())
        return {"out_user_stats": {"sharpe": 1.23}}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    results = engine.run(cfg, price_frames={"a": frame_one, "b": frame_two})

    assert len(results) == 1
    assert list(captured[0]["Date"]) == sorted(captured[0]["Date"].tolist())


def test_run_attaches_covariance_diagnostics(
    monkeypatch: pytest.MonkeyPatch, single_period: SimpleNamespace
) -> None:
    cfg = MinimalCfg(enable_cache=True)

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=6, freq="M"),
            "FundA": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "FundB": [0.02, 0.01, 0.00, -0.01, -0.02, -0.03],
        }
    )

    monkeypatch.setattr(engine, "generate_periods", lambda _: [single_period])

    def fake_run_analysis(df: pd.DataFrame, *_, **__):
        return {"out_user_stats": {"sharpe": 0.5}}

    monkeypatch.setattr(engine, "_run_analysis", fake_run_analysis)

    results = engine.run(cfg, df=df)

    assert len(results) == 1
    result = results[0]
    assert "cov_diag" in result
    assert isinstance(result["cov_diag"], list)
    assert result.get("cache_stats", {}).get("incremental_updates") == 0
