"""Extra coverage for ``trend_analysis.multi_period.engine.run`` price-frame
handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class DummyConfig:
    """Minimal config object that satisfies ``mp_engine.run`` dependencies."""

    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-03",
        }
    )
    data: Dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "standard",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: List[Any] = field(default_factory=list)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def test_run_requires_mapping_for_price_frames() -> None:
    cfg = DummyConfig()
    with pytest.raises(TypeError, match="price_frames must be a dict"):
        mp_engine.run(cfg, price_frames="not-a-dict")


def test_run_validates_price_frame_values_are_dataframes() -> None:
    cfg = DummyConfig()
    bad_frames = {"2020-01-31": "not-a-dataframe"}
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        mp_engine.run(cfg, price_frames=bad_frames)


def test_run_requires_date_column_in_price_frames() -> None:
    cfg = DummyConfig()
    missing_date = pd.DataFrame({"Value": [1.23]})
    frames = {"2020-01-31": missing_date}
    with pytest.raises(ValueError, match="missing required columns"):
        mp_engine.run(cfg, price_frames=frames)


def test_run_rejects_empty_price_frames_collection() -> None:
    cfg = DummyConfig()
    with pytest.raises(ValueError, match="price_frames is empty"):
        mp_engine.run(cfg, price_frames={})


def test_run_combines_price_frames_and_invokes_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DummyConfig()

    frame_one = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.1, 0.2],
        }
    )
    frame_two = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-02-29", "2020-03-31"]),
            "FundB": [0.3, 0.4],
        }
    )

    captured_dates: list[pd.Timestamp] = []
    captured_columns: list[List[str]] = []

    def fake_run_analysis(
        df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        # ``run`` should provide a DataFrame that is date-ordered and deduplicated.
        assert df["Date"].is_monotonic_increasing
        assert not df["Date"].duplicated().any()
        captured_dates.extend(df["Date"].tolist())
        captured_columns.append(sorted(df.columns))
        return {"analysis": "ok"}

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_run_analysis)

    results = mp_engine.run(
        cfg,
        price_frames={
            "2020-01-31": frame_one,
            "2020-02-29": frame_two,
        },
    )

    # ``run`` should return a result dict per generated period with the period tuple attached.
    assert results
    assert all("period" in r for r in results)
    # Ensure both price frame columns make it through the concatenation.
    assert any("FundA" in cols and "FundB" in cols for cols in captured_columns)
    # The combined data should cover all unique dates from both frames.
    expected_dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    assert set(captured_dates) == set(expected_dates)
