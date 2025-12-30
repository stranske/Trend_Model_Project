"""Extra coverage for ``trend_analysis.multi_period.engine.run`` price-frame
handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from trend_analysis.multi_period import engine as mp_engine


@dataclass
class DummyConfig:
    """Minimal config object that satisfies ``mp_engine.run`` dependencies."""

    multi_period: dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-03",
        }
    )
    data: dict[str, Any] = field(default_factory=lambda: {"csv_path": "unused.csv"})
    portfolio: dict[str, Any] = field(
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
    vol_adjust: dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: list[Any] = field(default_factory=list)
    run: dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

    def model_dump(self) -> dict[str, Any]:
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


def test_run_price_frames_type_error_mentions_key() -> None:
    cfg = DummyConfig()
    bad_frames = {"2020-02-29": [1, 2, 3]}

    with pytest.raises(TypeError) as excinfo:
        mp_engine.run(cfg, price_frames=bad_frames)

    message = str(excinfo.value)
    assert "price_frames['2020-02-29']" in message


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


def test_run_reports_available_columns_when_date_missing() -> None:
    cfg = DummyConfig()
    bad_frame = pd.DataFrame({"Close": [1.0], "Open": [0.9]})

    with pytest.raises(ValueError) as excinfo:
        mp_engine.run(cfg, price_frames={"2020-01-31": bad_frame})

    message = str(excinfo.value)
    assert "Available columns" in message
    assert "Close" in message and "Open" in message


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
    captured_columns: list[list[str]] = []

    def fake_run_analysis(df: pd.DataFrame, *args: Any, **kwargs: Any) -> dict[str, Any]:
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


def test_run_requires_csv_path_when_frame_not_provided() -> None:
    cfg = DummyConfig()
    cfg.data = {}

    with pytest.raises(KeyError, match=r"cfg\.data\['csv_path'\] must be provided"):
        mp_engine.run(cfg, df=None)


def test_run_raises_when_loader_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyConfig()

    def fake_loader(path: str, **_: object) -> pd.DataFrame | None:
        raise FileNotFoundError(path)

    monkeypatch.setattr(mp_engine, "load_csv", fake_loader)

    with pytest.raises(FileNotFoundError):
        mp_engine.run(cfg, df=None)
