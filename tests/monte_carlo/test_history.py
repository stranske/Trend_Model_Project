"""Tests for the canonical Monte Carlo history loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.io.market_data import MarketDataMode
from trend_analysis.monte_carlo import history
from trend_analysis.monte_carlo.folds import FoldGenerator


def _frame(values: list[float], *, mode: MarketDataMode) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"FundA": values},
        index=pd.date_range("2024-01-31", periods=len(values), freq="ME"),
    )
    frame.attrs["market_data_mode"] = mode.value
    return frame


def test_load_price_history_uses_canonical_csv_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_csv(path: str, **kwargs: object) -> pd.DataFrame:
        captured.update(path=path, kwargs=kwargs)
        return _frame([0.10, -0.05], mode=MarketDataMode.RETURNS)

    monkeypatch.setattr(history, "load_csv", fake_load_csv)

    result = history.load_price_history(Path("returns.csv"))

    assert captured == {
        "path": "returns.csv",
        "kwargs": {"errors": "raise", "include_date_column": False},
    }
    assert result["FundA"].tolist() == pytest.approx([110.0, 104.5])


def test_load_price_history_uses_canonical_parquet_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _frame([100.0, 101.0], mode=MarketDataMode.PRICE)
    monkeypatch.setattr(history, "load_parquet", lambda *_args, **_kwargs: expected)

    result = history.load_price_history(Path("prices.parquet"))

    pd.testing.assert_frame_equal(result, expected)


def test_file_backed_history_is_timezone_compatible_with_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _frame([100.0, 101.0, 102.0], mode=MarketDataMode.PRICE)
    expected.index = expected.index.tz_localize("UTC")
    monkeypatch.setattr(history, "load_csv", lambda *_args, **_kwargs: expected)

    result = history.load_price_history(Path("prices.csv"))
    generator = FoldGenerator(mode="count_spaced", start="2024-03-31", n_folds=1)
    fold = generator.generate(result.index)[0]
    calibration = result.loc[fold.calibration_start : fold.calibration_end]

    assert result.index.tz is None
    assert calibration.index.tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
    ]


@pytest.mark.parametrize("values", [[], [0.01, -1.0]])
def test_load_price_history_rejects_invalid_returns(
    monkeypatch: pytest.MonkeyPatch,
    values: list[float],
) -> None:
    monkeypatch.setattr(
        history,
        "load_csv",
        lambda *_args, **_kwargs: _frame(values, mode=MarketDataMode.RETURNS),
    )

    with pytest.raises(ValueError, match="returns"):
        history.load_price_history(Path("returns.csv"))
