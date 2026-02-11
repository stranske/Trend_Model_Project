from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.viz.adapters import (
    LOOKBACK_PERIODS_VALIDATION_MESSAGE,
    make_paths,
    terminal_returns,
)


def _sample_nav_paths() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    return pd.DataFrame(
        {
            0: [1.0, 1.1, 1.21, 1.20],
            1: [1.0, 0.9, 0.99, 1.01],
        },
        index=index,
    )


def test_terminal_returns_rejects_empty_lookback_iterable() -> None:
    canonical = make_paths(_sample_nav_paths())

    with pytest.raises(ValueError, match=LOOKBACK_PERIODS_VALIDATION_MESSAGE):
        terminal_returns(canonical, lookback_periods=[])


def test_terminal_returns_rejects_non_positive_lookbacks() -> None:
    canonical = make_paths(_sample_nav_paths())

    with pytest.raises(ValueError, match=LOOKBACK_PERIODS_VALIDATION_MESSAGE):
        terminal_returns(canonical, lookback_periods=[0, -2])


def test_terminal_returns_rejects_non_integer_lookbacks() -> None:
    canonical = make_paths(_sample_nav_paths())

    with pytest.raises(ValueError, match=LOOKBACK_PERIODS_VALIDATION_MESSAGE):
        terminal_returns(canonical, lookback_periods=[1.5, "3", None])


def test_terminal_returns_filters_invalid_iterable_entries_and_uses_first_valid() -> None:
    canonical = make_paths(_sample_nav_paths())

    lookback = terminal_returns(canonical, lookback_periods=[0, -1, 2, 1.1])

    assert lookback.loc[0, "lookback_periods"] == 2
    assert lookback.loc[0, "terminal_return"] == pytest.approx(0.09090909)
    assert lookback.loc[1, "terminal_return"] == pytest.approx(0.12222222)
