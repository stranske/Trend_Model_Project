from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.monte_carlo.folds import FoldGenerator


def test_explicit_mode_requires_fold_starts() -> None:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")

    with pytest.raises(ValueError, match="fold_starts"):
        FoldGenerator(mode="explicit").generate(index)


def test_explicit_mode_dedupes_and_aligns_starts() -> None:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    generator = FoldGenerator(
        mode="explicit",
        fold_starts=["2020-05-31", "2020-03-31", "2020-05-31"],
    )

    folds = generator.generate(index)

    assert [fold.fold_id for fold in folds] == [1, 2]
    assert [fold.forecast_start for fold in folds] == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-05-31"),
    ]
    assert folds[0].calibration_end == pd.Timestamp("2020-02-29")
    assert folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert folds[0].label == "2020-03"
