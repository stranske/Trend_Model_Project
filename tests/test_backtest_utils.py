import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backtest import shift_by_execution_lag


def test_shift_by_execution_lag_series() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    shifted = shift_by_execution_lag(series, lag=2)

    expected = pd.Series([np.nan, np.nan, 1.0, 2.0], index=idx, dtype=float)
    pdt.assert_series_equal(shifted, expected, check_exact=False)
    assert shifted.attrs["execution_lag"] == 2


def test_shift_by_execution_lag_dataframe() -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="B")
    frame = pd.DataFrame({"A": [0.2, 0.4, 0.6], "B": [0.8, 0.6, 0.4]}, index=idx)
    frame.attrs["source"] = "test"

    shifted = shift_by_execution_lag(frame, lag=1)

    expected = pd.DataFrame(
        {"A": [np.nan, 0.2, 0.4], "B": [np.nan, 0.8, 0.6]},
        index=idx,
        dtype=float,
    )
    pdt.assert_frame_equal(shifted, expected, check_exact=False)
    assert shifted.attrs["execution_lag"] == 1
    assert shifted.attrs["source"] == "test"


def test_shift_by_execution_lag_rejects_non_pandas() -> None:
    with pytest.raises(TypeError):
        shift_by_execution_lag([1, 2, 3], lag=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        shift_by_execution_lag(pd.Series([1, 2, 3]), lag=-1)
