import numpy as np
import pandas as pd
import pytest

from trend.cli_helpers import _apply_universe_mask


def test_apply_universe_mask_respects_membership_and_date_column_case_insensitive():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [4.0, 5.0, 6.0],
        }
    )
    mask = pd.DataFrame(
        {
            "AAA": [True, False, True],
            "BBB": [False, True, False],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    result = _apply_universe_mask(df, mask, date_column="date")

    expected = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "AAA": [1.0, np.nan, 3.0],
            "BBB": [np.nan, 5.0, np.nan],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_apply_universe_mask_raises_for_missing_member_columns():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "AAA": [1.0, 2.0],
        }
    )
    mask = pd.DataFrame(
        {
            "AAA": [True, True],
            "BBB": [False, True],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    with pytest.raises(KeyError) as excinfo:
        _apply_universe_mask(df, mask, date_column="date")

    assert "BBB" in str(excinfo.value)
