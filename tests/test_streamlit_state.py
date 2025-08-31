import pandas as pd


def test_date_range_month_count():
    dates = pd.date_range(start="2023-01-31", end="2023-12-31", freq="M")
    assert len(dates) == 12
