from collections import OrderedDict

import pandas as pd
import pytest

from trend_analysis.metrics.turnover import realized_turnover, turnover_cost


def test_realized_turnover_sorts_mapping_and_handles_missing():
    weights = OrderedDict(
        {
            pd.Timestamp("2020-03-31"): pd.Series({"A": 0.6, "B": 0.4}),
            pd.Timestamp("2020-01-31"): pd.Series({"A": 0.5, "B": 0.5}),
            pd.Timestamp("2020-02-29"): pd.Series({"A": 0.55}),
        }
    )

    turnover = realized_turnover(weights)

    # Index should be sorted chronologically and missing assets filled with 0.
    assert list(turnover.index) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
        pd.Timestamp("2020-03-31"),
    ]
    # Feb row compares against Jan weights -> |0.55-0.5| + |0-0.5|.
    feb_turnover = abs(0.55 - 0.5) + abs(0.0 - 0.5)
    mar_turnover = abs(0.6 - 0.55) + abs(0.4 - 0.0)
    assert turnover.loc[pd.Timestamp("2020-02-29"), "turnover"] == pytest.approx(feb_turnover)
    assert turnover.loc[pd.Timestamp("2020-03-31"), "turnover"] == pytest.approx(mar_turnover)


def test_turnover_cost_scales_basis_points():
    df = pd.DataFrame(
        {
            "A": [0.5, 0.45, 0.55],
            "B": [0.5, 0.55, 0.45],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )
    costs = turnover_cost(df, cost_bps=25)

    expected_turnover = realized_turnover(df)["turnover"]
    assert costs.equals(expected_turnover * 0.0025)
