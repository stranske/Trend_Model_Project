from __future__ import annotations

import pandas as pd

from trend_analysis.multi_period import engine as mp_engine


def test_lambda_tc_penalty_reduces_turnover() -> None:
    prev_weights = pd.Series({"A": 0.6, "B": 0.4})
    target_weights = pd.Series({"A": 0.2, "B": 0.8})

    base = mp_engine._apply_turnover_penalty(
        target_weights, prev_weights, 0.0, 0.0, 1.0
    )
    medium = mp_engine._apply_turnover_penalty(
        target_weights, prev_weights, 0.5, 0.0, 1.0
    )
    high = mp_engine._apply_turnover_penalty(
        target_weights, prev_weights, 0.9, 0.0, 1.0
    )

    def _turnover(weights: pd.Series) -> float:
        return float((weights - prev_weights).abs().sum())

    base_turnover = _turnover(base)
    assert base_turnover > _turnover(medium) > _turnover(high)
