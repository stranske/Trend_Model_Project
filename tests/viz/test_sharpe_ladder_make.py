"""Tests for Sharpe ladder visualization helpers."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import pytest

import trend_analysis.viz.sharpe_ladder as sharpe_ladder


def _sample_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 2, 2],
            "strategy": ["A", "B", "A", "C"],
            "paths": [20, 20, 20, 20],
            "sharpe": [1.1, -0.2, 0.9, 0.4],
        }
    )


def test_prepare_sharpe_ladder_aggregates_and_sorts() -> None:
    ladder = sharpe_ladder.prepare_sharpe_ladder(_sample_summary())

    assert list(ladder.columns) == ["strategy", "sharpe"]
    assert ladder["strategy"].tolist() == ["B", "C", "A"]
    assert ladder["sharpe"].tolist() == [-0.2, 0.4, 1.0]


def test_make_sharpe_ladder_minimal_input() -> None:
    fig = sharpe_ladder.make(_sample_summary())

    assert isinstance(fig, go.Figure)
    payload = json.loads(fig.to_json())
    assert isinstance(payload, dict)


def test_prepare_sharpe_ladder_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="strategy"):
        sharpe_ladder.prepare_sharpe_ladder(pd.DataFrame({"sharpe": [1.0]}))

    with pytest.raises(ValueError, match="sharpe"):
        sharpe_ladder.prepare_sharpe_ladder(pd.DataFrame({"strategy": ["A"]}))
