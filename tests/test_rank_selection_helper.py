import pandas as pd
import pytest

from trend_analysis.core.rank_selection import (
    RiskStatsConfig,
    _apply_transform,
    rank_select_funds,
)


def test_apply_transform_percentile_and_errors():
    s = pd.Series([3, 2, 1])
    out = _apply_transform(s, mode="percentile", rank_pct=0.5)
    assert out.notna().sum() == 2
    with pytest.raises(ValueError):
        _apply_transform(s, mode="percentile")


def test_apply_transform_zscore_and_unknown_mode():
    s = pd.Series([1, 1, 1])
    zeros = _apply_transform(s, mode="zscore")
    assert (zeros == 0).all()
    with pytest.raises(ValueError):
        _apply_transform(s, mode="nope")


def test_rank_select_funds_blended_requires_weights_and_no_dedupe():
    df = pd.DataFrame({"A 1": [0.1, 0.2], "A 2": [0.2, 0.3], "B": [0.3, 0.4]})
    cfg = RiskStatsConfig()
    with pytest.raises(ValueError):
        rank_select_funds(df, cfg, score_by="blended", inclusion_approach="top_n", n=1)
    res = rank_select_funds(df, cfg, inclusion_approach="top_n", n=2, limit_one_per_firm=False)
    assert len(res) == 2
