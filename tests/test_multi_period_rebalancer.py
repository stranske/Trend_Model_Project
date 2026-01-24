"""Focused tests for the multi-period ``Rebalancer`` helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.config import load_config
from trend_analysis.multi_period.replacer import Rebalancer


def _series(mapping: dict[str, float]) -> pd.Series:
    """Return a float series with deterministic index order for comparisons."""

    return pd.Series(mapping, dtype=float)


def test_rebalancer_drops_after_consecutive_soft_strikes() -> None:
    """Funds with repeated low z-scores should be removed."""

    reb = Rebalancer({"portfolio": {"threshold_hold": {"soft_strikes": 2}}})
    # Start with two holdings and trigger one consecutive low reading
    prev = _series({"A": 0.6, "B": 0.4})
    low_scores = pd.DataFrame({"zscore": {"A": -1.5, "B": 0.1}})

    mid = reb.apply_triggers(prev, low_scores)
    assert set(mid.index) == {"A", "B"}

    # Second strike should remove the struggling fund entirely
    final = reb.apply_triggers(mid, low_scores)
    assert list(final.index) == ["B"]
    assert pytest.approx(final.loc["B"], rel=1e-9) == 1.0


def test_rebalancer_hard_exit_forces_removal_below_threshold() -> None:
    """A hard exit threshold should force removals below the threshold."""

    cfg = {
        "portfolio": {
            "threshold_hold": {
                "z_exit_soft": -0.2,
                "soft_strikes": 1,
                "z_exit_hard": -0.5,
            }
        }
    }
    reb = Rebalancer(cfg)
    prev = {"A": 0.7, "B": 0.3}  # dict input exercises the Series conversion path
    frame = pd.DataFrame({"zscore": {"A": -0.6, "B": 0.0}})

    updated = reb.apply_triggers(prev, frame)
    assert set(updated.index) == {"B"}


def test_rebalancer_null_hard_thresholds_preserve_soft_exits() -> None:
    """Explicit null hard thresholds should not block soft exit behavior."""

    cfg = {
        "portfolio": {
            "threshold_hold": {
                "z_exit_soft": -0.2,
                "soft_strikes": 1,
                "z_exit_hard": None,
                "z_entry_hard": None,
            }
        }
    }
    reb = Rebalancer(cfg)
    prev = _series({"A": 0.7, "B": 0.3})
    frame = pd.DataFrame({"zscore": {"A": -0.4, "B": 0.0}})

    updated = reb.apply_triggers(prev, frame)
    assert list(updated.index) == ["B"]


def test_rebalancer_hard_candidates_fill_capacity_first() -> None:
    """Hard entry candidates should consume capacity before auto entries."""

    cfg = {
        "portfolio": {
            "threshold_hold": {
                "z_entry_hard": 2.0,
                "z_entry_soft": 1.0,
                "entry_soft_strikes": 1,
                "entry_eligible_strikes": 99,  # disable the eligible bucket
            },
            "constraints": {"max_funds": 2},
        }
    }
    reb = Rebalancer(cfg)
    prev = _series({"A": 1.0})
    frame = pd.DataFrame({"zscore": {"B": 3.0, "C": 1.2}})

    # Only the hard candidate can be added; auto candidate is skipped due to capacity
    result = reb.apply_triggers(prev, frame)
    assert set(result.index) == {"A", "B"}
    assert "C" not in result


def test_rebalancer_entry_hard_does_not_block_soft_entries() -> None:
    """Hard entry should not block soft/eligible additions."""

    cfg = {
        "portfolio": {
            "threshold_hold": {
                "z_entry_hard": 1.5,
                "z_entry_soft": 0.5,
                "entry_soft_strikes": 1,
                "entry_eligible_strikes": 1,
            },
            "constraints": {"max_funds": 3},
        }
    }
    reb = Rebalancer(cfg)
    prev = _series({"A": 1.0})
    frame = pd.DataFrame({"zscore": {"B": 1.0, "C": 1.6}})

    result = reb.apply_triggers(prev, frame)
    assert set(result.index) == {"A", "B", "C"}


def test_rebalancer_reads_hard_thresholds_from_portfolio_root() -> None:
    """Root-level threshold settings should reach the rebalancer."""

    cfg = {
        "portfolio": {
            "z_entry_hard": 1.5,
            "z_exit_hard": -0.5,
            "z_entry_soft": 0.5,
            "z_exit_soft": -0.2,
            "soft_strikes": 1,
            "entry_soft_strikes": 1,
            "entry_eligible_strikes": 1,
            "constraints": {"max_funds": 3},
        }
    }
    reb = Rebalancer(cfg)
    prev = _series({"A": 0.5, "B": 0.5})
    frame = pd.DataFrame({"zscore": {"A": -0.4, "B": -0.6, "C": 1.6, "D": 1.0}})

    result = reb.apply_triggers(prev, frame)
    assert set(result.index) == {"C", "D"}


def test_rebalancer_reads_hard_thresholds_from_loaded_config(
    tmp_path: Path,
) -> None:
    """Config loading should preserve hard thresholds for selection logic."""

    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = load_config(
        {
            "version": "1",
            "data": {
                "csv_path": str(csv_file),
                "date_column": "Date",
                "frequency": "M",
            },
            "preprocessing": {},
            "sample_split": {},
            "metrics": {},
            "export": {},
            "run": {},
            "portfolio": {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.5,
                "transaction_cost_bps": 10,
                "threshold_hold": {
                    "z_exit_soft": -0.2,
                    "soft_strikes": 1,
                    "z_exit_hard": -0.5,
                    "z_entry_hard": 1.5,
                    "z_entry_soft": 0.5,
                    "entry_soft_strikes": 1,
                    "entry_eligible_strikes": 1,
                },
                "constraints": {"max_funds": 3},
            },
            "vol_adjust": {"target_vol": 0.1},
        }
    )

    reb = Rebalancer(cfg.model_dump())
    prev = _series({"A": 0.5, "B": 0.5})
    frame = pd.DataFrame({"zscore": {"A": -0.4, "B": 0.1, "C": 1.0, "D": 1.6}})

    result = reb.apply_triggers(prev, frame)
    assert set(result.index) == {"A", "B", "D"}
    assert "C" not in result


def test_rebalancer_adds_eligible_after_multiple_periods() -> None:
    """Eligible candidates join once they accumulate sufficient strikes."""

    cfg = {
        "portfolio": {
            "threshold_hold": {
                "z_entry_soft": 1.0,
                "entry_soft_strikes": 3,
                "entry_eligible_strikes": 2,
            },
            "constraints": {"max_funds": 5},
        }
    }
    reb = Rebalancer(cfg)
    prev = _series({"A": 1.0})
    rising = pd.DataFrame({"zscore": {"B": 1.2}})

    first = reb.apply_triggers(prev, rising)
    assert list(first.index) == ["A"]  # not enough strikes yet

    second = reb.apply_triggers(first, rising)
    assert set(second.index) == {"A", "B"}


def test_rebalancer_score_prop_weighting_prefers_high_scores() -> None:
    """Score-proportional weighting redistributes using shifted z-scores."""

    cfg = {"portfolio": {"threshold_hold": {"weighting": "score_prop_bayes"}}}
    reb = Rebalancer(cfg)
    prev = _series({"A": 0.5, "B": 0.5})
    scores = pd.DataFrame({"zscore": {"A": -0.5, "B": 2.0}})

    weights = reb.apply_triggers(prev, scores)
    assert pytest.approx(weights.sum(), rel=1e-9) == 1.0
    assert weights.loc["B"] > weights.loc["A"] > 0.0


def test_rebalancer_score_prop_weighting_falls_back_to_equal() -> None:
    """Missing z-score columns should produce equal weights."""

    cfg = {"portfolio": {"threshold_hold": {"weighting": "score_prop_bayes"}}}
    reb = Rebalancer(cfg)
    prev = _series({"A": 0.2, "B": 0.8})
    frame = pd.DataFrame({"alpha": {"A": 0.1, "B": 0.5}})  # no ``zscore`` column

    weights = reb.apply_triggers(prev, frame)
    assert pytest.approx(weights.loc["A"], rel=1e-9) == 0.5
    assert pytest.approx(weights.loc["B"], rel=1e-9) == 0.5


def test_rebalancer_returns_empty_series_for_empty_holdings() -> None:
    """Edge cases with empty holdings should short-circuit."""

    reb = Rebalancer(None)
    prev = pd.Series(dtype=float)
    frame = pd.DataFrame(columns=["zscore"])  # no candidates to add

    weights = reb.apply_triggers(prev, frame)
    assert weights.empty
