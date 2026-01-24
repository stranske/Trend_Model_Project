"""Tests for buy_and_hold selection mode in multi-period simulation.

The buy_and_hold mode:
1. Selects funds initially using a configured method (top_n, top_pct, threshold, random)
2. Holds selected funds until their data disappears (they cease to exist)
3. Replaces exited funds using the same selection method

Key regression test case: A fund that is selected initially should remain selected
until its data actually disappears, even if other filters would normally exclude it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.config.legacy import Config
from trend_analysis.multi_period.engine import run as multi_period_run


def _make_test_data_with_disappearing_fund(
    n_funds: int = 10,
    n_periods: int = 84,
    seed: int = 42,
    boost_fund: int | None = 0,
    boost_amount: float = 0.02,
    disappear_fund: int | None = 0,
    disappear_at: int = 48,
) -> pd.DataFrame:
    """Create test returns data with a fund that disappears partway through.

    Parameters
    ----------
    n_funds : int
        Number of funds to create.
    n_periods : int
        Number of monthly periods.
    seed : int
        Random seed for reproducibility.
    boost_fund : int | None
        Fund index to boost returns (to make it rank #1).
    boost_amount : float
        Amount to add to boosted fund's returns.
    disappear_fund : int | None
        Fund index that will disappear (have NaN after disappear_at).
    disappear_at : int
        Period index at which fund disappears.

    Returns
    -------
    pd.DataFrame
        DataFrame with Date column and fund return columns.
    """
    dates = pd.date_range("2018-01-01", periods=n_periods, freq="ME")
    np.random.seed(seed)
    returns_data = {f"Fund_{i}": np.random.randn(n_periods) * 0.02 + 0.005 for i in range(n_funds)}
    returns = pd.DataFrame(returns_data, index=dates)

    if boost_fund is not None:
        returns[f"Fund_{boost_fund}"] = returns[f"Fund_{boost_fund}"] + boost_amount

    if disappear_fund is not None:
        returns.loc[returns.index[disappear_at:], f"Fund_{disappear_fund}"] = np.nan

    return returns.reset_index().rename(columns={"index": "Date"})


def _make_buy_and_hold_config(
    initial_method: str = "top_n",
    n: int = 5,
    pct: float = 0.2,
    threshold: float = 1.0,
    start: str = "2018-01",
    end: str = "2025-01",
) -> Config:
    """Create a Config for buy_and_hold mode testing.

    Parameters
    ----------
    initial_method : str
        Initial selection method: top_n, top_pct, threshold, or random.
    n : int
        Number of funds to select (for top_n mode).
    pct : float
        Percentage of funds to select (for top_pct mode).
    threshold : float
        Z-score threshold (for threshold mode).
    start : str
        Multi-period start date (YYYY-MM format).
    end : str
        Multi-period end date (YYYY-MM format).

    Returns
    -------
    Config
        Configuration for buy_and_hold mode.
    """
    return Config(
        version="1",
        data={
            "allow_risk_free_fallback": True,
            "missing_policy": "ffill",
        },
        preprocessing={},
        vol_adjust={
            "target_vol": 0.1,
            "floor_vol": None,
            "warmup_periods": 12,
            "window": None,
        },
        sample_split={},
        portfolio={
            "policy": "threshold_hold",
            "selection_mode": "buy_and_hold",
            "buy_and_hold": {
                "initial_method": initial_method,
                "n": n,
                "pct": pct,
                "threshold": threshold,
            },
            "rank": {
                "inclusion_approach": "top_n",
                "n": n,
                "score_by": "Sharpe",
            },
        },
        multi_period={
            "start": start,
            "end": end,
            "frequency": "A",
            "in_sample_len": 1,
            "out_sample_len": 1,
        },
        benchmarks={},
        seed=42,
        run={},
        performance={},
        metrics={},
        export={},
    )


class TestBuyAndHoldFundDisappearance:
    """Tests for buy_and_hold mode handling funds that disappear."""

    def test_fund_selected_initially_then_held_until_data_disappears(self) -> None:
        """Regression test: Fund is selected initially and held until data ends.

        This is the key regression test for the buy_and_hold feature:
        - Fund_0 is boosted to rank #1 by Sharpe
        - Fund_0 has data until 2022-01 (index 48), then NaN
        - Fund_0 should be selected in early periods (has data)
        - Fund_0 should NOT be selected in later periods (data disappeared)
        """
        # Create data where Fund_0 ranks #1 but disappears at 2022-01
        df = _make_test_data_with_disappearing_fund(
            n_funds=10,
            n_periods=84,
            seed=42,
            boost_fund=0,
            boost_amount=0.02,
            disappear_fund=0,
            disappear_at=48,  # 2022-01
        )

        cfg = _make_buy_and_hold_config(
            initial_method="top_n",
            n=5,
            start="2018-01",
            end="2025-01",
        )

        results = multi_period_run(cfg, df=df)

        # Should have multiple periods
        assert len(results) >= 4, "Expected at least 4 periods"

        # Fund_0 should be selected in early periods (before it disappears)
        early_periods = [r for r in results if r["period"][0][:4] < "2022"]
        for res in early_periods:
            funds = res.get("selected_funds", [])
            assert "Fund_0" in funds, (
                f"Fund_0 should be selected in period {res['period']} "
                f"(before data disappears). Got: {funds}"
            )

        # Fund_0 should NOT be selected in late periods (after it disappears)
        late_periods = [r for r in results if r["period"][0][:4] >= "2022"]
        for res in late_periods:
            funds = res.get("selected_funds", [])
            if funds:  # Skip empty periods (end of data)
                assert "Fund_0" not in funds, (
                    f"Fund_0 should NOT be selected in period {res['period']} "
                    f"(data has disappeared). Got: {funds}"
                )

    def test_replacement_funds_selected_after_exit(self) -> None:
        """Test that replacement funds are selected when original fund exits."""
        df = _make_test_data_with_disappearing_fund(
            n_funds=10,
            boost_fund=0,
            disappear_fund=0,
            disappear_at=48,
        )

        cfg = _make_buy_and_hold_config(
            initial_method="top_n",
            n=5,
            start="2018-01",
            end="2024-01",
        )

        results = multi_period_run(cfg, df=df)

        # Check that portfolio size is maintained after Fund_0 exits
        for res in results:
            funds = res.get("selected_funds", [])
            if funds:  # Skip empty periods
                # Should have approximately 5 funds (may vary slightly due to constraints)
                assert (
                    len(funds) >= 3
                ), f"Expected at least 3 funds in period {res['period']}, got {len(funds)}"

    def test_buy_and_hold_with_random_initial_selection(self) -> None:
        """Test buy_and_hold with random initial selection."""
        df = _make_test_data_with_disappearing_fund(n_funds=10)

        cfg = _make_buy_and_hold_config(
            initial_method="random",
            n=5,
            start="2018-01",
            end="2022-01",
        )

        results = multi_period_run(cfg, df=df)

        assert len(results) >= 2, "Expected at least 2 periods"

        # First period should have 5 randomly selected funds
        first_funds = results[0].get("selected_funds", [])
        assert len(first_funds) > 0, "First period should have selected funds"

    def test_buy_and_hold_with_top_pct_initial_selection(self) -> None:
        """Test buy_and_hold with top percentage initial selection."""
        df = _make_test_data_with_disappearing_fund(n_funds=10)

        cfg = _make_buy_and_hold_config(
            initial_method="top_pct",
            pct=0.3,  # Top 30% = 3 funds
            start="2018-01",
            end="2022-01",
        )

        results = multi_period_run(cfg, df=df)

        assert len(results) >= 2, "Expected at least 2 periods"
        first_funds = results[0].get("selected_funds", [])
        assert len(first_funds) > 0, "First period should have selected funds"


class TestBuyAndHoldConfigValidation:
    """Tests for buy_and_hold configuration handling."""

    def test_buy_and_hold_requires_threshold_hold_policy(self) -> None:
        """Test that buy_and_hold mode uses threshold_hold engine."""
        df = _make_test_data_with_disappearing_fund(n_funds=10)

        # Config with threshold_hold policy
        cfg_with_policy = _make_buy_and_hold_config()
        results = multi_period_run(cfg_with_policy, df=df)

        # Should get meaningful results
        assert len(results) > 0, "Should get results with threshold_hold policy"

    def test_buy_and_hold_defaults_to_top_n_when_initial_method_missing(self) -> None:
        """Test that missing initial_method defaults to top_n."""
        df = _make_test_data_with_disappearing_fund(n_funds=10)

        cfg = Config(
            version="1",
            data={"allow_risk_free_fallback": True, "missing_policy": "ffill"},
            preprocessing={},
            vol_adjust={"target_vol": 0.1},
            sample_split={},
            portfolio={
                "policy": "threshold_hold",
                "selection_mode": "buy_and_hold",
                "buy_and_hold": {"n": 5},  # No initial_method specified
                "rank": {"n": 5, "score_by": "Sharpe"},
            },
            multi_period={
                "start": "2018-01",
                "end": "2020-01",
                "frequency": "A",
                "in_sample_len": 1,
                "out_sample_len": 1,
            },
            metrics={},
            export={},
            run={},
        )

        results = multi_period_run(cfg, df=df)
        assert len(results) > 0, "Should work with default initial_method"


class TestBuyAndHoldManagerChanges:
    """Tests for manager change logging in buy_and_hold mode."""

    def test_manager_changes_logged_on_fund_exit(self) -> None:
        """Test that manager changes are logged when fund data disappears."""
        df = _make_test_data_with_disappearing_fund(
            boost_fund=0,
            disappear_fund=0,
            disappear_at=48,
        )

        cfg = _make_buy_and_hold_config(start="2018-01", end="2024-01")
        results = multi_period_run(cfg, df=df)

        # Find the period where Fund_0 should exit
        exit_period = None
        for i, res in enumerate(results):
            if i > 0:  # Not first period
                prev_funds = results[i - 1].get("selected_funds", [])
                curr_funds = res.get("selected_funds", [])
                if "Fund_0" in prev_funds and "Fund_0" not in curr_funds:
                    exit_period = res
                    break

        if exit_period is not None:
            changes = exit_period.get("manager_changes", [])
            drop_events = [c for c in changes if c.get("action") == "dropped"]
            fund_0_drops = [c for c in drop_events if c.get("manager") == "Fund_0"]
            assert (
                len(fund_0_drops) > 0 or exit_period.get("selected_funds") == []
            ), f"Expected Fund_0 drop event or empty portfolio in period {exit_period['period']}"


class TestBuyAndHoldIntegration:
    """Integration tests for buy_and_hold with Streamlit config building."""

    def test_streamlit_config_builds_correct_buy_and_hold_config(self) -> None:
        """Test that Streamlit analysis_runner builds correct config."""
        # Import using absolute path to avoid dependency checker confusion
        # This is an integration test for the Streamlit app
        import importlib.util
        import sys
        from pathlib import Path

        # Load the module directly from file path
        module_path = (
            Path(__file__).parents[1] / "streamlit_app" / "components" / "analysis_runner.py"
        )
        if not module_path.exists():
            pytest.skip("Streamlit app not available")

        spec = importlib.util.spec_from_file_location("analysis_runner", module_path)
        if spec is None or spec.loader is None:
            pytest.skip("Could not load analysis_runner module")

        analysis_runner = importlib.util.module_from_spec(spec)
        # Add streamlit_app to path for relative imports
        streamlit_app_path = str(Path(__file__).parents[1] / "streamlit_app")
        if streamlit_app_path not in sys.path:
            sys.path.insert(0, streamlit_app_path)

        try:
            spec.loader.exec_module(analysis_runner)
        except ImportError as e:
            pytest.skip(f"Could not import analysis_runner: {e}")

        AnalysisPayload = analysis_runner.AnalysisPayload
        _build_config = analysis_runner._build_config

        dates = pd.date_range("2018-01-01", periods=84, freq="ME")
        np.random.seed(42)
        returns = pd.DataFrame(
            {f"Fund_{i}": np.random.randn(84) * 0.02 for i in range(10)}, index=dates
        )

        model_state = {
            "selection_approach": "buy_and_hold",
            "buy_hold_initial": "top_n",
            "selection_count": 5,
            "weighting_scheme": "equal",
            "metric_weights": {"sharpe": 100.0},
            "risk_target": 0.1,
        }

        payload = AnalysisPayload(returns=returns, model_state=model_state, benchmark=None)
        cfg = _build_config(payload)

        assert cfg.portfolio.get("policy") == "threshold_hold"
        assert cfg.portfolio.get("selection_mode") == "buy_and_hold"
        assert cfg.portfolio.get("buy_and_hold", {}).get("initial_method") == "top_n"
        assert cfg.portfolio.get("buy_and_hold", {}).get("n") == 5
        assert cfg.data.get("allow_risk_free_fallback") is True


def test_manual_selection_mode_includes_funds_with_partial_data() -> None:
    """Regression test: manual selection should not filter out hired funds due to missing data.

    When threshold_hold fires a fund and hires a replacement, the newly hired fund
    may have had NaN values earlier in the analysis window. The pipeline should
    still include manually-selected funds even if they have partial missing data,
    as long as the columns exist in the data.

    This is a regression test for the bug where portfolio would shrink over time
    because hired funds were being filtered out by the missing data check before
    they could receive weights.
    """
    # Create test data where FundC has NaN in first 6 months but exists thereafter
    dates = pd.date_range("2018-01-01", periods=48, freq="ME")
    rng = np.random.default_rng(42)

    data = {
        "Date": dates,
        "Fund_A": rng.normal(0.01, 0.02, 48),
        "Fund_B": rng.normal(0.01, 0.02, 48),
        # FundC has NaN for first 6 months - simulates a fund that started later
        "Fund_C": [np.nan] * 6 + list(rng.normal(0.01, 0.02, 42)),
    }
    df = pd.DataFrame(data)

    # Configure a simple threshold_hold scenario
    cfg = Config(
        version="1",
        data={
            "allow_risk_free_fallback": True,
        },
        preprocessing={},
        vol_adjust={
            "target_vol": 0.1,
            "floor_vol": None,
            "warmup_periods": 6,
            "window": None,
        },
        sample_split={},
        portfolio={
            "policy": "threshold_hold",
            "selection_mode": "rank",
            "target_n": 3,  # Want all 3 funds
            "min_funds": 3,
            "max_funds": 3,
            "threshold_hold": {
                "z_entry_soft": 0.5,
                "z_exit_soft": -0.5,
            },
            "weighting": {"name": "equal"},
            "constraints": {},
            "rank": {"mode": "top_n", "n": 3, "score_by": "Sharpe"},
        },
        metrics={},
        benchmarks={},
        multi_period={
            # Use a window where FundC has NaN in the first in-sample period
            "start": "2018-01",
            "end": "2019-06",
            "in_sample_len": 4,  # 4 quarters = 1 year
            "out_sample_len": 1,  # 1 quarter
            "frequency": "Q",
        },
        run={},
        export={},
        seed=42,
    )

    results = multi_period_run(cfg, df=df)

    # Check that we got results for all periods
    assert len(results) > 0, "Expected at least one period result"

    # For each period, verify the portfolio wasn't filtered to zero or one fund
    # due to missing data in earlier periods
    for res in results:
        fund_weights = res.get("fund_weights", {})
        # Filter to only non-zero weights
        non_zero_weights = {k: v for k, v in fund_weights.items() if abs(v) > 1e-9}
        # We should have at least 2 funds in portfolio (even if one has partial NaN)
        assert len(non_zero_weights) >= 2, (
            f"Period {res.get('period', '?')} has only {len(non_zero_weights)} fund(s) "
            f"in portfolio - hired funds may be getting filtered by missing data"
        )
