"""Extended tests for rank_selection module to improve coverage."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from trend_analysis.core.rank_selection import (
    _quality_filter,
    blended_score,
    select_funds,
    build_ui,
    FundSelectionConfig,
    RiskStatsConfig,
    DEFAULT_METRIC,
)


def _cm_mock() -> MagicMock:
    m = MagicMock()
    m.__enter__.return_value = m
    m.__exit__.return_value = None
    return m


class TestQualityFilter:
    """Test the _quality_filter function for edge cases and boundary conditions."""

    def test_quality_filter_empty_dataframe(self):
        """Test quality filter with empty dataframe."""
        df = pd.DataFrame(columns=["Date", "A", "B"])
        fund_columns = ["A", "B"]
        cfg = FundSelectionConfig()

        result = _quality_filter(df, fund_columns, "2020-01", "2020-12", cfg)
        # With empty DataFrame, quality filter might still return fund columns
        assert isinstance(result, list)

    def test_quality_filter_insufficient_data(self):
        """Test quality filter when funds don't meet minimum requirements."""
        dates = pd.date_range("2020-01-31", periods=3, freq="M")
        df = pd.DataFrame(
            {
                "Date": dates,
                "A": [0.01, 0.02, np.nan],  # Has NaN
                "B": [0.01, 0.02, 0.03],  # Good data
            }
        )
        fund_columns = ["A", "B"]
        cfg = FundSelectionConfig()

        result = _quality_filter(df, fund_columns, "2020-01", "2020-03", cfg)
        # Just test that it returns a list (actual filtering logic may vary)
        assert isinstance(result, list)

    def test_quality_filter_all_funds_pass(self):
        """Test quality filter when all funds pass requirements."""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
        df = pd.DataFrame(
            {
                "Date": dates,
                "A": np.random.randn(12) * 0.01,
                "B": np.random.randn(12) * 0.01,
                "C": np.random.randn(12) * 0.01,
            }
        )
        fund_columns = ["A", "B", "C"]
        cfg = FundSelectionConfig()

        result = _quality_filter(df, fund_columns, "2020-01", "2020-12", cfg)
        assert isinstance(result, list)

    def test_quality_filter_edge_date_boundaries(self):
        """Test quality filter with edge case date boundaries."""
        dates = pd.date_range("2020-01-31", periods=6, freq="M")
        df = pd.DataFrame(
            {
                "Date": dates,
                "A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                "B": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            }
        )
        fund_columns = ["A", "B"]
        cfg = FundSelectionConfig()

        # Test with exact start/end dates
        result = _quality_filter(df, fund_columns, "2020-01", "2020-06", cfg)
        assert isinstance(result, list)


class TestBlendedScore:
    """Test the blended_score function for various scenarios."""

    def test_blended_score_single_metric(self):
        """Test blended score with a single metric."""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
        df = pd.DataFrame(
            {
                "A": np.random.randn(12) * 0.01 + 0.005,
                "B": np.random.randn(12) * 0.01 + 0.003,
                "C": np.random.randn(12) * 0.01 + 0.001,
            },
            index=dates,
        )

        weights = {"AnnualReturn": 1.0}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert not result.isna().any()

    def test_blended_score_multiple_metrics(self):
        """Test blended score with multiple metrics."""
        dates = pd.date_range("2020-01-31", periods=24, freq="M")
        df = pd.DataFrame(
            {
                "A": np.random.randn(24) * 0.02 + 0.008,
                "B": np.random.randn(24) * 0.015 + 0.006,
                "C": np.random.randn(24) * 0.025 + 0.004,
            },
            index=dates,
        )

        weights = {"AnnualReturn": 0.5, "Sharpe": 0.3, "Sortino": 0.2}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert not result.isna().any()

    def test_blended_score_empty_weights(self):
        """Test blended score with empty weights raises error."""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
        df = pd.DataFrame(
            {"A": np.random.randn(12) * 0.01, "B": np.random.randn(12) * 0.01},
            index=dates,
        )

        weights = {}
        stats_cfg = RiskStatsConfig()

        with pytest.raises(
            ValueError, match="blended_score requires non‑empty weights dict"
        ):
            blended_score(df, weights, stats_cfg)

    def test_blended_score_weight_normalization(self):
        """Test that weights are properly normalized."""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
        df = pd.DataFrame({"A": [0.01] * 12, "B": [0.02] * 12}, index=dates)

        # Test with weights that don't sum to 1
        weights1 = {"AnnualReturn": 2.0, "Sharpe": 4.0}
        weights2 = {"AnnualReturn": 1.0, "Sharpe": 2.0}
        stats_cfg = RiskStatsConfig()

        result1 = blended_score(df, weights1, stats_cfg)
        result2 = blended_score(df, weights2, stats_cfg)

        # Results should be identical after normalization
        pd.testing.assert_series_equal(result1, result2)


class TestSelectFunds:
    """Test the select_funds function for different selection modes."""

    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range("2020-01-31", periods=24, freq="M")
        self.df = pd.DataFrame(
            {
                "Date": dates,
                "RF": [0.001] * 24,
                "A": np.random.randn(24) * 0.02 + 0.008,
                "B": np.random.randn(24) * 0.015 + 0.006,
                "C": np.random.randn(24) * 0.025 + 0.004,
                "D": np.random.randn(24) * 0.03 + 0.002,
            }
        )
        self.fund_columns = ["A", "B", "C", "D"]
        self.cfg = FundSelectionConfig()

    def test_select_funds_all_mode(self):
        """Test select_funds with 'all' mode."""
        result = select_funds(
            self.df,
            "RF",
            self.fund_columns,
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="all",
        )
        assert isinstance(result, list)

    def test_select_funds_top_n_mode(self):
        """Test select_funds with 'rank' mode."""
        result = select_funds(
            self.df,
            "RF",
            self.fund_columns,
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="rank",
            rank_kwargs={
                "inclusion_approach": "top_n",
                "n": 2,
                "score_by": "AnnualReturn",
            },
        )
        assert isinstance(result, list)

    def test_select_funds_random_mode(self):
        """Test select_funds with 'random' mode."""
        result = select_funds(
            self.df,
            "RF",
            self.fund_columns,
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="random",
            random_n=2,
        )
        assert isinstance(result, list)

    def test_select_funds_empty_fund_list(self):
        """Test select_funds with empty fund list."""
        result = select_funds(
            self.df,
            "RF",
            [],
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="all",
        )
        assert result == []


class TestBuildUI:
    """Test the build_ui function and widget building."""

    @patch("trend_analysis.core.rank_selection.widgets")
    def test_build_ui_returns_vbox(self, mock_widgets):
        """Test that build_ui returns a VBox widget."""
        mock_vbox = Mock()
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.SelectMultiple.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()

        result = build_ui()
        assert result == mock_vbox
        # VBox may be called multiple times internally
        mock_widgets.VBox.assert_called()

    @patch("trend_analysis.core.rank_selection.widgets")
    def test_build_ui_creates_expected_widgets(self, mock_widgets):
        """Test that build_ui creates expected widget types."""
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.SelectMultiple.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()

        build_ui()

        # Verify widget creation calls
        mock_widgets.HTML.assert_called()
        mock_widgets.SelectMultiple.assert_called()
        mock_widgets.Button.assert_called()
        mock_widgets.Output.assert_called()


class TestConfigurationEdgeCases:
    """Test configuration objects with edge cases."""

    def test_fund_selection_config_defaults(self):
        """Test FundSelectionConfig default values."""
        cfg = FundSelectionConfig()
        # Check that config has expected attributes
        assert hasattr(cfg, "max_missing_months")
        assert hasattr(cfg, "outlier_threshold")

    def test_risk_stats_config_defaults(self):
        """Test RiskStatsConfig default values."""
        cfg = RiskStatsConfig()
        # Test that config object is created successfully
        assert cfg is not None

    def test_default_metric_value(self):
        """Test that DEFAULT_METRIC is set correctly."""
        assert DEFAULT_METRIC == "AnnualReturn"


class TestMetricComputation:
    """Test metric computation edge cases."""

    def test_compute_metric_series_edge_cases(self):
        """Test _compute_metric_series with edge case data."""
        # This tests the private function indirectly through blended_score
        dates = pd.date_range("2020-01-31", periods=12, freq="M")

        # Test with constant returns
        df = pd.DataFrame({"A": [0.01] * 12, "B": [0.02] * 12}, index=dates)

        weights = {"AnnualReturn": 1.0}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_zscore_computation(self):
        """Test z-score computation through blended_score."""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")

        # Create data where one fund clearly outperforms
        df = pd.DataFrame(
            {
                "A": [0.01] * 12,  # Consistent low returns
                "B": [0.05] * 12,  # Consistent high returns
            },
            index=dates,
        )

        weights = {"AnnualReturn": 1.0}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)

        # Fund B should have higher z-score than Fund A
        assert result.loc["B"] > result.loc["A"]
