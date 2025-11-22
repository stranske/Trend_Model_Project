"""Test rank selection functionality for improved coverage."""

import numpy as np
import pandas as pd
import pytest

from trend_analysis.core import rank_selection as rs


def make_extended_df():
    """Create a more comprehensive test DataFrame."""
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.001] * 12,
            "A": [
                0.02,
                0.03,
                -0.01,
                0.04,
                0.02,
                0.01,
                0.03,
                -0.02,
                0.05,
                0.01,
                0.04,
                0.02,
            ],
            "B": [
                0.01,
                0.02,
                -0.02,
                0.03,
                0.02,
                0.0,
                0.01,
                -0.01,
                0.04,
                0.02,
                0.03,
                0.01,
            ],
            "C": [
                0.005,
                0.01,
                -0.03,
                0.02,
                0.015,
                -0.005,
                0.02,
                -0.015,
                0.03,
                0.01,
                0.025,
                0.005,
            ],
            "D": [
                -0.01,
                0.005,
                -0.04,
                0.01,
                0.01,
                -0.01,
                0.015,
                -0.02,
                0.02,
                0.005,
                0.02,
                0.0,
            ],
        }
    )


class TestApplyTransform:
    """Test _apply_transform function."""

    def test_raw_transform(self):
        """Test raw transform returns original series."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = rs._apply_transform(series, mode="raw")
        pd.testing.assert_series_equal(result, series)

    def test_rank_transform(self):
        """Test rank transform."""
        series = pd.Series([1, 4, 2, 5, 3])
        result = rs._apply_transform(series, mode="rank")
        expected = pd.Series([5.0, 2.0, 4.0, 1.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test_percentile_transform(self):
        """Test percentile transform."""
        series = pd.Series([1, 4, 2, 5, 3])
        result = rs._apply_transform(series, mode="percentile", rank_pct=0.4)
        # Top 40% should be 2 items: 5 and 4
        assert result.notna().sum() == 2
        assert result.iloc[1] == 4  # Second highest
        assert result.iloc[3] == 5  # Highest

    def test_percentile_transform_no_rank_pct(self):
        """Test percentile transform raises error without rank_pct."""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="rank_pct must be set"):
            rs._apply_transform(series, mode="percentile")

    def test_zscore_transform(self):
        """Test z-score transform."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = rs._apply_transform(series, mode="zscore")
        # Z-score should have mean ~0 and std ~1
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 0.2  # More tolerant check

    def test_zscore_transform_with_window(self):
        """Test z-score transform with window."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        result = rs._apply_transform(series, mode="zscore", window=3)
        # Should use last 3 values for mean/std calculation
        recent = series.iloc[-3:]
        expected_mean = recent.mean()
        expected_std = recent.std(ddof=0)
        expected = (series - expected_mean) / expected_std
        pd.testing.assert_series_equal(result, expected)

    def test_unknown_transform_mode(self):
        """Test unknown transform mode raises error."""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="unknown transform mode"):
            rs._apply_transform(series, mode="invalid")


class TestRankSelectFunds:
    """Test rank_select_funds function."""

    def test_top_n_selection(self):
        """Test top_n selection approach."""
        df = make_extended_df()
        in_df = df.loc[df.index[:6], ["A", "B", "C", "D"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        selected = rs.rank_select_funds(
            in_df, cfg, inclusion_approach="top_n", n=2, score_by="AnnualReturn"
        )
        assert len(selected) == 2
        assert isinstance(selected, list)

    def test_top_pct_selection(self):
        """Test top_pct selection approach."""
        df = make_extended_df()
        in_df = df.loc[df.index[:6], ["A", "B", "C", "D"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        selected = rs.rank_select_funds(
            in_df, cfg, inclusion_approach="top_pct", pct=0.5, score_by="AnnualReturn"
        )
        assert len(selected) == 2  # 50% of 4 funds

    def test_threshold_selection(self):
        """Test threshold selection approach."""
        df = make_extended_df()
        in_df = df.loc[df.index[:6], ["A", "B", "C", "D"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        selected = rs.rank_select_funds(
            in_df,
            cfg,
            inclusion_approach="threshold",
            threshold=0.15,
            score_by="AnnualReturn",
        )
        assert isinstance(selected, list)

    def test_blended_score_selection(self):
        """Test blended score selection."""
        df = make_extended_df()
        in_df = df.loc[df.index[:6], ["A", "B", "C", "D"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        weights = {"AnnualReturn": 0.5, "Sharpe": 0.3, "Volatility": 0.2}
        selected = rs.rank_select_funds(
            in_df,
            cfg,
            inclusion_approach="top_n",
            n=2,
            score_by="blended",
            blended_weights=weights,
        )
        assert len(selected) == 2

    def test_invalid_inclusion_approach(self):
        """Test invalid inclusion approach raises error."""
        df = make_extended_df()
        in_df = df.loc[df.index[:3], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        with pytest.raises(ValueError, match="Unknown inclusion_approach"):
            rs.rank_select_funds(
                in_df, cfg, inclusion_approach="invalid", score_by="Sharpe"
            )

    def test_top_n_without_n_parameter(self):
        """Test top_n without n parameter raises error."""
        df = make_extended_df()
        in_df = df.loc[df.index[:3], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        with pytest.raises(ValueError, match="top_n requires parameter n"):
            rs.rank_select_funds(
                in_df, cfg, inclusion_approach="top_n", score_by="Sharpe"
            )

    def test_top_pct_without_pct_parameter(self):
        """Test top_pct without pct parameter raises error."""
        df = make_extended_df()
        in_df = df.loc[df.index[:3], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        with pytest.raises(ValueError, match="top_pct requires 0 < pct"):
            rs.rank_select_funds(
                in_df, cfg, inclusion_approach="top_pct", score_by="Sharpe"
            )

    def test_threshold_without_threshold_parameter(self):
        """Test threshold without threshold parameter raises error."""
        df = make_extended_df()
        in_df = df.loc[df.index[:3], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        with pytest.raises(ValueError, match="threshold approach requires"):
            rs.rank_select_funds(
                in_df, cfg, inclusion_approach="threshold", score_by="Sharpe"
            )


class TestBlendedScore:
    """Test blended_score function."""

    def test_blended_score_basic(self):
        """Test basic blended score calculation."""
        df = make_extended_df()
        in_df = df.loc[df.index[:6], ["A", "B", "C"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        weights = {"AnnualReturn": 0.6, "Sharpe": 0.4}
        result = rs.blended_score(in_df, weights, cfg)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert all(col in result.index for col in ["A", "B", "C"])

    def test_blended_score_empty_weights(self):
        """Test blended score with empty weights raises error."""
        df = make_extended_df()
        in_df = df.loc[df.index[:3], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        with pytest.raises(ValueError, match=r"blended_score requires non[-‑]empty"):
            rs.blended_score(in_df, {}, cfg)


class TestMetricRegistry:
    """Test metric registry functionality."""

    def test_register_metric(self):
        """Test metric registration."""

        @rs.register_metric("test_metric")
        def test_fn(series, **kwargs):
            return series.mean()

        assert "test_metric" in rs.METRIC_REGISTRY
        assert rs.METRIC_REGISTRY["test_metric"] == test_fn

    def test_canonical_metric_list(self):
        """Test canonical metric list conversion."""
        names = ["annual_return", "volatility", "CustomMetric"]
        result = rs.canonical_metric_list(names)
        expected = ["AnnualReturn", "Volatility", "CustomMetric"]
        assert result == expected

    def test_compute_metric_series(self):
        """Test metric series computation."""
        df = make_extended_df()
        in_df = df.loc[df.index[:6], ["A", "B", "C"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        result = rs._compute_metric_series(in_df, "AnnualReturn", cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_compute_metric_series_unknown_metric(self):
        """Test unknown metric raises error."""
        df = make_extended_df()
        in_df = df.loc[df.index[:3], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        with pytest.raises(ValueError, match="Metric 'unknown' not registered"):
            rs._compute_metric_series(in_df, "unknown", cfg)


class TestSelectFunds:
    """Test select_funds function."""

    def test_select_funds_all_mode(self):
        """Test select_funds with 'all' mode."""
        df = make_extended_df()
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config()

        result = rs.select_funds(
            df,
            "RF",
            fund_columns,
            "2020-01",
            "2020-06",
            "2020-07",
            "2020-12",
            cfg,
            "all",
        )
        assert isinstance(result, list)
        assert all(fund in fund_columns for fund in result)

    def test_select_funds_random_mode(self):
        """Test select_funds with 'random' mode."""
        df = make_extended_df()
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config()

        result = rs.select_funds(
            df,
            "RF",
            fund_columns,
            "2020-01",
            "2020-06",
            "2020-07",
            "2020-12",
            cfg,
            "random",
            random_n=2,
        )
        assert len(result) == 2

    def test_select_funds_random_mode_no_n(self):
        """Test select_funds random mode without random_n raises error."""
        df = make_extended_df()
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config()

        with pytest.raises(ValueError, match="random_n must be provided"):
            rs.select_funds(
                df,
                "RF",
                fund_columns,
                "2020-01",
                "2020-06",
                "2020-07",
                "2020-12",
                cfg,
                "random",
            )

    def test_select_funds_rank_mode(self):
        """Test select_funds with 'rank' mode."""
        df = make_extended_df()
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config()
        rank_kwargs = {
            "inclusion_approach": "top_n",
            "n": 2,
            "score_by": "AnnualReturn",
        }

        result = rs.select_funds(
            df,
            "RF",
            fund_columns,
            "2020-01",
            "2020-06",
            "2020-07",
            "2020-12",
            cfg,
            "rank",
            rank_kwargs=rank_kwargs,
        )
        assert len(result) <= 2

    def test_select_funds_rank_mode_no_kwargs(self):
        """Test select_funds rank mode without rank_kwargs raises error."""
        df = make_extended_df()
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config()

        with pytest.raises(ValueError, match="rank mode requires rank_kwargs"):
            rs.select_funds(
                df,
                "RF",
                fund_columns,
                "2020-01",
                "2020-06",
                "2020-07",
                "2020-12",
                cfg,
                "rank",
            )

    def test_select_funds_invalid_mode(self):
        """Test select_funds with invalid mode raises error."""
        df = make_extended_df()
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config()

        with pytest.raises(ValueError, match="Unsupported selection_mode"):
            rs.select_funds(
                df,
                "RF",
                fund_columns,
                "2020-01",
                "2020-06",
                "2020-07",
                "2020-12",
                cfg,
                "invalid",
            )


class TestQualityFilter:
    """Test _quality_filter function."""

    def test_quality_filter_basic(self):
        """Test basic quality filtering."""
        df = make_extended_df()
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config()

        result = rs._quality_filter(df, fund_columns, "2020-01", "2020-12", cfg)
        assert isinstance(result, list)
        assert all(fund in fund_columns for fund in result)

    def test_quality_filter_with_missing_data(self):
        """Test quality filter with missing data."""
        df = make_extended_df()
        # Add missing values to fund C
        df.loc[1:4, "C"] = np.nan
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config(max_missing_months=2)

        result = rs._quality_filter(df, fund_columns, "2020-01", "2020-12", cfg)
        # Fund C should be filtered out due to too many missing values
        assert "C" not in result

    def test_quality_filter_implausible_values(self):
        """Test quality filter with implausible values."""
        df = make_extended_df()
        # Add implausible value to fund D
        df.loc[2, "D"] = 2.0  # 200% return is implausible
        fund_columns = ["A", "B", "C", "D"]
        cfg = rs.default_quality_config(implausible_value_limit=1.0)

        result = rs._quality_filter(df, fund_columns, "2020-01", "2020-12", cfg)
        # Fund D should be filtered out
        assert "D" not in result


class TestQualityFilterEdgeCases:
    """Test quality filter edge cases for better coverage."""

    def test_quality_filter_max_missing_months(self):
        """Test quality filter with max_missing_months threshold."""
        df = make_extended_df()
        # Create a fund with too many missing months
        df.loc[df.index[:8], "C"] = np.nan  # 8 missing months

        cfg = rs.default_quality_config(max_missing_months=5)
        result = rs.quality_filter(df, cfg)

        # C should be excluded due to too many missing months
        assert "C" not in result
        assert "A" in result  # Should still be included

    def test_quality_filter_max_missing_ratio(self):
        """Test quality filter with max_missing_ratio threshold."""
        df = make_extended_df()
        # Create a fund with high missing ratio
        total_len = len(df)
        missing_count = int(total_len * 0.6)  # 60% missing
        df.iloc[:missing_count, df.columns.get_loc("C")] = np.nan

        cfg = rs.default_quality_config(max_missing_ratio=0.5)  # Allow max 50%
        result = rs.quality_filter(df, cfg)

        # C should be excluded due to high missing ratio
        assert "C" not in result
        assert "A" in result  # Should still be included

    def test_quality_filter_implausible_values(self):
        """Test quality filter with implausible value limits."""
        df = make_extended_df()
        # Create implausibly large return
        df.iloc[0, df.columns.get_loc("C")] = 10.0  # 1000% return

        cfg = rs.default_quality_config(implausible_value_limit=1.0)  # Max 100%
        result = rs.quality_filter(df, cfg)

        # C should be excluded due to implausible value
        assert "C" not in result
        assert "A" in result  # Should still be included


class TestBlendedScoreEdgeCases:
    """Test blended score edge cases."""

    def test_blended_score_ascending_metrics(self):
        """Test blended score with ascending metrics (smaller is better)."""
        df = make_extended_df()
        in_df = df.loc[df.index[:10], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        # Test with MaxDrawdown (ascending metric)
        weights = {"MaxDrawdown": 0.6, "Sharpe": 0.4}
        result = rs.blended_score(in_df, weights, cfg)

        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestSelectFundsEdgeCases:
    """Test select_funds function edge cases."""

    def test_select_funds_random_mode_with_seed(self):
        """Test random mode with consistent seeding."""
        df = make_extended_df()

        # Test with explicit random seed for reproducibility
        np.random.seed(42)
        result1 = rs.select_funds(df, "rf", mode="random", n=2)

        np.random.seed(42)
        result2 = rs.select_funds(df, "rf", mode="random", n=2)

        # Should get same result with same seed
        assert result1 == result2

    def test_select_funds_rank_mode_with_quality_filter(self):
        """Test rank mode with quality filtering."""
        df = make_extended_df()
        # Make one fund have implausible values
        df.iloc[0, df.columns.get_loc("C")] = 10.0

        result = rs.select_funds(
            df,
            "rf",
            mode="rank",
            quality_cfg=rs.default_quality_config(implausible_value_limit=1.0),
            inclusion_approach="top_n",
            n=2,
            score_by="Sharpe",
        )

        # C should be filtered out
        assert "C" not in result
        assert len(result) <= 2


class TestRegisterMetricFunction:
    """Test metric registration functionality."""

    def test_register_metric_decorator(self):
        """Test metric registration decorator."""

        # Test registering a custom metric
        @rs.register_metric("TestMetric")
        def test_metric(series, **kwargs):
            return series.mean()

        # Verify it was registered
        assert "TestMetric" in rs.METRIC_REGISTRY

        # Test using the registered metric
        df = make_extended_df()
        in_df = df.loc[df.index[:5], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        result = rs._compute_metric_series(in_df, "TestMetric", cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_canonical_metric_list(self):
        """Test canonical metric list function."""
        metrics = rs.canonical_metric_list()

        assert isinstance(metrics, list)
        assert "AnnualReturn" in metrics
        assert "Sharpe" in metrics
        assert "Volatility" in metrics


class TestZScoreFunction:
    """Test _zscore function edge cases."""

    def test_zscore_with_zero_std(self):
        """Test z-score calculation when standard deviation is zero."""
        # All values the same -> std = 0
        series = pd.Series([0.05, 0.05, 0.05, 0.05])
        result = rs._zscore(series)

        # Should return zeros when std is zero
        assert (result == 0.0).all()

    def test_zscore_with_single_value(self):
        """Test z-score with single value."""
        series = pd.Series([0.05])
        result = rs._zscore(series)

        # Single value should return 0
        assert result.iloc[0] == 0.0


class TestRankSelectFundsAdvanced:
    """Test advanced rank_select_funds scenarios."""

    def test_rank_select_funds_threshold_edge_cases(self):
        """Test threshold selection edge cases."""
        df = make_extended_df()
        in_df = df.loc[df.index[:10], ["A", "B", "C"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        # Test with very high threshold (should get no funds)
        result = rs.rank_select_funds(
            in_df,
            cfg,
            inclusion_approach="threshold",
            threshold=10.0,
            score_by="Sharpe",  # Impossibly high Sharpe
        )
        assert len(result) == 0

        # Test with very low threshold (should get all funds)
        result = rs.rank_select_funds(
            in_df,
            cfg,
            inclusion_approach="threshold",
            threshold=-10.0,
            score_by="Sharpe",  # Very low threshold
        )
        assert len(result) == 3

    def test_rank_select_funds_with_transform_modes(self):
        """Test rank selection with different transform modes."""
        df = make_extended_df()
        in_df = df.loc[df.index[:10], ["A", "B"]]
        cfg = rs.RiskStatsConfig(risk_free=0.0)

        # Test with percentile transform
        result = rs.rank_select_funds(
            in_df,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="Sharpe",
            transform_mode="percentile",
            rank_pct=0.5,
        )
        assert len(result) <= 1

        # Test with zscore transform
        result = rs.rank_select_funds(
            in_df,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="Sharpe",
            transform_mode="zscore",
            zscore_window=5,
        )
        assert len(result) <= 1


# ...existing code...
