"""Test the _is_zero_everywhere helper function extracted in issue #683."""

import numpy as np
import pandas as pd

from trend_analysis.metrics import _is_zero_everywhere


class TestIsZeroEverywhere:
    """Test the _is_zero_everywhere helper function."""

    def test_scalar_values(self):
        """Test _is_zero_everywhere with scalar values."""
        # Test integer zero
        assert _is_zero_everywhere(0) is True
        assert _is_zero_everywhere(1) is False
        assert _is_zero_everywhere(-1) is False

        # Test float zero
        assert _is_zero_everywhere(0.0) is True
        assert _is_zero_everywhere(1.0) is False
        assert _is_zero_everywhere(-1.0) is False
        assert _is_zero_everywhere(0.1) is False

        # Test edge cases with numerical tolerance
        assert _is_zero_everywhere(1e-16) is True  # Within tolerance, treated as zero
        assert (
            _is_zero_everywhere(1e-13) is False
        )  # Outside default tolerance, not zero
        # The tolerance can be overridden for scalar comparisons
        assert _is_zero_everywhere(1e-13, tol=1e-12) is True

        # Works with NumPy scalar types and still returns a Python ``bool``
        np_value = np.float64(1e-13)
        result = _is_zero_everywhere(np_value, tol=1e-12)
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_array_and_non_numeric(self):
        """NumPy arrays use element-wise tolerance; non-numeric raises fallback."""

        array_result = _is_zero_everywhere(np.array([5e-16, -3e-16, 4e-16]))
        assert array_result is True
        assert isinstance(array_result, bool)

        class NotNumeric:
            def __abs__(self) -> int:  # pragma: no cover - invoked indirectly
                raise TypeError("cannot take abs")

        assert _is_zero_everywhere(NotNumeric()) is False
        assert _is_zero_everywhere("not-a-number") is False

    def test_pandas_series(self):
        """Test _is_zero_everywhere with pandas Series."""
        # Test all-zero series
        zero_series = pd.Series([0, 0, 0])
        assert _is_zero_everywhere(zero_series) is True

        # Test series with mixed values
        mixed_series = pd.Series([0, 1, 0])
        assert _is_zero_everywhere(mixed_series) is False

        # Test all-nonzero series
        nonzero_series = pd.Series([1, 2, 3])
        assert _is_zero_everywhere(nonzero_series) is False

        # Test float series
        float_zero_series = pd.Series([0.0, 0.0, 0.0])
        assert _is_zero_everywhere(float_zero_series) is True

        float_mixed_series = pd.Series([0.0, 0.1, 0.0])
        assert _is_zero_everywhere(float_mixed_series) is False

    def test_pandas_dataframe(self):
        """Test _is_zero_everywhere with pandas DataFrame."""
        # Test all-zero DataFrame
        zero_df = pd.DataFrame([[0, 0], [0, 0]])
        assert _is_zero_everywhere(zero_df) is True

        # Test DataFrame with mixed values
        mixed_df = pd.DataFrame([[0, 1], [0, 0]])
        assert _is_zero_everywhere(mixed_df) is False

        # Test all-nonzero DataFrame
        nonzero_df = pd.DataFrame([[1, 2], [3, 4]])
        assert _is_zero_everywhere(nonzero_df) is False

        # Test float DataFrame
        float_zero_df = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]])
        assert _is_zero_everywhere(float_zero_df) is True

    def test_empty_containers(self):
        """Test _is_zero_everywhere with empty pandas containers."""
        # Empty series
        empty_series = pd.Series([], dtype=float)
        assert _is_zero_everywhere(empty_series) is True

        # Empty DataFrame
        empty_df = pd.DataFrame()
        assert _is_zero_everywhere(empty_df) is True

    def test_nan_values(self):
        """Test _is_zero_everywhere with NaN values."""
        # Series with NaN should not be zero everywhere
        nan_series = pd.Series([0, np.nan, 0])
        assert _is_zero_everywhere(nan_series) is False

        # DataFrame with NaN should not be zero everywhere
        nan_df = pd.DataFrame([[0, np.nan], [0, 0]])
        assert _is_zero_everywhere(nan_df) is False

    def test_integration_with_metrics(self):
        """Test that the helper function integrates correctly with metric
        functions."""
        # This test verifies that the refactored functions still work
        from trend_analysis.metrics import (
            information_ratio,
            sharpe_ratio,
            sortino_ratio,
        )

        # Create test data that would trigger the zero-check
        zero_returns = pd.Series([0, 0, 0])

        # These should return NaN when volatility/std is zero
        sharpe_result = sharpe_ratio(zero_returns)
        sortino_result = sortino_ratio(zero_returns)
        info_result = information_ratio(zero_returns)

        # All should return NaN due to zero volatility/std
        assert pd.isna(sharpe_result)
        assert pd.isna(sortino_result)
        assert pd.isna(info_result)
