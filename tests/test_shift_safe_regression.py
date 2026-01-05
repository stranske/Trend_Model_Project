import math

import pandas as pd

from trend_analysis.pipeline import compute_signal


def test_regression_compute_signal_causal_nan_warmup():
    df = pd.DataFrame({"returns": [0.1, -0.2, 0.05, 0.03, -0.01, 0.04]})
    window = 3
    sig = compute_signal(df, window=window)

    # Warmup: with shift applied after rolling mean, first valid value appears at index == window (0-based)
    # for an implementation that shifts by one AND requires full window. Actual current behavior shows first
    # valid at index 3 for window=3 (so indices 0..2 NaN, index 3 valid). Assert that pattern.
    assert sig.iloc[0:window].isna().all(), sig  # indices 0,1,2
    assert not pd.isna(sig.iloc[window]), sig  # index 3 is first non-NaN

    # Validate each subsequent point uses strictly prior observations
    for i in range(4, len(sig)):
        history = df["returns"].iloc[i - window : i]  # strictly prior window
        expected = history.mean()
        assert math.isclose(sig.iloc[i], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_regression_compute_signal_no_current_row_dependency():
    df = pd.DataFrame({"returns": [0.05, 0.07, -0.02, 0.04, 0.09]})
    sig = compute_signal(df, window=3)

    # Perturb current-row value and assert previously computed signals unchanged
    for i in range(len(df)):
        if i < 3:  # warmup region (NaNs)
            continue
        df_alt = df.copy()
        orig_scalar = df_alt.loc[df_alt.index[i], "returns"]
        orig_val = float(orig_scalar)
        df_alt.loc[df_alt.index[i], "returns"] = orig_val + 10.0  # large shock at current row
        sig_alt = compute_signal(df_alt, window=3)
        pd.testing.assert_series_equal(sig.iloc[:i], sig_alt.iloc[:i])
