import numpy as np
import pandas as pd

from trend_analysis.perf.cache import compute_cov_payload, incremental_cov_update


def _make_df(rows: int = 20, cols: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(rows, cols))
    columns = [f"A{i}" for i in range(cols)]
    return pd.DataFrame(data, columns=columns)


def test_incremental_cov_equivalence():
    df = _make_df(30, 4)
    window = df.iloc[:20]
    payload = compute_cov_payload(window, materialise_aggregates=True)

    # simulate sliding window forward by one row multiple times
    for step in range(5):
        old_row = window.iloc[0].to_numpy()
        new_row = df.iloc[20 + step].to_numpy()
        window = pd.concat([window.iloc[1:], df.iloc[20 + step : 20 + step + 1]])
        # full recompute
        full_payload = compute_cov_payload(window, materialise_aggregates=True)
        # incremental update
        payload = incremental_cov_update(payload, old_row, new_row)
        assert payload.n == full_payload.n
        assert payload.assets == full_payload.assets
        np.testing.assert_allclose(payload.mean, full_payload.mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(payload.cov, full_payload.cov, rtol=1e-10, atol=1e-10)
