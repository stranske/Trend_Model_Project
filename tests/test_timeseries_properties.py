"""Property-based tests for time-series robustness.

The generators deliberately introduce missing days, outliers, flat segments,
and staggered starts while keeping the data contracts enforced via
``validate_prices`` (Issue 2) and the monotonic timing guarantees (Issue 3).
"""

from __future__ import annotations

from datetime import timezone
from typing import Any

import numpy as np
import pandas as pd
from hypothesis import given, note, settings
from hypothesis import seed as set_seed
from hypothesis import strategies as st

from data.contracts import coerce_to_utc, validate_prices
from trend_analysis.pipeline import compute_signal, position_from_signal

HYPOTHESIS_SEED = 20240618
HYPOTHESIS_SETTINGS = settings(max_examples=25, deadline=None, derandomize=True)
# Allow up to roughly one sixth of the days to be missing to simulate patchy data
MAX_MISSING_RATIO = 6
# Allow staggered starts up to roughly one eighth of the total length
MAX_STAGGER_RATIO = 8


@st.composite
def _price_frame_strategy(draw: st.DrawFn) -> pd.DataFrame:
    days = draw(st.integers(min_value=30, max_value=120))
    base_index = pd.date_range(pd.Timestamp("2023-01-01", tz=timezone.utc), periods=days, freq="D")

    missing_candidates = draw(
        st.sets(
            st.integers(min_value=0, max_value=days - 1),
            max_size=max(1, days // MAX_MISSING_RATIO),
        )
    )
    keep_positions = [idx for idx in range(days) if idx not in missing_candidates]
    index = base_index[keep_positions]
    if len(index) < 10:
        index = base_index[:10]

    n_assets = draw(st.integers(min_value=1, max_value=4))
    columns: dict[str, pd.Series[Any]] = {}

    for asset_idx in range(n_assets):
        steps = np.array(
            draw(
                st.lists(
                    st.floats(
                        min_value=-0.03,
                        max_value=0.03,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=len(index),
                    max_size=len(index),
                )
            )
        )
        path = 100.0 * np.cumprod(1.0 + steps)

        flat_len = draw(st.integers(min_value=1, max_value=min(5, len(path))))
        flat_len = min(flat_len, len(path))
        flat_start = draw(st.integers(min_value=0, max_value=max(0, len(path) - flat_len)))
        path[flat_start : flat_start + flat_len] = path[flat_start]

        outlier_idx = draw(st.integers(min_value=0, max_value=len(path) - 1))
        outlier_mult = draw(st.sampled_from([0.25, 0.5, 1.5, 2.0, 3.0]))
        path[outlier_idx] = max(path[outlier_idx] * outlier_mult, 1e-3)

        offset = draw(st.integers(min_value=0, max_value=max(1, len(index) // MAX_STAGGER_RATIO)))
        series_index = index[offset:]
        asset_series = pd.Series(path[offset:], index=series_index, name=f"A{asset_idx}")
        columns[asset_series.name] = asset_series

    frame = pd.concat(columns.values(), axis=1)
    frame = frame.sort_index()
    frame = frame.ffill().bfill()
    frame.index = frame.index.tz_convert(timezone.utc)
    frame.attrs["market_data_mode"] = "price"
    return frame


@set_seed(HYPOTHESIS_SEED)
@HYPOTHESIS_SETTINGS
@given(_price_frame_strategy())
def test_price_frame_satisfies_contracts_and_timing(price_frame: pd.DataFrame) -> None:
    validated = validate_prices(coerce_to_utc(price_frame), freq=None)

    note(f"rows={len(validated)} cols={len(validated.columns)}")
    assert validated.index.is_monotonic_increasing
    assert validated.index.tz is timezone.utc
    assert not validated.isna().any().any()


@set_seed(HYPOTHESIS_SEED)
@HYPOTHESIS_SETTINGS
@given(_price_frame_strategy())
def test_pipeline_outputs_are_finite_and_nan_free(price_frame: pd.DataFrame) -> None:
    validated = validate_prices(coerce_to_utc(price_frame), freq=None)

    returns = validated.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    signal_source = pd.DataFrame({"returns": returns.mean(axis=1).astype(float)})
    signal = compute_signal(signal_source, window=5, min_periods=1).fillna(0.0)
    positions = position_from_signal(signal)

    turnover = positions.diff().abs().fillna(0.0)
    transaction_costs = turnover * 0.0005

    note("pipeline stats: min_ret=" f"{returns.min().min():.6f}, max_ret={returns.max().max():.6f}")
    assert not signal.isna().any()
    assert not positions.isna().any()
    assert np.isfinite(turnover.to_numpy()).all()
    assert np.isfinite(transaction_costs.to_numpy()).all()
    assert validated.index.equals(signal.index)
    assert validated.index.is_monotonic_increasing
