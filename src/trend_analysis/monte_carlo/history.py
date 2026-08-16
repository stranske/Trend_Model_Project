"""Canonical market-data loading for Monte Carlo price histories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trend_analysis.data import load_csv, load_parquet
from trend_analysis.io.market_data import MarketDataMode


def load_price_history(path: Path) -> pd.DataFrame:
    """Load validated CSV/Parquet history and normalize returns to prices."""

    loader = load_parquet if path.suffix.lower() in {".parquet", ".pq"} else load_csv
    frame = loader(
        str(path),
        errors="raise",
        include_date_column=False,
    )
    if frame is None:  # pragma: no cover - errors="raise" guarantees a result or exception
        raise RuntimeError(f"Market-data loader returned no data for {path}")

    history = frame.copy()
    if history.attrs.get("market_data_mode") == MarketDataMode.RETURNS.value:
        if history.empty:
            raise ValueError("returns data must not be empty")
        if (history <= -1.0).any().any():
            raise ValueError("returns contain values <= -1; cannot convert to prices")
        return (1.0 + history).cumprod() * 100.0
    return history
