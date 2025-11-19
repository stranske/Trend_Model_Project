"""Input helpers for multi-period runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from data.contracts import coerce_to_utc, validate_prices

from ..data import load_csv

__all__ = ["load_prices", "load_membership", "load_benchmarks"]


def _coerce_path(value: Any, *, field: str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise KeyError(f"{field} must be provided")
        path = Path(stripped)
    else:
        raise KeyError(f"{field} must be provided")
    path = path.expanduser()
    if not path.is_absolute():
        path = path.resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"{field} '{path}' does not exist")
    if path.is_dir():
        raise IsADirectoryError(f"{field} '{path}' must point to a file")
    return path


def _data_section(cfg: Any) -> Mapping[str, Any]:
    data = getattr(cfg, "data", None)
    if isinstance(data, Mapping):
        return data
    return {}


def load_prices(cfg: Any) -> pd.DataFrame:
    """Load the returns CSV referenced by ``cfg`` into a validated DataFrame."""

    data = _data_section(cfg)
    csv_path = data.get("csv_path")
    if not csv_path:
        raise KeyError("cfg.data['csv_path'] must be provided")
    resolved = _coerce_path(csv_path, field="data.csv_path")
    missing_policy = data.get("missing_policy")
    if missing_policy is None:
        missing_policy = data.get("nan_policy")
    missing_limit = data.get("missing_limit")
    if missing_limit is None:
        missing_limit = data.get("nan_limit")
    frame = load_csv(
        str(resolved),
        errors="raise",
        missing_policy=missing_policy,
        missing_limit=missing_limit,
    )
    if frame is None:
        raise FileNotFoundError(str(resolved))
    frame = coerce_to_utc(frame)
    freq_code = frame.attrs.get("market_data_frequency_code")
    freq = str(freq_code) if freq_code else "D"
    validate_prices(frame, freq=freq)
    return frame


def load_membership(cfg: Any) -> pd.DataFrame:
    """Return the validated universe membership ledger as a DataFrame."""

    data = _data_section(cfg)
    membership_path = data.get("universe_membership_path")
    columns = ["fund", "effective_date", "end_date"]
    if not membership_path:
        return pd.DataFrame(columns=columns)
    resolved = _coerce_path(membership_path, field="data.universe_membership_path")
    table = pd.read_csv(resolved)
    if table.empty:
        return pd.DataFrame(columns=columns)
    lookup = {str(col).strip().lower(): col for col in table.columns}
    fund_col = lookup.get("fund") or lookup.get("symbol")
    eff_col = lookup.get("effective_date")
    end_col = lookup.get("end_date")
    required_sources = (("fund", fund_col), ("effective_date", eff_col))
    missing = [name for name, col in required_sources if col is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Universe membership file is missing required columns: " f"{joined}"
        )
    assert fund_col is not None
    assert eff_col is not None
    rename: dict[str, str] = {
        fund_col: "fund",
        eff_col: "effective_date",
    }
    if end_col is not None:
        rename[end_col] = "end_date"
    normalised = table.rename(columns=rename)
    if "end_date" not in normalised.columns:
        normalised["end_date"] = pd.NaT
    normalised["fund"] = normalised["fund"].astype(str).str.strip()
    normalised["effective_date"] = pd.to_datetime(normalised["effective_date"])
    normalised["end_date"] = pd.to_datetime(normalised["end_date"])
    if normalised["effective_date"].isna().any():
        raise ValueError(
            "Universe membership entries must include valid effective dates"
        )
    normalised.sort_values(["fund", "effective_date"], inplace=True)
    return normalised.reset_index(drop=True)[columns]


def load_benchmarks(cfg: Any, prices: pd.DataFrame) -> pd.DataFrame:
    """Extract benchmark series referenced by ``cfg`` from ``prices``."""

    benchmarks = getattr(cfg, "benchmarks", None)
    if not isinstance(benchmarks, Mapping) or not benchmarks:
        return pd.DataFrame(columns=["Date"])
    if "Date" not in prices.columns:
        raise KeyError("prices DataFrame must include a 'Date' column for benchmarks")
    frame = pd.DataFrame({"Date": pd.to_datetime(prices["Date"])})
    missing: list[str] = []
    for label, column in benchmarks.items():
        if column not in prices.columns:
            missing.append(str(column))
            continue
        series = pd.to_numeric(prices[column], errors="coerce")
        frame[str(label)] = series
    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise KeyError(f"Benchmark columns missing from price data: {joined}")
    return frame
