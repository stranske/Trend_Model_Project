"""Export helpers for trend analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

Formatter = Callable[[pd.DataFrame], pd.DataFrame]


def _ensure_dir(path: Path) -> None:
    """Create parent directories for the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_format(df: pd.DataFrame, formatter: Formatter | None) -> pd.DataFrame:
    """Return ``df`` after applying the optional ``formatter``."""
    return formatter(df) if formatter else df


def export_to_excel(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formatter: Formatter | None = None,
) -> None:
    """Export dataframes to an Excel workbook."""
    path = Path(output_path)
    _ensure_dir(path)
    with pd.ExcelWriter(path) as writer:
        # We must iterate over the mapping of DataFrames so each becomes its own
        # sheet. A vectorised approach would obscure the intent here.
        for sheet, df in data.items():
            formatted = _apply_format(df, formatter)
            formatted.to_excel(writer, sheet_name=sheet, index=False)


def export_to_csv(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formatter: Formatter | None = None,
) -> None:
    """Export each dataframe to an individual CSV file using ``output_path`` as prefix."""
    prefix = Path(output_path)
    _ensure_dir(prefix)
    # Looping over the ``data`` dictionary ensures each frame gets its own file.
    for name, df in data.items():
        formatted = _apply_format(df, formatter)
        formatted.to_csv(prefix.with_name(f"{prefix.stem}_{name}.csv"), index=False)


def export_to_json(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formatter: Formatter | None = None,
) -> None:
    """Export each dataframe to an individual JSON file using ``output_path`` as prefix."""
    prefix = Path(output_path)
    _ensure_dir(prefix)
    # Iterate over the mapping so each DataFrame is written to its own JSON file.
    for name, df in data.items():
        formatted = _apply_format(df, formatter)
        formatted.to_json(
            prefix.with_name(f"{prefix.stem}_{name}.json"), orient="records", indent=2
        )


EXPORTERS: dict[
    str, Callable[[Mapping[str, pd.DataFrame], str, Formatter | None], None]
] = {
    "xlsx": export_to_excel,
    "csv": export_to_csv,
    "json": export_to_json,
}


def export_data(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formats: Iterable[str],
    formatter: Formatter | None = None,
) -> None:
    """Export ``data`` to the specified ``formats``."""
    for fmt in formats:
        exporter = EXPORTERS.get(fmt)
        if exporter is None:
            raise ValueError(f"Unsupported format: {fmt}")
        path = str(Path(output_path).with_suffix(f".{fmt}"))
        exporter(data, path, formatter)


__all__ = [
    "export_to_excel",
    "export_to_csv",
    "export_to_json",
    "export_data",
]
