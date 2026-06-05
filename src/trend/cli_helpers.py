from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd


def _apply_trend_spec_preset(cfg: Any, preset: Any) -> None:
    """Merge TrendSpec preset parameters into ``cfg`` in-place."""

    payload = preset.as_signal_config()
    if isinstance(cfg, dict):
        existing = cfg.get("signals")
        merged = dict(existing) if isinstance(existing, Mapping) else {}
        merged.update(payload)
        cfg["signals"] = merged
        cfg["trend_spec_preset"] = preset.name
        return

    existing = getattr(cfg, "signals", None)
    if isinstance(existing, Mapping):
        merged = dict(existing)
        merged.update(payload)
    else:
        merged = dict(payload)

    try:
        setattr(cfg, "signals", merged)
    except ValueError:
        object.__setattr__(cfg, "signals", merged)
    try:
        setattr(cfg, "trend_spec_preset", preset.name)
    except ValueError:
        object.__setattr__(cfg, "trend_spec_preset", preset.name)


def _apply_universe_mask(df: pd.DataFrame, mask: pd.DataFrame, *, date_column: str) -> pd.DataFrame:
    """Apply a time-varying membership mask to returns data."""

    if mask.empty:
        return df
    working = df.copy()
    lookup = {str(col).lower(): col for col in working.columns}
    try:
        date_col = lookup[date_column.lower()]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Date column '{date_column}' is missing from the returns data") from exc

    working[date_col] = pd.to_datetime(working[date_col])
    working = working.set_index(date_col)
    aligned_mask = mask.reindex(index=working.index, fill_value=False)

    missing = [col for col in aligned_mask.columns if col not in working.columns]
    if missing:
        preview = ", ".join(missing[:3])
        raise KeyError(
            "Universe members missing from returns data: "
            f"{preview}" + ("..." if len(missing) > 3 else "")
        )

    masked = working.copy()
    masked.loc[:, aligned_mask.columns] = masked.loc[:, aligned_mask.columns].where(aligned_mask)
    masked.reset_index(inplace=True)
    return masked


def _attach_universe_paths(cfg: Any, spec: Any, *, csv_path: str | None) -> None:
    """Persist the selected universe paths onto ``cfg.data`` when possible."""

    logger = logging.getLogger(__name__)
    membership_value = str(spec.membership_path)
    csv_value = csv_path
    data_section = getattr(cfg, "data", None)
    if isinstance(data_section, Mapping):
        merged = dict(data_section)
        merged["universe_membership_path"] = membership_value
        if csv_value:
            merged.setdefault("csv_path", csv_value)
        try:
            setattr(cfg, "data", merged)
        except (AttributeError, TypeError, ValueError):
            object.__setattr__(cfg, "data", merged)
        return

    if data_section is None:
        payload: dict[str, str] = {"universe_membership_path": membership_value}
        if csv_value:
            payload["csv_path"] = csv_value
        try:
            setattr(cfg, "data", payload)
        except (AttributeError, TypeError, ValueError):
            object.__setattr__(cfg, "data", payload)
        return

    try:
        setattr(data_section, "universe_membership_path", membership_value)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug(
            "Unable to attach universe membership path with setattr; trying object.__setattr__: %s",
            exc,
        )
        try:
            object.__setattr__(data_section, "universe_membership_path", membership_value)
        except (AttributeError, TypeError, ValueError) as fallback_exc:
            logger.debug(
                "Unable to attach universe membership path to config data section: %s",
                fallback_exc,
            )
            data_section = None

    if csv_value and data_section is not None:
        try:
            setattr(data_section, "csv_path", csv_value)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "Unable to attach universe csv path with setattr; trying object.__setattr__: %s",
                exc,
            )
            try:
                object.__setattr__(data_section, "csv_path", csv_value)
            except (AttributeError, TypeError, ValueError) as fallback_exc:
                logger.debug(
                    "Unable to attach universe csv path to config data section: %s",
                    fallback_exc,
                )
