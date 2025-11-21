from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import yaml

from .universe import build_membership_mask

__all__ = ["NamedUniverse", "load_universe", "load_universe_spec"]

_SUPPORTED_SUFFIXES: tuple[str, ...] = (".yml", ".yaml")
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "config" / "universe"


@dataclass(frozen=True)
class NamedUniverse:
    """Describes a named universe backed by CSV inputs."""

    key: str
    data_path: Path
    membership_path: Path
    members: tuple[str, ...] | None
    date_column: str
    description: str | None = None
    name: str | None = None


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    raw = Path(value).expanduser()
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            (base_dir / raw).resolve(),
            (base_dir.parent / raw).resolve(),
            Path.cwd().resolve() / raw,
        ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                raise IsADirectoryError(f"Path '{candidate}' must be a file")
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{value}'. Checked: {', '.join(str(c) for c in candidates)}"
    )


def _resolve_universe_file(key: str | Path, *, base_dir: Path | None) -> Path:
    if isinstance(key, Path):
        candidate = key
    else:
        candidate = Path(str(key))
    if candidate.suffix.lower() in _SUPPORTED_SUFFIXES and candidate.exists():
        return candidate.resolve()

    search_dir = base_dir or _DEFAULT_DIR
    for suffix in _SUPPORTED_SUFFIXES:
        candidate = (search_dir / f"{key}{suffix}").resolve()
        if candidate.exists():
            return candidate
    if Path(key).exists():  # Final fallback when caller passed a bare filename
        return Path(key).resolve()
    raise FileNotFoundError(f"Named universe '{key}' was not found under {search_dir}")


def _clean_members(values: Iterable[object] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    cleaned: list[str] = []
    for value in values:
        label = str(value).strip()
        if label:
            cleaned.append(label)
    return tuple(cleaned) if cleaned else None


def load_universe_spec(
    key: str | Path, *, base_dir: Path | None = None
) -> NamedUniverse:
    """Load a named universe definition from ``config/universe``.

    Parameters
    ----------
    key:
        Universe identifier (e.g. ``core``). If a file path is provided it is
        used directly; otherwise the loader resolves ``config/universe/{key}.yml``.
    base_dir:
        Override the lookup directory for tests.
    """

    cfg_path = _resolve_universe_file(key, base_dir=base_dir)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Universe config must contain a mapping at the root level")

    version = raw.get("version")
    if version not in (1, "1"):
        raise ValueError("Universe config version must be 1")

    members = _clean_members(raw.get("members"))
    date_column = str(raw.get("date_column") or "Date")
    data_field = raw.get("data_csv") or raw.get("csv_path")
    membership_field = raw.get("membership_csv") or raw.get("membership_path")
    if not data_field or not membership_field:
        raise ValueError("Universe config must provide data_csv and membership_csv")
    data_path = _resolve_path(data_field, base_dir=cfg_path.parent)
    membership_path = _resolve_path(membership_field, base_dir=cfg_path.parent)
    key_value = str(raw.get("key") or cfg_path.stem)

    return NamedUniverse(
        key=key_value,
        data_path=data_path,
        membership_path=membership_path,
        members=members,
        date_column=date_column,
        description=raw.get("description"),
        name=raw.get("name"),
    )


def _resolve_date_column(prices: pd.DataFrame, candidate: str) -> str:
    lookup = {str(col).lower(): col for col in prices.columns}
    try:
        return str(lookup[candidate.lower()])
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(
            f"Date column '{candidate}' was not found in price data"
        ) from exc


def _filter_membership(
    membership: pd.DataFrame, members: Sequence[str] | None
) -> pd.DataFrame:
    if members is None:
        return membership
    member_set = {str(item) for item in members}
    filtered = membership[membership["fund"].astype(str).isin(member_set)]
    return filtered.reset_index(drop=True)


def load_universe(
    key: str | Path,
    *,
    base_dir: Path | None = None,
    prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, NamedUniverse]:
    """Return a boolean membership mask and the resolved universe spec."""

    spec = load_universe_spec(key, base_dir=base_dir)
    prices_frame = prices.copy() if prices is not None else pd.read_csv(spec.data_path)
    if prices_frame.empty:
        return pd.DataFrame(), spec
    date_col = _resolve_date_column(prices_frame, spec.date_column)
    prices_frame[date_col] = pd.to_datetime(prices_frame[date_col])

    membership = pd.read_csv(spec.membership_path)
    membership = membership.rename(columns={"symbol": "fund"})
    if "fund" not in membership.columns:
        raise ValueError("Universe membership file must include a 'fund' column")
    membership = _filter_membership(membership, spec.members)

    mask = build_membership_mask(prices_frame[date_col], membership)
    ordered_cols: Sequence[str]
    if spec.members:
        ordered_cols = spec.members
    else:
        ordered_cols = sorted(membership["fund"].astype(str).unique())
    if ordered_cols:
        mask = mask.reindex(columns=ordered_cols, fill_value=False)
    resolved_spec = replace(spec, date_column=date_col)
    return mask, resolved_spec
