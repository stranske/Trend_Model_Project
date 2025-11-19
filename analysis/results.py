"""Structured results helpers for reproducible summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import hashlib
import importlib
import importlib.metadata as importlib_metadata

import pandas as pd

__all__ = ["Results", "build_metadata", "compute_universe_fingerprint"]

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_PATH = _ROOT / "Trend Universe Data.csv"
_DEFAULT_MEMBERSHIP_PATH = _ROOT / "Trend Universe Membership.csv"


def _coerce_series(obj: Any) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj.astype(float)
    if isinstance(obj, Mapping):
        return pd.Series(obj, dtype=float)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return pd.Series(list(obj), dtype=float)
    return pd.Series(dtype=float)


def _read_bytes(path: Path | None) -> bytes:
    if path is None:
        return b""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _normalise_membership(path: Path | None, columns: Sequence[str]) -> bytes:
    if path is None or not path.exists():
        return b""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return b""
    available = [col for col in columns if col in frame.columns]
    if not available:
        available = list(frame.columns)
    subset = frame[available].fillna("")
    subset = subset.astype(str)
    subset = subset.sort_values(by=available).reset_index(drop=True)
    return subset.to_csv(index=False).encode("utf-8")


def compute_universe_fingerprint(
    data_path: str | Path | None = _DEFAULT_DATA_PATH,
    membership_path: str | Path | None = _DEFAULT_MEMBERSHIP_PATH,
    *,
    membership_columns: Sequence[str] = ("fund", "effective_date", "end_date"),
) -> str:
    """Return a short hash describing the current dataset inputs."""

    data_bytes = _read_bytes(Path(data_path) if data_path else None)
    membership_bytes = _normalise_membership(
        Path(membership_path) if membership_path else None, membership_columns
    )
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(data_bytes)
    hasher.update(membership_bytes)
    return hasher.hexdigest()[:12]


def _resolve_version(explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    try:
        trend_module = importlib.import_module("trend_analysis")
    except Exception:
        trend_module = None
    if trend_module is not None:
        version = getattr(trend_module, "__version__", None)
        if version:
            return str(version)
    for dist_name in ("trend_analysis", "trend-analysis"):
        try:
            return importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def build_metadata(
    *,
    universe: Sequence[str],
    lookbacks: Mapping[str, Any],
    costs: Mapping[str, Any],
    selected: Sequence[str] | None = None,
    code_version: str | None = None,
    data_path: str | Path | None = _DEFAULT_DATA_PATH,
    membership_path: str | Path | None = _DEFAULT_MEMBERSHIP_PATH,
) -> dict[str, Any]:
    """Return a metadata dictionary capturing run context."""

    universe_members = sorted({str(member) for member in universe})
    selected_members = (
        [str(member) for member in selected]
        if selected is not None
        else []
    )
    lookbacks_payload = {
        "in_sample": {
            "start": lookbacks.get("in_start"),
            "end": lookbacks.get("in_end"),
        },
        "out_sample": {
            "start": lookbacks.get("out_start"),
            "end": lookbacks.get("out_end"),
        },
    }
    cost_map: dict[str, float] = {}
    for key, value in costs.items():
        if value is None:
            continue
        try:
            cost_map[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    metadata: dict[str, Any] = {
        "universe": {
            "members": universe_members,
            "count": len(universe_members),
        },
        "lookbacks": lookbacks_payload,
        "costs": cost_map,
        "code_version": _resolve_version(code_version),
        "fingerprint": compute_universe_fingerprint(
            data_path=data_path, membership_path=membership_path
        ),
        "fingerprint_sources": {
            "data": str(data_path) if data_path else None,
            "membership": str(membership_path) if membership_path else None,
        },
    }
    if selected_members:
        metadata["universe"]["selected"] = selected_members
        metadata["universe"]["selected_count"] = len(selected_members)
    return metadata


@dataclass(slots=True)
class Results:
    """Structured representation of a single analysis run."""

    returns: pd.Series
    weights: pd.Series
    exposures: pd.Series
    turnover: pd.Series
    costs: Mapping[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Results":
        """Build :class:`Results` from a pipeline result mapping."""

        metadata = dict(payload.get("metadata", {}))
        portfolio = None
        for key in (
            "portfolio_equal_weight_combined",
            "portfolio",
            "returns",
        ):
            candidate = payload.get(key)
            if candidate is not None:
                portfolio = candidate
                break
        returns_series = _coerce_series(portfolio)
        weights = _coerce_series(payload.get("fund_weights", {}))

        risk_diag = payload.get("risk_diagnostics", {})
        if isinstance(risk_diag, Mapping):
            exposures = _coerce_series(risk_diag.get("final_weights"))
            turnover = _coerce_series(risk_diag.get("turnover"))
        else:
            exposures = pd.Series(dtype=float)
            turnover = pd.Series(dtype=float)

        costs = metadata.get("costs")
        if not isinstance(costs, Mapping):
            turnover_value = (
                risk_diag.get("turnover_value")
                if isinstance(risk_diag, Mapping)
                else None
            )
            costs = {"turnover_applied": float(turnover_value)} if turnover_value else {}

        return cls(
            returns=returns_series,
            weights=weights,
            exposures=exposures,
            turnover=turnover,
            costs=dict(costs),
            metadata=metadata,
        )

    def fingerprint(self) -> str | None:
        """Return the fingerprint recorded in :attr:`metadata`."""

        value = self.metadata.get("fingerprint")
        return str(value) if value is not None else None
