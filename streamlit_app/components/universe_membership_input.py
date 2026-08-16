"""Universe membership helpers for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from trend_analysis.universe import load_universe_membership

UNIVERSE_MEMBERSHIP_SESSION_KEY = "universe_membership_path"
UNIVERSE_MEMBERSHIP_SUMMARY_KEY = "universe_membership_summary"


def resolve_universe_membership_path(session_state: Mapping[str, Any] | None = None) -> str | None:
    """Return the validated on-disk membership path from session state, if any."""

    if session_state is None:
        import streamlit as st

        session_state = st.session_state
    candidate = session_state.get(UNIVERSE_MEMBERSHIP_SESSION_KEY)
    if isinstance(candidate, str) and candidate.strip():
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def validate_membership_file(path: str | Path) -> None:
    """Validate a membership CSV using the canonical loader."""

    load_universe_membership(path)


def summarise_membership(path: str | Path) -> dict[str, object]:
    """Summarise membership windows for UI feedback."""

    membership = load_universe_membership(path)
    funds = sorted(membership.keys())
    exited = [
        fund
        for fund, windows in membership.items()
        if any(window.end_date is not None for window in windows)
    ]
    return {
        "fund_count": len(funds),
        "funds": funds,
        "exited_fund_count": len(exited),
        "exited_funds": sorted(exited),
    }


def persist_membership_upload(upload_bytes: bytes, *, upload_dir: Path, filename: str) -> str:
    """Persist an uploaded membership CSV and validate it before returning the path."""

    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / filename
    target.write_bytes(upload_bytes)
    validate_membership_file(target)
    return str(target)
