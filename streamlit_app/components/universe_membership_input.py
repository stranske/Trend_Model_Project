"""Universe membership helpers for the Streamlit UI."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from trend_analysis.universe import load_universe_membership

UNIVERSE_MEMBERSHIP_SESSION_KEY = "universe_membership_path"
UNIVERSE_MEMBERSHIP_SUMMARY_KEY = "universe_membership_summary"
UNIVERSE_MEMBERSHIP_UPLOAD_DIGEST_KEY = "universe_membership_upload_digest"
UNIVERSE_MEMBERSHIP_UPLOAD_WIDGET_VERSION_KEY = "universe_membership_upload_widget_version"


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


def membership_cache_fingerprint(session_state: Mapping[str, Any] | None = None) -> str:
    """Return a stable cache identity for the active membership file, if any."""

    path = resolve_universe_membership_path(session_state)
    if not path:
        return "__none__"
    membership_path = Path(path)
    try:
        digest = hashlib.sha256(membership_path.read_bytes()).hexdigest()[:16]
    except OSError:
        digest = "missing"
    return f"{membership_path}:{digest}"


def invalidate_analysis_for_membership_change() -> None:
    """Clear cached and session analysis results after membership changes."""

    from streamlit_app import state as app_state
    from streamlit_app.components import analysis_runner

    analysis_runner.clear_cached_analysis()
    app_state.clear_analysis_results()


def set_active_universe_membership(
    path: str, summary: Mapping[str, object], session_state: dict[str, Any]
) -> bool:
    """Store an active membership file and report whether it changed."""

    changed = session_state.get(UNIVERSE_MEMBERSHIP_SESSION_KEY) != path
    session_state[UNIVERSE_MEMBERSHIP_SESSION_KEY] = path
    session_state[UNIVERSE_MEMBERSHIP_SUMMARY_KEY] = dict(summary)
    return changed


def clear_active_universe_membership(session_state: dict[str, Any]) -> bool:
    """Clear membership state and reset the uploader's retained file value."""

    changed = bool(session_state.get(UNIVERSE_MEMBERSHIP_SESSION_KEY))
    session_state.pop(UNIVERSE_MEMBERSHIP_SESSION_KEY, None)
    session_state.pop(UNIVERSE_MEMBERSHIP_SUMMARY_KEY, None)
    session_state.pop(UNIVERSE_MEMBERSHIP_UPLOAD_DIGEST_KEY, None)
    session_state[UNIVERSE_MEMBERSHIP_UPLOAD_WIDGET_VERSION_KEY] = (
        int(session_state.get(UNIVERSE_MEMBERSHIP_UPLOAD_WIDGET_VERSION_KEY, 0)) + 1
    )
    return changed


def persist_membership_upload(upload_bytes: bytes, *, upload_dir: Path, filename: str) -> str:
    """Persist an uploaded membership CSV and validate it before returning the path."""

    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".csv"
    target = upload_dir / f"universe-membership-{uuid4().hex}{suffix}"
    target.write_bytes(upload_bytes)
    try:
        validate_membership_file(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return str(target)
