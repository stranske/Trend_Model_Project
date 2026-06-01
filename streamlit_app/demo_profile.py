"""Runtime demo profiles for the browser (stlite/Pyodide) and server app.

This module centralises the two public-demo runtime profiles required by the
stlite/Pyodide browser demo (see ``demo/wasm/``):

``presentation_safe``
    Bundled synthetic data only. LLM panels, custom-analysis entry, and
    custom-data/upload surfaces are disabled. The deterministic synthetic-data
    engine still runs end to end. This is the default so a locked-down work PC
    or a presentation never surfaces an LLM/upload control unexpectedly.

``public_llm_demo``
    Bundled synthetic data by default, but the LangChain-backed LLM UI is
    available. No secrets are bundled: any provider key/endpoint must be
    supplied explicitly at runtime.

The resolution helpers are intentionally pure and side-effect free so they can
be unit tested without a Streamlit runtime. ``streamlit`` is imported lazily and
guarded (mirroring :mod:`streamlit_app.state`) so importing this module under
Pyodide, in tests, or in the CLI never requires a live Streamlit session.
"""

from __future__ import annotations

import os
from typing import Optional

PRESENTATION_SAFE = "presentation_safe"
PUBLIC_LLM_DEMO = "public_llm_demo"

#: All recognised profiles, ordered for display (safe default first).
VALID_PROFILES: tuple[str, ...] = (PRESENTATION_SAFE, PUBLIC_LLM_DEMO)

#: The default profile when nothing explicit is configured. Safe by design.
DEFAULT_PROFILE = PRESENTATION_SAFE

#: Env var / query-param / session-state key used to select the active profile.
PROFILE_ENV_VAR = "TREND_DEMO_PROFILE"
PROFILE_QUERY_PARAM = "profile"
PROFILE_SESSION_KEY = "demo_profile_active"

_LABELS = {
    PRESENTATION_SAFE: "Presentation-safe (no LLM, no uploads)",
    PUBLIC_LLM_DEMO: "Public LLM demo (LangChain enabled)",
}


def normalize_profile(value: Optional[str]) -> str:
    """Return a valid profile name, falling back to :data:`DEFAULT_PROFILE`.

    Matching is case-insensitive and tolerant of surrounding whitespace. Any
    unknown or empty value resolves to the safe default rather than raising, so
    a malformed ``?profile=`` query string can never expose LLM/upload surfaces.
    """

    if not value:
        return DEFAULT_PROFILE
    candidate = value.strip().lower()
    if candidate in VALID_PROFILES:
        return candidate
    return DEFAULT_PROFILE


def resolve_profile(
    *,
    session: Optional[str] = None,
    query_param: Optional[str] = None,
    env: Optional[str] = None,
) -> str:
    """Resolve the active profile from the available signals (pure).

    Precedence, highest first:

    1. ``session`` -- an explicit in-session override (e.g. a reviewer toggling
       the sidebar switcher) wins so mode changes take effect immediately.
    2. ``query_param`` -- the ``?profile=`` URL parameter used by deep links and
       the stlite entrypoint.
    3. ``env`` -- the ``TREND_DEMO_PROFILE`` environment variable for
       server/container deployments.
    4. :data:`DEFAULT_PROFILE` -- the safe fallback.

    Each signal is normalised, and an unrecognised value is ignored in favour of
    the next signal so a bad query string cannot silently disable safety.
    """

    for raw in (session, query_param, env):
        if raw is None:
            continue
        text = raw.strip()
        if not text:
            continue
        if text.strip().lower() in VALID_PROFILES:
            return normalize_profile(text)
    return DEFAULT_PROFILE


def _query_param_value(st_module) -> Optional[str]:
    """Best-effort read of the ``?profile=`` query parameter."""

    params = getattr(st_module, "query_params", None)
    if params is None:
        return None
    try:
        value = params.get(PROFILE_QUERY_PARAM)
    except (AttributeError, TypeError):
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def get_active_profile() -> str:
    """Resolve the active profile from the live Streamlit/runtime environment.

    Safe to call without a Streamlit session: if ``streamlit`` is unavailable
    only the environment variable and default are consulted.
    """

    session_value: Optional[str] = None
    query_value: Optional[str] = None
    try:
        import streamlit as st  # noqa: PLC0415 -- lazy, optional dependency
    except ModuleNotFoundError:
        st = None
    if st is not None:
        try:
            session_value = st.session_state.get(PROFILE_SESSION_KEY)
        except (AttributeError, TypeError):
            session_value = None
        query_value = _query_param_value(st)
    return resolve_profile(
        session=session_value,
        query_param=query_value,
        env=os.environ.get(PROFILE_ENV_VAR),
    )


def profile_label(profile: Optional[str] = None) -> str:
    """Return a human-readable label for ``profile`` (or the active one)."""

    resolved = normalize_profile(profile) if profile is not None else get_active_profile()
    return _LABELS.get(resolved, _LABELS[DEFAULT_PROFILE])


def llm_enabled(profile: Optional[str] = None) -> bool:
    """Whether LLM / LangChain surfaces should be shown for ``profile``."""

    resolved = normalize_profile(profile) if profile is not None else get_active_profile()
    return resolved == PUBLIC_LLM_DEMO


def custom_analysis_enabled(profile: Optional[str] = None) -> bool:
    """Whether the custom-analysis flow should be reachable for ``profile``."""

    resolved = normalize_profile(profile) if profile is not None else get_active_profile()
    return resolved != PRESENTATION_SAFE


def uploads_enabled(profile: Optional[str] = None) -> bool:
    """Whether custom-data/upload surfaces should be shown for ``profile``."""

    resolved = normalize_profile(profile) if profile is not None else get_active_profile()
    return resolved != PRESENTATION_SAFE


def set_active_profile(profile: str) -> str:
    """Persist the active profile into Streamlit session state (no-op offline).

    Returns the normalised profile that was stored so callers can update local
    UI state in the same expression.
    """

    resolved = normalize_profile(profile)
    try:
        import streamlit as st  # noqa: PLC0415 -- lazy, optional dependency
    except ModuleNotFoundError:
        return resolved
    try:
        st.session_state[PROFILE_SESSION_KEY] = resolved
    except (AttributeError, TypeError):
        pass
    return resolved


def render_profile_controls(st_module=None) -> str:
    """Render the sidebar mode banner + switcher and return the active profile.

    Lets a reviewer switch between ``presentation_safe`` and ``public_llm_demo``
    and immediately see the visible UI change (an explicit acceptance criterion
    of issue #5343). Safe to call when Streamlit is unavailable -- it simply
    returns the resolved profile without rendering.
    """

    if st_module is None:
        try:
            import streamlit as st_module  # noqa: PLC0415
        except ModuleNotFoundError:
            return get_active_profile()

    active = get_active_profile()
    sidebar = getattr(st_module, "sidebar", st_module)
    try:
        index = VALID_PROFILES.index(active)
    except ValueError:
        index = 0
    selected = sidebar.selectbox(
        "Demo mode",
        options=list(VALID_PROFILES),
        index=index,
        format_func=profile_label,
        help=(
            "presentation_safe disables LLM, custom analysis, and uploads. "
            "public_llm_demo enables the LangChain UI (no secrets bundled)."
        ),
        key="demo_profile_selectbox",
    )
    if selected != active:
        active = set_active_profile(selected)
    if llm_enabled(active):
        sidebar.caption("🔓 LLM/LangChain UI enabled. Provider keys are runtime-only.")
    else:
        sidebar.caption("🔒 Presentation-safe: no LLM, custom analysis, or uploads.")
    return active
