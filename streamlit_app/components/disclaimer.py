"""Disclaimer modal component for the Streamlit app."""

import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, ContextManager, cast

import streamlit as st

LICENSE_URL = os.environ.get(
    "LICENSE_URL", "https://github.com/stranske/Trend_Model_Project/blob/main/LICENSE"
)
SECURITY_URL = os.environ.get(
    "SECURITY_URL",
    "https://github.com/stranske/Trend_Model_Project/blob/main/SECURITY.md",
)


def show_disclaimer() -> bool:
    """Render the disclaimer modal and return acceptance state.

    Returns True if the user has accepted the disclaimer, False otherwise.
    """
    if "disclaimer_accepted" not in st.session_state:
        st.session_state["disclaimer_accepted"] = False

    if not st.session_state["disclaimer_accepted"]:
        modal_attr = getattr(st, "modal", None)
        context_manager: ContextManager[Any]
        if callable(modal_attr):
            modal_callable = cast(Callable[[str], ContextManager[Any]], modal_attr)
            context_manager = modal_callable("Disclaimer")
        else:
            context_manager = nullcontext()

        with context_manager:
            st.markdown(
                f"By continuing you agree to our [License]({LICENSE_URL}) and "
                f"[Security Policy]({SECURITY_URL})."
            )
            if st.checkbox("I understand and accept", key="disclaimer_checkbox"):
                st.session_state["disclaimer_accepted"] = True
                st.rerun()

    return st.session_state["disclaimer_accepted"]
