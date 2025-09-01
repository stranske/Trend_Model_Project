"""Disclaimer modal component for the Streamlit app."""

import streamlit as st

LICENSE_URL = "https://github.com/stranske/Trend_Model_Project/blob/main/LICENSE"
SECURITY_URL = "https://github.com/stranske/Trend_Model_Project/blob/main/SECURITY.md"


def show_disclaimer() -> bool:
    """Render the disclaimer modal and return acceptance state.

    Returns True if the user has accepted the disclaimer, False otherwise.
    """
    if "disclaimer_accepted" not in st.session_state:
        st.session_state["disclaimer_accepted"] = False

    if not st.session_state["disclaimer_accepted"]:
        with st.modal("Disclaimer"):
            st.markdown(
                f"By continuing you agree to our [License]({LICENSE_URL}) and "
                f"[Security Policy]({SECURITY_URL})."
            )
            if st.checkbox("I understand and accept", key="disclaimer_checkbox"):
                st.session_state["disclaimer_accepted"] = True
                st.rerun()

    return st.session_state["disclaimer_accepted"]
