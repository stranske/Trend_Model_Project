"""Streamlit WebSocket proxy server.

This module provides a proxy server that forwards both HTTP requests and
WebSocket connections to a Streamlit application, solving the issue
where Streamlit's frontend requires WebSocket endpoints like
`/_stcore/stream` for bidirectional updates.
"""

from .server import StreamlitProxy

__all__ = ["StreamlitProxy"]
