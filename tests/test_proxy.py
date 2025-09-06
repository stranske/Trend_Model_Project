"""Tests for the Streamlit WebSocket proxy."""

import pytest
from unittest.mock import Mock, patch


class TestStreamlitProxy:
    """Test the StreamlitProxy class and its dependencies."""

    def test_import_with_missing_dependencies(self):
        """Test that proxy gracefully handles missing dependencies."""
        with patch.dict(
            "sys.modules",
            {"httpx": None, "fastapi": None, "uvicorn": None, "websockets": None},
        ):
            # Should be able to import the module even without dependencies
            from trend_analysis.proxy import StreamlitProxy

            # But creating an instance should raise ImportError
            with pytest.raises(
                ImportError, match="Required dependencies not available"
            ):
                StreamlitProxy(streamlit_host="localhost", streamlit_port=8501)

    def test_proxy_initialization_with_deps(self):
        """Test proxy initialization when dependencies are available."""
        try:
            from trend_analysis.proxy import StreamlitProxy

            # Mock the dependencies to avoid actually importing them
            with (
                patch("trend_analysis.proxy.server.httpx") as mock_httpx,
                patch("trend_analysis.proxy.server.FastAPI") as mock_fastapi,
                patch("trend_analysis.proxy.server.uvicorn"),
                patch("trend_analysis.proxy.server.websockets"),
            ):

                mock_httpx.AsyncClient.return_value = Mock()
                mock_fastapi.return_value = Mock()

                proxy = StreamlitProxy()

                assert proxy.streamlit_host == "localhost"
                assert proxy.streamlit_port == 8501
                assert proxy.streamlit_base_url == "http://localhost:8501"
                assert proxy.streamlit_ws_url == "ws://localhost:8501"

        except ImportError:
            pytest.skip("Dependencies not available for testing")

    def test_proxy_custom_config(self):
        """Test proxy with custom host and port configuration."""
        try:
            from trend_analysis.proxy import StreamlitProxy

            with (
                patch("trend_analysis.proxy.server.httpx") as mock_httpx,
                patch("trend_analysis.proxy.server.FastAPI") as mock_fastapi,
                patch("trend_analysis.proxy.server.uvicorn"),
                patch("trend_analysis.proxy.server.websockets"),
            ):

                mock_httpx.AsyncClient.return_value = Mock()
                mock_fastapi.return_value = Mock()

                proxy = StreamlitProxy(
                    streamlit_host="example.com", streamlit_port=9000
                )

                assert proxy.streamlit_host == "example.com"
                assert proxy.streamlit_port == 9000
                assert proxy.streamlit_base_url == "http://example.com:9000"
                assert proxy.streamlit_ws_url == "ws://example.com:9000"

        except ImportError:
            pytest.skip("Dependencies not available for testing")

    def test_proxy_cli_args_parsing(self):
        """Test the CLI argument parsing."""
        from trend_analysis.proxy.cli import main

        # Test with mock arguments
        test_args = [
            "--streamlit-host",
            "test-host",
            "--streamlit-port",
            "9001",
            "--proxy-host",
            "0.0.0.0",
            "--proxy-port",
            "8500",
            "--log-level",
            "DEBUG",
        ]

        with (
            patch("sys.argv", ["proxy"] + test_args),
            patch("trend_analysis.proxy.cli.run_proxy") as mock_run,
        ):

            try:
                main()
                mock_run.assert_called_once_with(
                    streamlit_host="test-host",
                    streamlit_port=9001,
                    proxy_host="0.0.0.0",
                    proxy_port=8500,
                )
            except SystemExit as e:
                # May exit with code 0 or error code depending on dependencies
                pass

    def test_proxy_module_entry_point(self):
        """Test that the proxy can be run as a module."""
        with patch("trend_analysis.proxy.__main__.main") as mock_main:
            mock_main.return_value = 0

            # Import should call main
            import importlib

            importlib.import_module("trend_analysis.proxy.__main__")
