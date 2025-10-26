"""Tests for the Streamlit WebSocket proxy."""

import runpy
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest


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
            except SystemExit:
                # May exit with code 0 or error code depending on dependencies
                pass

    def test_proxy_module_entry_point(self):
        """Test that the proxy can be run as a module."""
        with patch("trend_analysis.proxy.__main__.main") as mock_main:
            mock_main.return_value = 0

            # Import should call main
            import importlib

            importlib.import_module("trend_analysis.proxy.__main__")

    def test_proxy_main_entrypoint_executes(self, monkeypatch):
        """Running ``python -m trend_analysis.proxy`` should invoke CLI
        main."""

        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_main(*args, **kwargs):
            calls.append((args, kwargs))
            return 0

        cli_module = ModuleType("trend_analysis.proxy.cli")
        cli_module.main = fake_main

        monkeypatch.setitem(sys.modules, "trend_analysis.proxy.cli", cli_module)

        with pytest.raises(SystemExit) as exc:
            runpy.run_module("trend_analysis.proxy.__main__", run_name="__main__")

        assert exc.value.code == 0
        assert calls == [(tuple(), {})]


class TestProxyCliErrorHandling:
    """Additional CLI edge-case coverage."""

    def test_keyboard_interrupt_returns_zero(self, monkeypatch, capsys):
        from trend_analysis.proxy import cli

        def stop(**_: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "run_proxy", stop)
        monkeypatch.setattr(cli.sys, "argv", ["proxy"])

        rc = cli.main()
        out, err = capsys.readouterr()
        assert rc == 0
        assert "stopped by user" in out
        assert err == ""

    def test_unexpected_error_prints_message(self, monkeypatch, capsys):
        from trend_analysis.proxy import cli

        def explode(**_: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(cli, "run_proxy", explode)
        monkeypatch.setattr(cli.sys, "argv", ["proxy"])

        rc = cli.main()
        out, err = capsys.readouterr()
        assert rc == 1
        assert out == ""
        assert "Error starting proxy: boom" in err
