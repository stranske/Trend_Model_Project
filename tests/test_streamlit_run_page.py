"""Tests for the enhanced Streamlit Run page functionality."""

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the functions we want to test
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "streamlit" / "pages"))

# Mock external dependencies before importing our module
sys.modules["streamlit"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

from trend_analysis.api import RunResult  # noqa: E402


def _ctx_mock() -> MagicMock:
    """Return a MagicMock that supports context management."""
    m = MagicMock()
    m.__enter__.return_value = m
    m.__exit__.return_value = None
    return m


def create_mock_streamlit():
    """Create a mock streamlit module for testing."""
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.success = MagicMock()
    mock_st.info = MagicMock()
    mock_st.empty = MagicMock()
    mock_st.container = _ctx_mock()
    mock_st.expander = _ctx_mock()
    mock_st.code = MagicMock()
    mock_st.rerun = MagicMock()

    # Make progress return a context-aware mock object
    progress_mock = _ctx_mock()
    progress_mock.progress = MagicMock()
    mock_st.progress = MagicMock(return_value=progress_mock)

    return mock_st


@pytest.fixture
def mock_streamlit():
    """Fixture to provide mocked streamlit."""
    return create_mock_streamlit()


@pytest.fixture
def sample_returns_data():
    """Sample returns data for testing."""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "Asset_A": [0.01, -0.02, 0.03, 0.01, 0.02, -0.01] * 4,
            "Asset_B": [0.02, -0.01, 0.02, -0.01, 0.03, 0.01] * 4,
            "RF": [0.0] * 24,
        }
    )


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "start": date(2021, 1, 1),
        "end": date(2022, 12, 31),
        "lookback_months": 12,
        "risk_target": 1.0,
        "portfolio": {"selection_mode": "all"},
        "benchmarks": {},
        "metrics": {},
        "run": {"monthly_cost": 0.01},
    }


class TestErrorFormatting:
    """Test error message formatting functionality."""

    def test_format_error_message_key_error(self):
        """Test formatting of KeyError."""
        # Import the function directly since we need to test it
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            # We need to import after mocking streamlit
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            error = KeyError("missing_field")
            result = run_page.format_error_message(error)
            assert "Missing required data field" in result

    def test_format_error_message_date_specific(self):
        """Test formatting of Date-related errors."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            error = ValueError("Date column not found")
            result = run_page.format_error_message(error)
            assert "Date column" in result
            assert "properly formatted dates" in result

    def test_format_error_message_generic(self):
        """Test formatting of generic errors."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            error = RuntimeError("Some runtime issue")
            result = run_page.format_error_message(error)
            assert "RuntimeError" in result
            assert "Some runtime issue" in result


class TestConfigCreation:
    """Test configuration creation from session state."""

    def test_create_config_from_session_state_success(self, sample_config):
        """Test successful config creation."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            # Mock streamlit session state
            with patch.object(
                run_page.st, "session_state", {"sim_config": sample_config}
            ):
                config = run_page.create_config_from_session_state()

                assert config is not None
                assert getattr(config, "vol_adjust", None) is not None
                assert config.vol_adjust["target_vol"] == 1.0
                assert "2020-01" in config.sample_split["in_start"]

    def test_create_config_missing_config(self):
        """Test behavior when config is missing."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            with patch.object(run_page.st, "session_state", {}):
                config = run_page.create_config_from_session_state()
                assert config is None

    def test_create_config_missing_dates(self):
        """Test behavior when start/end dates are missing."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            incomplete_config = {"risk_target": 1.0}
            with patch.object(
                run_page.st, "session_state", {"sim_config": incomplete_config}
            ):
                config = run_page.create_config_from_session_state()
                assert config is None


class TestDataPreparation:
    """Test data preparation functionality."""

    def test_prepare_returns_data_success(self, sample_returns_data):
        """Test successful data preparation."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            with patch.object(
                run_page.st, "session_state", {"returns_df": sample_returns_data}
            ):
                df = run_page.prepare_returns_data()

                assert df is not None
                assert "Date" in df.columns
                assert len(df) == 24

    def test_prepare_returns_data_missing(self):
        """Test behavior when data is missing."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            with patch.object(run_page.st, "session_state", {}):
                df = run_page.prepare_returns_data()
                assert df is None

    def test_prepare_returns_data_no_date_column(self):
        """Test data preparation when Date column is missing."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            # Data without Date column
            df_no_date = pd.DataFrame(
                {"Asset_A": [0.01, 0.02, 0.03], "Asset_B": [0.02, -0.01, 0.02]}
            )

            with patch.object(run_page.st, "session_state", {"returns_df": df_no_date}):
                df = run_page.prepare_returns_data()
                # Should still work if there's an index that can be converted
                assert df is not None or run_page.st.error.called


class TestLogHandler:
    """Test the custom log handler."""

    def test_streamlit_log_handler_creation(self):
        """Test creating the log handler."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            handler = run_page.StreamlitLogHandler()
            assert handler is not None
            assert handler.log_messages == []

    def test_streamlit_log_handler_emit(self):
        """Test log message emission."""
        import logging

        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            handler = run_page.StreamlitLogHandler()

            # Create a log record
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            handler.emit(record)
            logs = handler.get_logs()

            assert len(logs) == 1
            assert logs[0]["level"] == "INFO"
            assert "Test message" in logs[0]["message"]

    def test_streamlit_log_handler_clear(self):
        """Test clearing log messages."""
        import logging

        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            handler = run_page.StreamlitLogHandler()

            # Add a log message
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

            assert len(handler.get_logs()) == 1

            handler.clear_logs()
            assert len(handler.get_logs()) == 0


class TestAnalysisIntegration:
    """Test the full analysis integration."""

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_success(
        self, mock_run_simulation, sample_returns_data, sample_config
    ):
        """Test successful analysis run."""
        # Create mock result
        mock_result = RunResult(
            metrics=pd.DataFrame({"metric": [1.0, 2.0]}),
            details={"test": "data"},
            seed=42,
            environment={"python": "3"},
        )
        mock_run_simulation.return_value = mock_result

        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            # Mock session state with required data
            session_state = {
                "returns_df": sample_returns_data,
                "sim_config": sample_config,
            }

            with patch.object(run_page.st, "session_state", session_state):
                # Mock streamlit UI elements
                with patch.object(run_page.st, "container", return_value=_ctx_mock()):
                    with patch.object(
                        run_page.st, "progress", return_value=_ctx_mock()
                    ):
                        with patch.object(
                            run_page.st, "empty", return_value=_ctx_mock()
                        ):
                            result = run_page.run_analysis_with_progress()

                assert result is not None
                assert isinstance(result, RunResult)
                mock_run_simulation.assert_called_once()

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_failure(
        self, mock_run_simulation, sample_returns_data, sample_config
    ):
        """Test analysis run with failure."""
        # Make run_simulation raise an exception
        mock_run_simulation.side_effect = ValueError("Test error")

        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            assert spec is not None and spec.loader is not None
            run_page = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_page)

            session_state = {
                "returns_df": sample_returns_data,
                "sim_config": sample_config,
            }

            with patch.object(run_page.st, "session_state", session_state):
                with patch.object(run_page.st, "container", return_value=_ctx_mock()):
                    with patch.object(
                        run_page.st, "progress", return_value=_ctx_mock()
                    ):
                        with patch.object(
                            run_page.st, "empty", return_value=_ctx_mock()
                        ):
                            with patch.object(run_page.st, "error") as mock_error:
                                with patch.object(
                                    run_page.st, "expander", return_value=_ctx_mock()
                                ):
                                    result = run_page.run_analysis_with_progress()

                assert result is None
                mock_error.assert_called()


def test_smoke_test_imports():
    """Smoke test to ensure all imports work."""
    with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_page",
            Path(__file__).parent.parent / "app" / "streamlit" / "pages" / "03_Run.py",
        )
    assert spec is not None and spec.loader is not None
    run_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_page)

    # Check that key functions are available
    assert hasattr(run_page, "format_error_message")
    assert hasattr(run_page, "create_config_from_session_state")
    assert hasattr(run_page, "prepare_returns_data")
    assert hasattr(run_page, "run_analysis_with_progress")
    assert hasattr(run_page, "StreamlitLogHandler")
