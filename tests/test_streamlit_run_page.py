"""Tests for the enhanced Streamlit Run page functionality."""

import importlib.util
import logging
import sys
from datetime import date, datetime
from importlib.abc import Loader
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch
from importlib.machinery import ModuleSpec

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the functions we want to test
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "streamlit" / "pages"))

from trend_analysis.api import RunResult  # noqa: E402


@pytest.fixture
def _mock_plotting_modules(monkeypatch):
    """Provide lightweight stand-ins for optional heavy dependencies."""
    monkeypatch.setitem(sys.modules, "streamlit", Mock())
    monkeypatch.setitem(sys.modules, "matplotlib", MagicMock())
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", MagicMock())


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


def _spec_from_path(module_name: str, path: Path) -> ModuleSpec:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load spec for {module_name}")
    return spec


def _exec_module(spec: ModuleSpec, module: ModuleType) -> None:
    loader: Loader | None = spec.loader
    if loader is None:
        raise AssertionError(f"Spec loader unavailable for {module.__name__}")
    loader.exec_module(module)


def load_run_page_module(mock_st: MagicMock | None = None):
    """Import the Run page module with a provided Streamlit mock."""

    if mock_st is None:
        mock_st = create_mock_streamlit()

    spec = _spec_from_path(
        "run_page_module",
        Path(__file__).parent.parent / "app" / "streamlit" / "pages" / "03_Run.py",
    )
    run_page = importlib.util.module_from_spec(spec)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        _exec_module(spec, run_page)

    return run_page, mock_st


def setup_analysis_ui(mock_st: MagicMock):
    """Prepare common Streamlit UI mocks used by integration tests."""

    status_placeholder = MagicMock()
    mock_st.empty = MagicMock(return_value=status_placeholder)

    progress_widget = _ctx_mock()
    progress_widget.progress = MagicMock()
    mock_st.progress = MagicMock(return_value=progress_widget)

    log_display = MagicMock()
    log_expander = MagicMock()
    log_expander.empty.return_value = log_display
    mock_st.expander = MagicMock(return_value=log_expander)

    return status_placeholder, progress_widget, log_expander, log_display


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


@pytest.mark.usefixtures("_mock_plotting_modules")
class TestErrorFormatting:
    """Test error message formatting functionality."""

    def test_format_error_message_key_error(self):
        """Test formatting of KeyError."""
        # Import the function directly since we need to test it
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            # We need to import after mocking streamlit
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

            error = KeyError("missing_field")
            result = run_page.format_error_message(error)
            assert "Missing required data field" in result

    def test_format_error_message_date_specific(self):
        """Test formatting of Date-related errors."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

            error = ValueError("Date column not found")
            result = run_page.format_error_message(error)
            assert "Date column" in result
            assert "properly formatted dates" in result

    def test_format_error_message_generic(self):
        """Test formatting of generic errors."""
        run_page, _ = load_run_page_module()

        error = RuntimeError("Some runtime issue")
        result = run_page.format_error_message(error)
        assert "RuntimeError" in result
        assert "Some runtime issue" in result

    def test_format_error_message_sample_split_hint(self):
        """Errors mentioning sample_split should return config guidance."""

        run_page, _ = load_run_page_module()

        message = run_page.format_error_message(ValueError("bad sample_split"))
        assert "Invalid date ranges" in message
        assert "in-sample" in message

    def test_format_error_message_returns_hint(self):
        """Errors mentioning returns should reference data guidance."""

        run_page, _ = load_run_page_module()

        message = run_page.format_error_message(RuntimeError("returns blew up"))
        assert "Invalid returns data format" in message

    def test_format_error_message_config_hint(self):
        """Errors mentioning config provide configuration guidance."""

        run_page, _ = load_run_page_module()

        message = run_page.format_error_message(RuntimeError("config mismatch"))
        assert "Configuration error" in message
        assert "review your analysis parameters" in message


@pytest.mark.usefixtures("_mock_plotting_modules")
class TestConfigCreation:
    """Test configuration creation from session state."""

    def test_create_config_from_session_state_success(self, sample_config):
        """Test successful config creation."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

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

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

            with patch.object(run_page.st, "session_state", {}):
                config = run_page.create_config_from_session_state()
                assert config is None

    def test_create_config_missing_dates(self):
        """Test behavior when start/end dates are missing."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

            incomplete_config = {"risk_target": 1.0}
            with patch.object(
                run_page.st, "session_state", {"sim_config": incomplete_config}
            ):
                config = run_page.create_config_from_session_state()
                assert config is None

    def test_create_config_runtime_error(self, sample_config):
        """Errors while building the config should surface friendly
        messages."""

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)

        mock_st.session_state["sim_config"] = sample_config

        with patch(
            "trend_analysis.config.Config",
            side_effect=RuntimeError("Config creation failed"),
        ):
            config = run_page.create_config_from_session_state()

        assert config is None
        mock_st.error.assert_called()
        message = mock_st.error.call_args[0][0]
        assert "Failed to create configuration" in message


@pytest.mark.usefixtures("_mock_plotting_modules")
class TestDataPreparation:
    """Test data preparation functionality."""

    def test_prepare_returns_data_success(self, sample_returns_data):
        """Test successful data preparation."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

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

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

            with patch.object(run_page.st, "session_state", {}):
                df = run_page.prepare_returns_data()
                assert df is None

    def test_prepare_returns_data_no_date_column(self):
        """Test data preparation when Date column is missing."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

            # Data without Date column
            df_no_date = pd.DataFrame(
                {"Asset_A": [0.01, 0.02, 0.03], "Asset_B": [0.02, -0.01, 0.02]}
            )

            with patch.object(run_page.st, "session_state", {"returns_df": df_no_date}):
                df = run_page.prepare_returns_data()
                # Should still work if there's an index that can be converted
                assert df is not None or run_page.st.error.called

    def test_prepare_returns_data_respects_index_name(self, sample_returns_data):
        """When the index has a name it should become the Date column."""

        df = sample_returns_data.set_index("Date").copy()
        df.index.name = "custom_date"

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["returns_df"] = df

        result = run_page.prepare_returns_data()
        assert result is not None
        assert "Date" in result.columns
        mock_st.error.assert_not_called()

    def test_prepare_returns_data_renames_index_column(self, sample_returns_data):
        """Existing index columns named 'index' should be renamed."""

        df = sample_returns_data.rename(columns={"Date": "index"}).copy()

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["returns_df"] = df

        result = run_page.prepare_returns_data()
        assert result is not None
        assert "Date" in result.columns

    def test_prepare_returns_data_renames_date_like_columns(self, sample_returns_data):
        """Date-like column names should be normalised to 'Date'."""

        df = sample_returns_data.rename(columns={"Date": "TradeDate"}).set_index(
            "TradeDate"
        )

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["returns_df"] = df

        result = run_page.prepare_returns_data()
        assert result is not None
        assert "Date" in result.columns

    def test_prepare_returns_data_errors_when_no_date_like_columns(self):
        """If no date information is found an error should be shown."""

        df = pd.DataFrame({"Asset_A": [0.1, 0.2], "Asset_B": [0.2, 0.3]})
        df.index.name = "row_id"

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["returns_df"] = df

        result = run_page.prepare_returns_data()
        assert result is None
        mock_st.error.assert_called()

    def test_prepare_returns_data_handles_exceptions(self):
        """Unexpected data types should trigger a friendly error message."""

        class BadObject:
            @property
            def columns(self):  # pragma: no cover - property intentionally raises
                raise RuntimeError("Intentional test error: property access failed")

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["returns_df"] = BadObject()

        result = run_page.prepare_returns_data()
        assert result is None
        mock_st.error.assert_called()


@pytest.mark.usefixtures("_mock_plotting_modules")
class TestLogHandler:
    """Test the custom log handler."""

    def test_streamlit_log_handler_creation(self):
        """Test creating the log handler."""
        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

            handler = run_page.StreamlitLogHandler()
            assert handler is not None
            assert handler.log_messages == []

    def test_streamlit_log_handler_emit(self):
        """Test log message emission."""
        import logging

        with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
            import importlib.util

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

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

            spec = _spec_from_path(
                "run_page",
                Path(__file__).parent.parent
                / "app"
                / "streamlit"
                / "pages"
                / "03_Run.py",
            )
            run_page = importlib.util.module_from_spec(spec)
            _exec_module(spec, run_page)

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


@pytest.mark.usefixtures("_mock_plotting_modules")
class TestAnalysisIntegration:
    """Test the full analysis integration."""

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_success(
        self, mock_run_simulation, sample_returns_data, sample_config
    ):
        """Test successful analysis run."""

        mock_result = RunResult(
            metrics=pd.DataFrame({"metric": [1.0, 2.0]}),
            details={"test": "data"},
            seed=42,
            environment={"python": "3"},
        )

        def _side_effect(config, returns_df):
            logging.getLogger("trend_analysis").info("analysis completed")
            return mock_result

        mock_run_simulation.side_effect = _side_effect

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state.update(
            {"returns_df": sample_returns_data, "sim_config": sample_config}
        )

        _, _, _, log_display = setup_analysis_ui(mock_st)

        result = run_page.run_analysis_with_progress()

        assert result is mock_result
        assert isinstance(result, RunResult)
        mock_run_simulation.assert_called_once()
        log_display.code.assert_called()

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_failure(
        self, mock_run_simulation, sample_returns_data, sample_config
    ):
        """Test analysis run with failure."""

        def _raise_error(config, returns_df):
            logging.getLogger("trend_analysis").info("about to fail")
            raise ValueError("Test error")

        mock_run_simulation.side_effect = _raise_error

        mock_st = create_mock_streamlit()
        mock_st.code = MagicMock()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state.update(
            {"returns_df": sample_returns_data, "sim_config": sample_config}
        )

        _, _, log_expander, log_display = setup_analysis_ui(mock_st)

        fallback_expander = MagicMock()
        fallback_expander.code.side_effect = RuntimeError("expander broken")
        mock_st.expander.side_effect = [log_expander, fallback_expander]

        result = run_page.run_analysis_with_progress()

        assert result is None
        mock_st.error.assert_called()
        log_display.code.assert_called()
        mock_st.code.assert_called()

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_returns_none_when_config_missing(
        self, mock_run_simulation, sample_returns_data
    ):
        """If the configuration cannot be created the run should abort
        early."""

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["returns_df"] = sample_returns_data

        setup_analysis_ui(mock_st)

        with patch.object(
            run_page, "create_config_from_session_state", return_value=None
        ):
            result = run_page.run_analysis_with_progress()

        assert result is None
        mock_run_simulation.assert_not_called()

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_returns_none_when_data_missing(
        self, mock_run_simulation, sample_config
    ):
        """Missing returns data should also abort the run."""

        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["sim_config"] = sample_config

        setup_analysis_ui(mock_st)

        with patch.object(
            run_page, "create_config_from_session_state", return_value=object()
        ):
            with patch.object(run_page, "prepare_returns_data", return_value=None):
                result = run_page.run_analysis_with_progress()

        assert result is None
        mock_run_simulation.assert_not_called()

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_handles_empty_returns(
        self, mock_run_simulation, sample_config
    ):
        """Empty data should raise a user-visible validation error."""

        mock_st = create_mock_streamlit()
        mock_st.code = MagicMock()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["sim_config"] = sample_config

        setup_analysis_ui(mock_st)

        empty_df = pd.DataFrame({"Date": []})

        with patch.object(
            run_page, "create_config_from_session_state", return_value=object()
        ):
            with patch.object(run_page, "prepare_returns_data", return_value=empty_df):
                result = run_page.run_analysis_with_progress()

        assert result is None
        mock_st.error.assert_called()
        mock_run_simulation.assert_not_called()

    @patch("trend_analysis.api.run_simulation")
    def test_run_analysis_with_progress_handles_missing_date_column(
        self, mock_run_simulation, sample_config
    ):
        """Data without a Date column should surface a clear error."""

        mock_st = create_mock_streamlit()
        mock_st.code = MagicMock()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.session_state["sim_config"] = sample_config

        setup_analysis_ui(mock_st)

        df_no_date = pd.DataFrame({"Value": [0.1, 0.2]})

        with patch.object(
            run_page, "create_config_from_session_state", return_value=object()
        ):
            with patch.object(
                run_page, "prepare_returns_data", return_value=df_no_date
            ):
                result = run_page.run_analysis_with_progress()

        assert result is None
        mock_st.error.assert_called()
        mock_run_simulation.assert_not_called()


@pytest.mark.usefixtures("_mock_plotting_modules")
class TestRunPageMain:
    """Tests covering the Streamlit Run page top-level UI flow."""

    def test_main_requires_returns_data(self, sample_config):
        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.title = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.info = MagicMock()
        mock_st.session_state.clear()

        run_page.main()

        mock_st.warning.assert_called_once()
        mock_st.info.assert_called()

    def test_main_requires_configuration(self, sample_returns_data):
        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)
        mock_st.title = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.info = MagicMock()
        mock_st.session_state["returns_df"] = sample_returns_data

        run_page.main()

        assert mock_st.warning.call_count == 1
        mock_st.info.assert_called()

    def test_main_runs_analysis_successfully(self, sample_returns_data, sample_config):
        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)

        columns = (_ctx_mock(), _ctx_mock())
        for col in columns:
            col.info = MagicMock()
            col.write = MagicMock()

        mock_st.title = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.info = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.success = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.button = MagicMock(side_effect=[True, False])
        mock_st.spinner = MagicMock(return_value=_ctx_mock())
        mock_st.columns = MagicMock(return_value=columns)

        mock_st.session_state.update(
            {"returns_df": sample_returns_data, "sim_config": sample_config}
        )

        metrics_df = pd.DataFrame({"metric": [1.0]})
        run_result = RunResult(metrics=metrics_df, details={}, seed=1, environment={})

        with patch.object(
            run_page, "run_analysis_with_progress", return_value=run_result
        ) as mock_runner:
            run_page.main()

        mock_runner.assert_called_once()
        mock_st.success.assert_called()
        mock_st.dataframe.assert_called()
        mock_st.markdown.assert_called()
        mock_st.caption.assert_called()
        assert "sim_results" in mock_st.session_state
        assert "last_run_timestamp" in mock_st.session_state

    def test_main_shows_info_when_metrics_empty(
        self, sample_returns_data, sample_config
    ):
        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)

        columns = (_ctx_mock(), _ctx_mock())
        mock_st.columns = MagicMock(return_value=columns)
        mock_st.title = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.info = MagicMock()
        mock_st.success = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.button = MagicMock(side_effect=[True, False])
        mock_st.spinner = MagicMock(return_value=_ctx_mock())
        mock_st.markdown = MagicMock()

        mock_st.session_state.update(
            {"returns_df": sample_returns_data, "sim_config": sample_config}
        )

        empty_result = RunResult(
            metrics=pd.DataFrame(), details={}, seed=1, environment={}
        )

        with patch.object(
            run_page, "run_analysis_with_progress", return_value=empty_result
        ):
            run_page.main()

        mock_st.info.assert_any_call("No metrics to display.")

    def test_main_clear_results_reruns_app(self, sample_returns_data, sample_config):
        mock_st = create_mock_streamlit()
        run_page, mock_st = load_run_page_module(mock_st)

        columns = (_ctx_mock(), _ctx_mock())
        mock_st.columns = MagicMock(return_value=columns)
        mock_st.title = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.info = MagicMock()
        mock_st.success = MagicMock()
        mock_st.rerun = MagicMock()
        mock_st.button = MagicMock(side_effect=[False, True])

        mock_st.session_state.update(
            {
                "returns_df": sample_returns_data,
                "sim_config": sample_config,
                "sim_results": RunResult(
                    metrics=pd.DataFrame({"metric": [1.0]}),
                    details={},
                    seed=1,
                    environment={},
                ),
                "last_run_timestamp": datetime.now(),
            }
        )

        run_page.main()

        mock_st.success.assert_called_with("Results cleared!")
        mock_st.rerun.assert_called_once()
        assert "sim_results" not in mock_st.session_state
        assert "last_run_timestamp" not in mock_st.session_state


@pytest.mark.usefixtures("_mock_plotting_modules")
def test_smoke_test_imports():
    """Smoke test to ensure all imports work."""
    with patch.dict("sys.modules", {"streamlit": create_mock_streamlit()}):
        import importlib.util

        spec = _spec_from_path(
            "run_page",
            Path(__file__).parent.parent / "app" / "streamlit" / "pages" / "03_Run.py",
        )
    run_page = importlib.util.module_from_spec(spec)
    _exec_module(spec, run_page)

    # Check that key functions are available
    assert hasattr(run_page, "format_error_message")
    assert hasattr(run_page, "create_config_from_session_state")
    assert hasattr(run_page, "prepare_returns_data")
    assert hasattr(run_page, "run_analysis_with_progress")
    assert hasattr(run_page, "StreamlitLogHandler")
