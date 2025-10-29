"""Tests for the GUI app module to improve coverage."""

import asyncio
import contextlib
import importlib
import pickle
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable
from unittest.mock import ANY, MagicMock, Mock, patch

import pandas as pd
import pytest

import trend_analysis.gui.app as app_module
import yaml
from trend_analysis.gui.app import (
    _build_rank_options,
    _build_step0,
    launch,
    load_state,
    save_state,
)
from trend_analysis.gui.store import ParamStore


def _cm_mock() -> MagicMock:
    m = MagicMock()
    m.__enter__.return_value = m
    m.__exit__.return_value = None
    return m


class TestLoadSaveState:
    """Test state loading and saving functionality."""

    def test_load_state_empty(self):
        """Test loading state when no file exists."""
        with patch("trend_analysis.gui.app.STATE_FILE") as mock_file:
            mock_file.exists.return_value = False
            mock_file.open.return_value = _cm_mock()
            store = load_state()
            assert isinstance(store, ParamStore)

    def test_load_state_valid_file(self):
        """Test loading state from valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            test_data = {"test_key": "test_value"}
            yaml.safe_dump(test_data, f)
            f.flush()

            with patch("trend_analysis.gui.app.STATE_FILE", Path(f.name)):
                store = load_state()
                assert isinstance(store, ParamStore)

    def test_load_state_malformed_file(self):
        """Test loading state from malformed file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("invalid: yaml: content: {")
            f.flush()

            with patch("trend_analysis.gui.app.STATE_FILE", Path(f.name)):
                with patch("warnings.warn") as mock_warn:
                    store = load_state()
                    assert isinstance(store, ParamStore)
                    mock_warn.assert_called()

    def test_save_state(self):
        """Test saving state to file."""
        store = ParamStore()
        store.cfg = {"test": "value"}

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "test_state.yml"
            weight_file = state_file.with_suffix(".pkl")

            with patch("trend_analysis.gui.app.STATE_FILE", state_file):
                with patch("trend_analysis.gui.app.WEIGHT_STATE_FILE", weight_file):
                    save_state(store)
                    assert state_file.exists()

    def test_save_state_with_weight_state(self):
        """Test saving state with weight data."""
        store = ParamStore()
        store.cfg = {"test": "value"}
        store.weight_state = {"weights": [1, 2, 3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "test_state.yml"
            weight_file = state_file.with_suffix(".pkl")

            with patch("trend_analysis.gui.app.STATE_FILE", state_file):
                with patch("trend_analysis.gui.app.WEIGHT_STATE_FILE", weight_file):
                    save_state(store)
                    assert state_file.exists()
                    assert weight_file.exists()


@pytest.fixture()
def reset_version_cache():
    """Ensure the cached default version does not leak between tests."""

    original = getattr(app_module, "_DEFAULT_VERSION_CACHE", None)
    app_module._DEFAULT_VERSION_CACHE = None
    try:
        yield
    finally:
        app_module._DEFAULT_VERSION_CACHE = original


class TestEnsureVersion:
    """Cover `_ensure_version` branches for config defaults."""

    def test_preserves_existing_version(self, reset_version_cache):
        """Should not mutate cache or hit defaults when version exists."""

        cfg = {"version": " 2 "}
        app_module._DEFAULT_VERSION_CACHE = "cached"

        app_module._ensure_version(cfg)

        assert cfg["version"].strip() == "2"
        assert app_module._DEFAULT_VERSION_CACHE == "cached"

    def test_reads_default_version_from_file(
        self, tmp_path: Path, monkeypatch, reset_version_cache
    ) -> None:
        """Populate missing version from the defaults YAML payload."""

        default_file = tmp_path / "defaults.yml"
        default_file.write_text("version: '7'\n", encoding="utf-8")
        monkeypatch.setattr(app_module, "DEFAULTS", default_file)

        cfg: dict[str, str] = {}
        app_module._ensure_version(cfg)

        assert cfg["version"] == "7"

    def test_falls_back_to_hardcoded_version(
        self, tmp_path: Path, monkeypatch, reset_version_cache
    ) -> None:
        """Use fallback value when defaults file lacks a version entry."""

        default_file = tmp_path / "defaults.yml"
        default_file.write_text("other: value\n", encoding="utf-8")
        monkeypatch.setattr(app_module, "DEFAULTS", default_file)

        cfg: dict[str, str] = {}
        app_module._ensure_version(cfg)

        assert cfg["version"] == "1"


class TestBuildStep0:
    """Test the _build_step0 function for config loading UI."""

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_build_step0_basic(self, mock_list_cfgs, mock_widgets):
        """Test basic _build_step0 functionality."""
        mock_list_cfgs.return_value = ["demo", "default"]
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = Mock()
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()
        mock_output = Mock(spec=["hold_trait_notifications"])
        mock_output.hold_trait_notifications.return_value = _cm_mock()
        mock_widgets.Output.return_value = mock_output

        store = ParamStore()
        result = _build_step0(store)

        assert result is not None
        mock_widgets.VBox.assert_called_once()

    @patch.dict(sys.modules, {"ipydatagrid": Mock(DataGrid=Mock())})
    @patch("trend_analysis.gui.app.HAS_DATAGRID", True)
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    @patch("trend_analysis.gui.app.widgets")
    def test_build_step0_with_datagrid(self, mock_widgets, mock_list_cfgs):
        """Test _build_step0 with DataGrid available."""
        mock_list_cfgs.return_value = ["demo"]
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        # Mock DataGrid availability
        mock_datagrid_instance = MagicMock()
        mock_datagrid_instance.on = MagicMock()  # Add the missing 'on' method
        mock_datagrid_instance.hold_trait_notifications.return_value = _cm_mock()
        mock_datagrid_class = MagicMock(return_value=mock_datagrid_instance)

        with patch("trend_analysis.gui.app.DataGrid", mock_datagrid_class):
            store = ParamStore()
            result = _build_step0(store)
            assert result is not None
            mock_datagrid_instance.on.assert_called()

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_build_step0_upload_callback(self, mock_list_cfgs, mock_widgets):
        """Test upload callback functionality."""
        mock_list_cfgs.return_value = ["demo"]

        mock_upload = Mock()
        mock_upload.value = {"test.yml": {"content": b"key: value"}}
        mock_widgets.FileUpload.return_value = mock_upload
        mock_widgets.Dropdown.return_value = Mock()
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        store = ParamStore()

        with patch("trend_analysis.gui.app.reset_weight_state"):
            _build_step0(store)

            # Verify upload observer was set
            mock_upload.observe.assert_called()

    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_upload_refreshes_grid(
        self, mock_list_cfgs: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uploading a config should update the DataGrid view and mark dirty."""

        mock_list_cfgs.return_value = ["demo"]

        mock_widgets = MagicMock()
        monkeypatch.setattr(app_module, "widgets", mock_widgets)

        upload = MagicMock()
        upload.value = {"new.yml": {"content": b"mode: all"}}
        mock_widgets.FileUpload.return_value = upload
        mock_widgets.Dropdown.return_value = MagicMock()
        mock_widgets.Button.return_value = MagicMock()
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HBox.return_value = MagicMock()

        grid = MagicMock()
        grid.data = []
        grid.hold_trait_notifications.return_value = _cm_mock()

        with (
            patch("trend_analysis.gui.app.HAS_DATAGRID", True),
            patch("trend_analysis.gui.app.DataGrid", return_value=grid),
            patch("trend_analysis.gui.app.reset_weight_state") as mock_reset,
        ):
            store = ParamStore()
            _build_step0(store)

            upload_callback = upload.observe.call_args[0][0]
            upload_callback({"new": upload.value}, store=store)

        assert store.cfg == {"mode": "all"}
        assert store.dirty is True
        assert grid.data == [store.cfg]
        grid.hold_trait_notifications.assert_called()
        mock_reset.assert_called_once_with(store)

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_template_error_handling_missing_file(self, mock_list_cfgs, mock_widgets):
        """Test template loading with missing file."""
        mock_list_cfgs.return_value = ["missing_template"]

        mock_dropdown = Mock()
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()
        mock_output = Mock(spec=["hold_trait_notifications"])
        mock_output.hold_trait_notifications.return_value = _cm_mock()
        mock_widgets.Output.return_value = mock_output

        store = ParamStore()

        with (
            patch("trend_analysis.gui.app.reset_weight_state"),
            patch("warnings.warn") as mock_warn,
        ):
            _build_step0(store)

            # Simulate template dropdown change with missing file
            template_callback = mock_dropdown.observe.call_args[0][0]
            change_event = {"new": "missing_template"}

            template_callback(change_event, store=store)

            # Verify warning was issued for missing file
            mock_warn.assert_called()
            warning_msg = str(mock_warn.call_args[0][0])
            assert "Template config file not found" in warning_msg

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_template_error_handling_invalid_yaml(self, mock_list_cfgs, mock_widgets):
        """Test template loading with invalid YAML."""
        mock_list_cfgs.return_value = ["invalid_template"]

        mock_dropdown = Mock()
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        store = ParamStore()

        with (
            patch("trend_analysis.gui.app.reset_weight_state"),
            patch("warnings.warn") as mock_warn,
            patch("pathlib.Path.read_text", return_value="invalid: yaml: content: {"),
        ):
            _build_step0(store)

            # Simulate template dropdown change with invalid YAML
            template_callback = mock_dropdown.observe.call_args[0][0]
            change_event = {"new": "invalid_template"}

            template_callback(change_event, store=store)

            # Verify warning was issued for invalid YAML
            mock_warn.assert_called()
            warning_msg = str(mock_warn.call_args[0][0])
            assert "Invalid YAML in template config" in warning_msg

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_template_error_handling_permission_error(
        self, mock_list_cfgs, mock_widgets
    ):
        """Test template loading with permission error."""
        mock_list_cfgs.return_value = ["permission_template"]

        mock_dropdown = Mock()
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        store = ParamStore()

        with (
            patch("trend_analysis.gui.app.reset_weight_state"),
            patch("warnings.warn") as mock_warn,
            patch(
                "pathlib.Path.read_text",
                side_effect=PermissionError("Permission denied"),
            ),
        ):
            _build_step0(store)

            # Simulate template dropdown change with permission error
            template_callback = mock_dropdown.observe.call_args[0][0]
            change_event = {"new": "permission_template"}

            template_callback(change_event, store=store)

            # Verify warning was issued for permission error
            mock_warn.assert_called()
            warning_msg = str(mock_warn.call_args[0][0])
            assert "Permission denied reading template config" in warning_msg

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_template_error_handling_generic_exception(
        self, mock_list_cfgs, mock_widgets
    ):
        """Template handler should surface unexpected exceptions."""

        mock_list_cfgs.return_value = ["broken_template"]

        mock_dropdown = Mock()
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        store = ParamStore()

        with (
            patch("trend_analysis.gui.app.reset_weight_state"),
            patch("warnings.warn") as mock_warn,
            patch("pathlib.Path.read_text", side_effect=RuntimeError("boom")),
        ):
            _build_step0(store)

            template_callback = mock_dropdown.observe.call_args[0][0]
            change_event = {"new": "broken_template"}

            template_callback(change_event, store=store)

            mock_warn.assert_called()
            warning_msg = str(mock_warn.call_args[0][0])
            assert "Failed to load template config" in warning_msg

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    def test_template_loading_success(self, mock_list_cfgs, mock_widgets):
        """Test successful template loading doesn't crash."""
        mock_list_cfgs.return_value = ["demo"]  # Use actual existing template

        mock_dropdown = Mock()
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        store = ParamStore()

        with patch("trend_analysis.gui.app.reset_weight_state", lambda store: None):
            with patch("pathlib.Path.read_text", return_value="foo: bar"):
                _build_step0(store)

                # Simulate template dropdown change with existing template
                template_callback = mock_dropdown.observe.call_args[0][0]
                change_event = {"new": "demo"}

                import warnings

                # This should not crash - the function should handle any errors gracefully
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        template_callback(change_event, store=store)
                    success = True
                except Exception:
                    success = False

                assert success, "Template loading should handle errors gracefully"


class TestBuildRankOptions:
    """Test the _build_rank_options function."""

    @patch("trend_analysis.gui.app.widgets")
    def test_build_rank_options_basic(self, mock_widgets):
        """Test basic _build_rank_options functionality."""
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.SelectMultiple.return_value = Mock()
        mock_widgets.FloatSlider.return_value = Mock()

        with patch(
            "trend_analysis.core.rank_selection.METRIC_REGISTRY",
            {"AnnualReturn": Mock()},
        ):
            store = ParamStore()
            result = _build_rank_options(store)

            assert result is not None
            mock_widgets.VBox.assert_called()


class TestLaunch:
    """Test the launch function."""

    @patch("trend_analysis.gui.app.widgets")
    def test_launch_basic(self, mock_widgets):
        """Test basic launch functionality."""
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()

        result = launch()

        assert result is not None
        mock_widgets.VBox.assert_called()

    @patch("trend_analysis.gui.app.widgets")
    def test_launch_upload_callback(self, mock_widgets):
        """Test upload callback in launch."""
        mock_upload = Mock()
        mock_upload.value = {}
        mock_widgets.FileUpload.return_value = mock_upload
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()
        mock_widgets.VBox.return_value = Mock()

        launch()

        # Verify widgets were created
        mock_widgets.VBox.assert_called()


class TestLaunchApp:
    """Test the launch function."""

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.load_state")
    @patch("trend_analysis.gui.app.discover_plugins")
    def test_launch_basic(self, mock_discover, mock_load_state, mock_widgets):
        """Test basic launch functionality."""
        mock_load_state.return_value = ParamStore()
        mock_discover.return_value = None

        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()

        with patch("trend_analysis.gui.app._build_step0") as mock_step0:
            with patch("trend_analysis.gui.app._build_rank_options") as mock_rank:
                mock_step0.return_value = Mock()
                mock_rank.return_value = Mock()

                result = launch()

                assert result is not None
                mock_widgets.VBox.assert_called()

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.load_state")
    @patch("trend_analysis.gui.app.discover_plugins")
    def test_launch_with_plugins(self, mock_discover, mock_load_state, mock_widgets):
        """Test launch with plugins discovered."""
        mock_load_state.return_value = ParamStore()

        # Mock plugin discovery
        mock_plugin = Mock()
        mock_plugin.name = "test_plugin"
        mock_plugin.ui_builder = Mock(return_value=Mock())
        mock_plugin.__name__ = "test_plugin"  # Add the missing __name__ attribute

        with patch("trend_analysis.gui.app.iter_plugins", return_value=[mock_plugin]):
            mock_widgets.VBox.return_value = Mock()
            mock_widgets.HTML.return_value = Mock()
            mock_widgets.Button.return_value = Mock()
            mock_widgets.Output.return_value = _cm_mock()

            with patch("trend_analysis.gui.app._build_step0") as mock_step0:
                with patch("trend_analysis.gui.app._build_rank_options") as mock_rank:
                    mock_step0.return_value = Mock()
                    mock_rank.return_value = Mock()

                    result = launch()

                    assert result is not None
                    mock_widgets.VBox.assert_called()


class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    @patch("trend_analysis.gui.app.widgets")
    def test_file_upload_validation(self, mock_widgets):
        """Test file upload validation and error handling."""
        mock_upload = Mock()
        mock_upload.value = {"test.csv": {"content": b"invalid,csv,data"}}
        mock_widgets.FileUpload.return_value = mock_upload
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()
        mock_widgets.VBox.return_value = Mock()

        store = ParamStore()

        # Test that upload handling doesn't crash on invalid data
        result = _build_step0(store)
        assert result is not None

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.Path")
    def test_config_template_loading(self, mock_path, mock_widgets):
        """Test config template loading functionality."""
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = Mock()
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        # Mock path resolution
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = "key: value"
        mock_path.return_value.resolve.return_value.parents = [
            Mock(),
            Mock(),
            Mock(),
            Mock(),
        ]
        mock_path.return_value.resolve.return_value.parents[3] = Mock()
        mock_path.return_value.resolve.return_value.parents[3].__truediv__ = Mock(
            return_value=mock_config_path
        )

        with patch("trend_analysis.gui.app.list_builtin_cfgs", return_value=["demo"]):
            store = ParamStore()
            result = _build_step0(store)

            assert result is not None

    @patch("trend_analysis.gui.app.widgets")
    def test_button_click_handlers(self, mock_widgets):
        """Test button click event handlers."""
        mock_save_btn = Mock()
        mock_download_btn = Mock()
        mock_widgets.Button.side_effect = [mock_save_btn, mock_download_btn]
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = Mock()
        mock_widgets.Label.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        with patch("trend_analysis.gui.app.list_builtin_cfgs", return_value=["demo"]):
            with patch("trend_analysis.gui.app.save_state"):
                store = ParamStore()
                _build_step0(store)

                # Verify button click handlers were set
                mock_save_btn.on_click.assert_called()
                mock_download_btn.on_click.assert_called()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_load_state_weight_file_error(self):
        """Test loading state when weight file is corrupted."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            f.write(b"corrupted pickle data")
            f.flush()

            with patch("trend_analysis.gui.app.STATE_FILE") as mock_state_file:
                with patch("trend_analysis.gui.app.WEIGHT_STATE_FILE", Path(f.name)):
                    mock_state_file.exists.return_value = False

                    with patch("warnings.warn") as mock_warn:
                        store = load_state()
                        assert isinstance(store, ParamStore)
                        mock_warn.assert_called()

    @patch.dict(sys.modules, {"ipydatagrid": Mock(DataGrid=Mock())})
    @patch("trend_analysis.gui.app.HAS_DATAGRID", True)
    @patch("trend_analysis.gui.app.list_builtin_cfgs")
    @patch("trend_analysis.gui.app.widgets")
    def test_datagrid_cell_change_error(self, mock_widgets, mock_list_cfgs):
        """Test DataGrid cell change error handling."""
        mock_list_cfgs.return_value = ["demo"]
        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        # Mock DataGrid availability
        mock_datagrid_instance = MagicMock()
        mock_datagrid_instance.on = MagicMock()  # Add the missing 'on' method
        mock_datagrid_instance.hold_trait_notifications.return_value = _cm_mock()
        mock_datagrid_class = MagicMock(return_value=mock_datagrid_instance)

        with patch("trend_analysis.gui.app.DataGrid", mock_datagrid_class):
            store = ParamStore()
            _build_step0(store)

            # Verify on method was called (for cell_edited event)
            mock_datagrid_instance.on.assert_called_with("cell_edited", ANY)

    @patch("trend_analysis.gui.app.widgets")
    @patch("trend_analysis.gui.app.asyncio")
    def test_async_operations(self, mock_asyncio, mock_widgets):
        """Test async operations in UI components."""
        mock_loop = Mock()
        mock_asyncio.get_event_loop.return_value = mock_loop

        mock_widgets.FileUpload.return_value = Mock()
        mock_widgets.Dropdown.return_value = Mock()
        mock_widgets.Label.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HBox.return_value = Mock()

        with patch("trend_analysis.gui.app.list_builtin_cfgs", return_value=["demo"]):
            store = ParamStore()
            result = _build_step0(store)

            assert result is not None


class DummyLayout:
    """Simple layout object mirroring ipywidgets layout attributes."""

    def __init__(self) -> None:
        self.border = ""
        self.display = "flex"


class DummyValueWidget:
    """Basic widget capturing observers and supporting value updates."""

    def __init__(self, value: object | None = None) -> None:
        self.value = value
        self._observers: list[Callable[[dict[str, object]], None]] = []
        self.layout = DummyLayout()

    def observe(
        self, callback: Callable[[dict[str, object]], None], names: str | None = None
    ) -> None:  # pragma: no cover - stub
        self._observers.append(callback)

    def set_value(self, value: object) -> None:
        self.value = value
        for cb in list(self._observers):
            cb({"new": value})


class DummyFileUpload(DummyValueWidget):
    """File upload stub returning stored payloads."""

    def __init__(self, accept: str = "", multiple: bool = False) -> None:
        super().__init__(value={})

    def trigger(self) -> None:
        for cb in list(self._observers):
            cb({"new": self.value})


class DummyDropdown(DummyValueWidget):
    """Dropdown widget exposing options and value change notifications."""

    def __init__(
        self,
        options: list[object] | None = None,
        value: object | None = None,
        description: str = "",
    ) -> None:
        opts = list(options or [])
        default = value if value is not None else (opts[0] if opts else None)
        super().__init__(default)
        self.options = opts
        self.description = description


class DummyCheckbox(DummyValueWidget):
    """Checkbox widget supporting observe callbacks."""

    def __init__(
        self, value: bool = False, description: str = "", indent: bool = True
    ) -> None:
        super().__init__(value)
        self.description = description
        self.indent = indent


class DummyToggleButtons(DummyValueWidget):
    """Toggle buttons stub mirroring ipywidgets API."""

    def __init__(self, options: list[str], value: str, description: str = "") -> None:
        super().__init__(value)
        self.options = options
        self.description = description


class DummyBoundedIntText(DummyValueWidget):
    def __init__(self, value: int = 0, **kwargs) -> None:
        super().__init__(value)
        for key, val in kwargs.items():
            setattr(self, key, val)


class DummyBoundedFloatText(DummyValueWidget):
    def __init__(self, value: float = 0.0, **kwargs) -> None:
        super().__init__(value)
        for key, val in kwargs.items():
            setattr(self, key, val)


class DummyFloatText(DummyValueWidget):
    def __init__(self, value: float = 0.0, **kwargs) -> None:
        super().__init__(value)
        for key, val in kwargs.items():
            setattr(self, key, val)


class DummyFloatSlider(DummyValueWidget):
    def __init__(self, value: float = 0.0, **kwargs) -> None:
        super().__init__(value)
        for key, val in kwargs.items():
            setattr(self, key, val)


class DummyButton:
    """Button stub storing click handlers."""

    def __init__(self, description: str = "") -> None:
        self.description = description
        self.layout = DummyLayout()
        self._handlers: list[Callable[["DummyButton"], None]] = []

    def on_click(
        self, callback: Callable[["DummyButton"], None]
    ) -> None:  # pragma: no cover - simple setter
        self._handlers.append(callback)

    def click(self) -> None:
        for cb in list(self._handlers):
            cb(self)


class DummyBox:
    """Container widget preserving child references."""

    def __init__(
        self, children: list[object] | tuple[object, ...] | None = None
    ) -> None:
        self.children = tuple(children or [])
        self.layout = DummyLayout()


class DummyLabel:
    def __init__(self, value: str = "") -> None:
        self.value = value
        self.layout = DummyLayout()


class FakeDataGrid:
    """Minimal DataGrid stand-in capturing callbacks and data updates."""

    def __init__(self, df, editable: bool = True) -> None:
        self.df = df
        self.editable = editable
        self.layout = DummyLayout()
        self.data: object | None = None
        self.callbacks: dict[str, Callable[..., None]] = {}

    def on(self, name: str, callback: Callable[..., None]) -> None:
        self.callbacks[name] = callback

    def hold_trait_notifications(self):  # pragma: no cover - trivial passthrough
        return contextlib.nullcontext(self)


class FakeLoop:
    """Event-loop stub executing callbacks immediately."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, Callable[..., None]]] = []

    def call_later(self, delay: float, callback: Callable[..., None]) -> None:
        self.calls.append((delay, callback))
        callback()


class ImmediateTask:
    """Wrapper returned by the immediate create_task helper."""

    def __init__(self, task: asyncio.Task) -> None:
        self._task = task

    def cancel(self) -> None:  # pragma: no cover - no asynchronous backlog in tests
        if not self._task.done():
            self._task.cancel()


def immediate_create_task(coro):
    """Create a task and execute it promptly for debounce callbacks."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(coro)
            loop.run_until_complete(task)
        finally:
            loop.close()
        return ImmediateTask(task)
    else:
        task = loop.create_task(coro)
        return ImmediateTask(task)


async def fake_sleep(_: float) -> None:
    """Async sleep stub returning immediately."""


def _call_observer(callback, change, store):
    """Invoke widget observer and execute coroutine callbacks when needed."""

    try:
        result = callback(change, store=store)
    except TypeError:
        result = callback(change)
    if asyncio.iscoroutine(result):
        asyncio.run(result)


def test_datagrid_import_sets_flag(monkeypatch):
    """Reload module with ipydatagrid present to hit import branch."""

    fake_module = SimpleNamespace(DataGrid=object)
    monkeypatch.setitem(sys.modules, "ipydatagrid", fake_module)

    reloaded = importlib.reload(app_module)
    try:
        assert reloaded.HAS_DATAGRID is True
    finally:
        monkeypatch.delitem(sys.modules, "ipydatagrid", raising=False)
        importlib.reload(app_module)


def test_load_state_with_weight_variants(tmp_path, monkeypatch):
    """Ensure load_state handles both new and legacy weight files."""

    state_file = tmp_path / "state.yml"
    weight_file = tmp_path / "weights.pkl"
    state_file.write_text("mode: all\n")
    weight_file.write_bytes(pickle.dumps({"adaptive_bayes_posteriors": {"fund": 1.0}}))

    monkeypatch.setattr(app_module, "STATE_FILE", state_file)
    monkeypatch.setattr(app_module, "WEIGHT_STATE_FILE", weight_file)

    store = app_module.load_state()
    assert store.weight_state == {"fund": 1.0}

    weight_file.write_bytes(pickle.dumps([0.1, 0.2]))
    store = app_module.load_state()
    assert store.weight_state == [0.1, 0.2]


def test_reset_weight_state_removes_file(tmp_path, monkeypatch):
    """reset_weight_state should clear persisted weight cache."""

    weight_file = tmp_path / "weights.pkl"
    weight_file.write_bytes(b"cached")

    store = ParamStore()
    store.weight_state = {"legacy": True}

    monkeypatch.setattr(app_module, "WEIGHT_STATE_FILE", weight_file)
    app_module.reset_weight_state(store)

    assert store.weight_state is None
    assert not weight_file.exists()


def test_build_config_dict_populates_defaults():
    """Non-minimal configs should receive expected default sections."""

    store = ParamStore()
    store.cfg = {"mode": "rank", "data": {"csv_path": "demo.csv"}}

    cfg = app_module.build_config_dict(store)

    for key in [
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "benchmarks",
        "metrics",
        "export",
        "run",
        "multi_period",
    ]:
        assert key in cfg


def test_build_config_from_store_uses_config_factory(monkeypatch):
    """build_config_from_store should honour the Config factory."""

    store = ParamStore()
    store.cfg = {"mode": "rank", "output": {"format": "csv"}}

    created: list[SimpleNamespace] = []

    def fake_config(**kwargs):
        ns = SimpleNamespace(**kwargs)
        created.append(ns)
        return ns

    monkeypatch.setattr(app_module, "Config", fake_config)
    cfg_obj = app_module.build_config_from_store(store)

    assert created and cfg_obj.mode == "rank"


def test_build_step0_datagrid_callbacks(monkeypatch, tmp_path):
    """Exercise the DataGrid editing and upload paths of _build_step0."""

    monkeypatch.setattr(app_module, "HAS_DATAGRID", True)
    monkeypatch.setattr(app_module, "DataGrid", FakeDataGrid)
    monkeypatch.setattr(app_module.asyncio, "get_event_loop", lambda: FakeLoop())
    monkeypatch.setattr(app_module, "list_builtin_cfgs", lambda: ["demo"])
    monkeypatch.setattr(app_module, "STATE_FILE", tmp_path / "state.yml")
    monkeypatch.setattr(app_module, "WEIGHT_STATE_FILE", tmp_path / "weights.pkl")

    save_calls: list[dict[str, object]] = []
    display_calls: list[object] = []
    monkeypatch.setattr(
        app_module, "save_state", lambda store: save_calls.append(store.to_dict())
    )
    monkeypatch.setattr(
        app_module,
        "reset_weight_state",
        lambda store: store.cfg.setdefault("reset", True),
    )
    monkeypatch.setattr(app_module, "display", lambda obj: display_calls.append(obj))
    monkeypatch.setattr(app_module, "FileLink", lambda path: f"link:{path}")

    orig_safe_load = yaml.safe_load

    def guarded_safe_load(text: str):
        if text == "bad":
            raise ValueError("invalid")
        return orig_safe_load(text)

    monkeypatch.setattr(app_module.yaml, "safe_load", guarded_safe_load)

    monkeypatch.setattr(app_module.widgets, "FileUpload", DummyFileUpload)
    monkeypatch.setattr(app_module.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app_module.widgets, "Label", DummyLabel)
    monkeypatch.setattr(app_module.widgets, "Button", DummyButton)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)
    monkeypatch.setattr(app_module.widgets, "HBox", DummyBox)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "demo.yml").write_text("version: '1'\nalpha: 1\n")
    monkeypatch.setattr(app_module, "_find_config_directory", lambda: cfg_dir)

    store = ParamStore()
    store.cfg = {"alpha": 1}

    container = app_module._build_step0(store)
    template, upload, grid, buttons = container.children
    save_btn, download_btn = buttons.children

    # Column other than value column should be ignored by the handler
    grid.callbacks["cell_edited"]({"column": 0, "row": 0, "new": "ignored"})
    assert store.cfg == {"alpha": 1}

    grid.callbacks["cell_edited"]({"column": 1, "row": 0, "new": "2"})
    assert store.cfg["alpha"] == 2 and store.dirty

    grid.callbacks["cell_edited"]({"column": 1, "row": 0, "new": "bad"})
    assert grid.layout.border == ""

    upload.value = {"user.yml": {"content": b"foo: bar"}}
    upload.trigger()
    assert store.cfg["foo"] == "bar" and store.cfg["reset"]

    template.set_value("demo")
    assert "alpha" in store.cfg

    save_btn.click()
    assert save_calls and not store.dirty

    download_btn.click()
    download_path = app_module.STATE_FILE.with_name("config_download.yml")
    assert download_path.exists()
    assert display_calls == [f"link:{download_path}"]


def test_build_step0_datagrid_missing_on(monkeypatch, tmp_path):
    """If DataGrid lacks an ``on`` handler the code should degrade
    gracefully."""

    class GridNoOn(FakeDataGrid):
        def on(self, *args, **kwargs):
            raise AttributeError("no handler")

    monkeypatch.setattr(app_module, "HAS_DATAGRID", True)
    monkeypatch.setattr(app_module, "DataGrid", GridNoOn)
    monkeypatch.setattr(app_module, "list_builtin_cfgs", lambda: ["demo"])
    monkeypatch.setattr(app_module, "STATE_FILE", tmp_path / "state.yml")
    monkeypatch.setattr(app_module, "WEIGHT_STATE_FILE", tmp_path / "weights.pkl")

    monkeypatch.setattr(app_module, "reset_weight_state", lambda store: None)
    monkeypatch.setattr(app_module, "save_state", lambda store: None)
    monkeypatch.setattr(app_module, "FileLink", lambda path: path)
    monkeypatch.setattr(app_module, "display", lambda obj: obj)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "demo.yml").write_text("version: '1'\nalpha: 1\n")
    monkeypatch.setattr(app_module, "_find_config_directory", lambda: cfg_dir)

    monkeypatch.setattr(app_module.widgets, "FileUpload", DummyFileUpload)
    monkeypatch.setattr(app_module.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app_module.widgets, "Label", DummyLabel)
    monkeypatch.setattr(app_module.widgets, "Button", DummyButton)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)
    monkeypatch.setattr(app_module.widgets, "HBox", DummyBox)

    store = ParamStore()
    store.cfg = {"alpha": 1}

    # Should not raise even though GridNoOn.on raises AttributeError
    container = app_module._build_step0(store)
    assert isinstance(container, DummyBox)


def test_build_rank_options_observers(monkeypatch):
    """Verify rank option widgets persist changes back into the store."""

    monkeypatch.setattr(app_module.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app_module.widgets, "BoundedIntText", DummyBoundedIntText)
    monkeypatch.setattr(app_module.widgets, "BoundedFloatText", DummyBoundedFloatText)
    monkeypatch.setattr(app_module.widgets, "FloatText", DummyFloatText)
    monkeypatch.setattr(app_module.widgets, "FloatSlider", DummyFloatSlider)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)
    monkeypatch.setattr(app_module.asyncio, "create_task", immediate_create_task)
    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    import trend_analysis.gui.utils as utils_module

    monkeypatch.setattr(utils_module.asyncio, "create_task", immediate_create_task)
    monkeypatch.setattr(utils_module.asyncio, "sleep", fake_sleep)
    time_values = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    monkeypatch.setattr(utils_module.time, "time", lambda: next(time_values))

    metrics = {"Sharpe": object(), "Sortino": object(), "MaxDrawdown": object()}
    import trend_analysis.core.rank_selection as rank_selection_module

    monkeypatch.setattr(rank_selection_module, "METRIC_REGISTRY", metrics)

    store = ParamStore()
    result = app_module._build_rank_options(store)
    incl, metric, n_text, pct_text, thresh, blended = result.children
    m1, w1, m2, w2, m3, w3 = blended.children

    incl.value = "threshold"
    _call_observer(incl._observers[0], {"new": "threshold"}, store)

    metric.value = "blended"
    for callback in metric._observers:
        _call_observer(callback, {"new": "blended"}, store)

    n_text.value = 5
    _call_observer(n_text._observers[0], {"new": 5}, store)

    pct_text.value = 0.2
    _call_observer(pct_text._observers[0], {"new": 0.2}, store)

    thresh.value = 1.5
    _call_observer(thresh._observers[0], {"new": 1.5}, store)

    w1.value = 0.4
    _call_observer(w1._observers[0], {"new": 0.4}, store)

    assert store.cfg["rank"]["inclusion_approach"] == "threshold"
    assert blended.layout.display == "flex"


def test_build_manual_override_datagrid(monkeypatch):
    """Manual override grid should mutate manual lists and weights."""

    monkeypatch.setattr(app_module, "HAS_DATAGRID", True)
    monkeypatch.setattr(app_module, "DataGrid", FakeDataGrid)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)
    fake_module = SimpleNamespace(DataGrid=FakeDataGrid)
    monkeypatch.setitem(sys.modules, "ipydatagrid", fake_module)

    store = ParamStore()
    store.cfg = {
        "portfolio": {
            "custom_weights": {"FundA": 0.1},
            "manual_list": ["FundA", "FundB"],
        }
    }

    box = app_module._build_manual_override(store)
    grid = box.children[0]

    # Column outside editable range should be ignored
    grid.callbacks["cell_edited"]({"row": 0, "column": 0, "new": "noop"})

    grid.callbacks["cell_edited"]({"row": 1, "column": 1, "new": False})
    grid.callbacks["cell_edited"]({"row": 1, "column": 1, "new": True})
    grid.callbacks["cell_edited"]({"row": 0, "column": 2, "new": "0.3"})
    grid.callbacks["cell_edited"]({"row": 0, "column": 2, "new": None})
    grid.callbacks["cell_edited"]({"row": 0, "column": 2, "new": "-1"})

    manual = store.cfg["portfolio"]["manual_list"]
    weights = store.cfg["portfolio"]["custom_weights"]

    assert manual.count("FundB") == 1 and weights["FundA"] == 0.3
    assert store.dirty is True


def test_build_manual_override_datagrid_missing_on(monkeypatch):
    """Gracefully handle DataGrid implementations lacking an ``on`` method."""

    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)

    class NoOnDataGrid:
        def __init__(self, df, editable: bool = True) -> None:
            self.df = df
            self.editable = editable
            self.layout = DummyLayout()
            self.data = None

    fake_module = ModuleType("ipydatagrid")
    fake_module.DataGrid = NoOnDataGrid  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ipydatagrid", fake_module)

    store = ParamStore()
    store.cfg = {
        "portfolio": {"custom_weights": {"FundA": 0.4}, "manual_list": ["FundA"]}
    }

    box = app_module._build_manual_override(store)

    assert isinstance(box.children[0], NoOnDataGrid)


def test_build_manual_override_import_error(monkeypatch):
    """Import failures should fall back to the non-DataGrid widgets."""

    class DummySelectMultiple(DummyValueWidget):
        def __init__(self, options=None, value=(), description: str = "") -> None:
            super().__init__(tuple(value))
            self.options = tuple(options or [])
            self.description = description

    monkeypatch.setitem(sys.modules, "ipydatagrid", ModuleType("ipydatagrid"))
    monkeypatch.setattr(app_module.widgets, "Label", DummyLabel)
    monkeypatch.setattr(app_module.widgets, "SelectMultiple", DummySelectMultiple)
    monkeypatch.setattr(app_module.widgets, "FloatText", DummyFloatText)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)

    store = ParamStore()
    store.cfg = {
        "portfolio": {"custom_weights": {"FundA": 0.2}, "manual_list": ["FundA"]}
    }

    box = app_module._build_manual_override(store)
    warn, select, weights_box = box.children

    assert isinstance(warn, DummyLabel)
    assert isinstance(select, DummySelectMultiple)
    if hasattr(weights_box, "children"):
        assert weights_box.children and isinstance(
            weights_box.children[0], DummyFloatText
        )
    else:
        assert isinstance(weights_box, DummyFloatText)

    # Trigger selection change and ensure manual list updates via fallback handlers
    select.set_value(("FundA", "FundA"))
    assert store.cfg["portfolio"]["manual_list"] == ["FundA", "FundA"]


def test_build_weighting_options_callbacks(monkeypatch):
    """Weighting widgets should keep params in sync with the store."""

    monkeypatch.setattr(app_module.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app_module.widgets, "IntSlider", DummyBoundedIntText)
    monkeypatch.setattr(app_module.widgets, "FloatSlider", DummyFloatSlider)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)
    monkeypatch.setattr(app_module.asyncio, "create_task", immediate_create_task)
    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    import trend_analysis.gui.utils as utils_module

    monkeypatch.setattr(utils_module.asyncio, "create_task", immediate_create_task)
    monkeypatch.setattr(utils_module.asyncio, "sleep", fake_sleep)
    time_values = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    monkeypatch.setattr(utils_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(
        app_module, "iter_plugins", lambda: [type("Plugin", (), {"__name__": "custom"})]
    )

    store = ParamStore()
    result = app_module._build_weighting_options(store)
    method, adv_box = result.children
    hl, obs, max_w, tau = adv_box.children

    method.value = "adaptive_bayes"
    for callback in method._observers:
        _call_observer(callback, {"new": "adaptive_bayes"}, store)

    hl.value = 120
    _call_observer(hl._observers[0], {"new": 120}, store)

    obs.value = 0.4
    _call_observer(obs._observers[0], {"new": 0.4}, store)

    max_w.value = 0.25
    _call_observer(max_w._observers[0], {"new": 0.25}, store)

    tau.value = 1.2
    _call_observer(tau._observers[0], {"new": 1.2}, store)

    weight_cfg = store.cfg["portfolio"]["weighting"]
    assert weight_cfg["name"] == "adaptive_bayes"
    assert weight_cfg["params"]["prior_tau"] == 1.2
    assert adv_box.layout.display == "flex"


def test_launch_interactions(monkeypatch, tmp_path):
    """Validate launch wiring including run/export callbacks."""

    monkeypatch.setattr(app_module.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app_module.widgets, "Checkbox", DummyCheckbox)
    monkeypatch.setattr(app_module.widgets, "ToggleButtons", DummyToggleButtons)
    monkeypatch.setattr(app_module.widgets, "Button", DummyButton)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)

    rank_box = DummyBox()
    manual_box = DummyBox()
    weight_box = DummyBox()

    monkeypatch.setattr(app_module, "_build_step0", lambda store: DummyBox())
    monkeypatch.setattr(app_module, "_build_rank_options", lambda store: rank_box)
    monkeypatch.setattr(app_module, "_build_manual_override", lambda store: manual_box)
    monkeypatch.setattr(
        app_module, "_build_weighting_options", lambda store: weight_box
    )
    monkeypatch.setattr(app_module, "discover_plugins", lambda: None)

    store = ParamStore()
    store.cfg = {"output": {"format": "excel"}}

    monkeypatch.setattr(app_module, "load_state", lambda: store)

    sample_split = {
        "in_start": "2020-01",
        "in_end": "2020-06",
        "out_start": "2020-07",
        "out_end": "2020-12",
    }

    def build_cfg(store_obj: ParamStore) -> SimpleNamespace:
        output_cfg = store_obj.cfg.get("output", {}).copy()
        output_cfg.setdefault("path", str(tmp_path / "out"))
        return SimpleNamespace(output=output_cfg, sample_split=sample_split)

    monkeypatch.setattr(app_module, "build_config_from_store", build_cfg)

    run_calls: list[pd.DataFrame] = []
    full_calls: list[object] = []
    export_calls: list[tuple] = []
    json_calls: list[tuple] = []
    save_calls: list[ParamStore] = []
    reset_calls: list[ParamStore] = []
    theme_calls: list[str] = []

    monkeypatch.setattr(
        app_module.pipeline,
        "run",
        lambda cfg: run_calls.append(pd.DataFrame({"a": [1]}))
        or pd.DataFrame({"a": [1]}),
    )
    monkeypatch.setattr(
        app_module.pipeline,
        "run_full",
        lambda cfg: full_calls.append({"metrics": pd.DataFrame({"a": [1]})})
        or {"metrics": pd.DataFrame({"a": [1]})},
    )
    monkeypatch.setattr(
        app_module.export,
        "make_summary_formatter",
        lambda *args, **kwargs: lambda df: df,
    )
    monkeypatch.setattr(
        app_module.export,
        "export_to_excel",
        lambda data, path, default_sheet_formatter=None: export_calls.append(
            (path, data)
        ),
    )

    exporters = dict(app_module.export.EXPORTERS)
    exporters["json"] = lambda data, path, _: json_calls.append((path, data))
    monkeypatch.setattr(app_module.export, "EXPORTERS", exporters)

    monkeypatch.setattr(
        app_module, "save_state", lambda store: save_calls.append(store)
    )
    monkeypatch.setattr(
        app_module, "reset_weight_state", lambda store: reset_calls.append(store)
    )
    monkeypatch.setattr(app_module, "Javascript", lambda script: script)
    monkeypatch.setattr(
        app_module, "display", lambda payload: theme_calls.append(payload)
    )

    container = app_module.launch()
    _, mode, vol_adj, use_rank, _, _, _, fmt_dd, theme, reset_btn, run_btn = (
        container.children
    )

    mode.set_value("rank")
    assert store.cfg["mode"] == "rank" and rank_box.layout.display == "flex"

    use_rank.set_value(True)
    assert rank_box.layout.display == "flex"

    mode.set_value("manual")
    assert manual_box.layout.display == "flex"

    vol_adj.set_value(True)
    assert store.cfg["use_vol_adjust"] is True

    fmt_dd.set_value("json")
    assert store.cfg["output"]["format"] == "json"

    theme.set_value("dark")
    assert store.theme == "dark" and theme_calls

    run_btn.click()
    assert json_calls and save_calls and not store.dirty

    fmt_dd.set_value("excel")
    run_btn.click()
    assert export_calls

    reset_btn.click()
    assert reset_calls


def test_launch_run_with_empty_metrics(monkeypatch, tmp_path):
    """Ensure on_run exits early when pipeline returns an empty metrics
    frame."""

    class DummyOutput:
        def __init__(self) -> None:
            self._ctx = contextlib.nullcontext(self)

        def hold_trait_notifications(self):  # pragma: no cover - trivial passthrough
            return self._ctx

    monkeypatch.setattr(app_module, "discover_plugins", lambda: None)
    monkeypatch.setattr(app_module, "list_builtin_cfgs", lambda: ["demo"])

    monkeypatch.setattr(app_module.widgets, "FileUpload", DummyFileUpload)
    monkeypatch.setattr(app_module.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app_module.widgets, "Checkbox", DummyCheckbox)
    monkeypatch.setattr(app_module.widgets, "ToggleButtons", DummyToggleButtons)
    monkeypatch.setattr(app_module.widgets, "Button", DummyButton)
    monkeypatch.setattr(app_module.widgets, "Label", DummyLabel)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)
    monkeypatch.setattr(app_module.widgets, "HBox", DummyBox)
    monkeypatch.setattr(app_module.widgets, "Output", DummyOutput)
    monkeypatch.setattr(app_module.widgets, "FloatText", DummyFloatText)
    monkeypatch.setattr(app_module.widgets, "BoundedFloatText", DummyBoundedFloatText)
    monkeypatch.setattr(app_module.widgets, "IntText", DummyBoundedIntText)
    monkeypatch.setattr(app_module.widgets, "BoundedIntText", DummyBoundedIntText)
    monkeypatch.setattr(app_module.widgets, "IntSlider", DummyBoundedIntText)
    monkeypatch.setattr(app_module.widgets, "FloatSlider", DummyFloatSlider)
    monkeypatch.setattr(app_module.widgets, "SelectMultiple", DummyDropdown)

    store = ParamStore()
    store.cfg = {"output": {"format": "excel", "path": str(tmp_path / "out")}}
    monkeypatch.setattr(app_module, "load_state", lambda: store)

    sample_split = {
        "in_start": "2021-01",
        "in_end": "2021-06",
        "out_start": "2021-07",
        "out_end": "2021-12",
    }

    monkeypatch.setattr(
        app_module,
        "build_config_from_store",
        lambda store_obj: SimpleNamespace(
            output=store_obj.cfg.get("output", {}), sample_split=sample_split
        ),
    )

    run_calls: list[object] = []
    monkeypatch.setattr(
        app_module.pipeline, "run", lambda cfg: run_calls.append(cfg) or pd.DataFrame()
    )
    monkeypatch.setattr(
        app_module.pipeline,
        "run_full",
        lambda cfg: pytest.fail("run_full should not execute"),
    )
    saved: list[ParamStore] = []
    monkeypatch.setattr(
        app_module, "save_state", lambda store_obj: saved.append(store_obj)
    )

    container = app_module.launch()
    _, _, _, _, _, _, _, _, _, _, run_btn = container.children

    run_btn.click()

    assert run_calls, "Expected pipeline.run to be invoked"
    assert saved == [], "save_state should not be called when metrics are empty"
    assert store.cfg["output"]["format"] == "excel"


def test_launch_run_with_custom_exporter(monkeypatch, tmp_path):
    """Execute alternative exporter paths when format is not Excel."""

    monkeypatch.setattr(app_module, "discover_plugins", lambda: None)
    monkeypatch.setattr(app_module, "_build_step0", lambda store: DummyBox())
    monkeypatch.setattr(app_module, "_build_rank_options", lambda store: DummyBox())
    monkeypatch.setattr(app_module, "_build_manual_override", lambda store: DummyBox())
    monkeypatch.setattr(
        app_module, "_build_weighting_options", lambda store: DummyBox()
    )
    monkeypatch.setattr(app_module, "reset_weight_state", lambda store: None)

    monkeypatch.setattr(app_module.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app_module.widgets, "Checkbox", DummyCheckbox)
    monkeypatch.setattr(app_module.widgets, "ToggleButtons", DummyToggleButtons)
    monkeypatch.setattr(app_module.widgets, "Button", DummyButton)
    monkeypatch.setattr(app_module.widgets, "VBox", DummyBox)
    monkeypatch.setattr(app_module.widgets, "Label", DummyLabel)
    monkeypatch.setattr(app_module.widgets, "Output", DummyBox)
    monkeypatch.setattr(app_module.widgets, "FileUpload", DummyFileUpload)

    store = ParamStore()
    store.cfg = {"output": {"format": "json", "path": str(tmp_path / "payload")}}
    monkeypatch.setattr(app_module, "load_state", lambda: store)

    sample_split = {
        "in_start": "2022-01",
        "in_end": "2022-06",
        "out_start": "2022-07",
        "out_end": "2022-12",
    }

    def build_cfg(store_obj: ParamStore) -> SimpleNamespace:
        return SimpleNamespace(
            output=store_obj.cfg.get("output", {}), sample_split=sample_split
        )

    monkeypatch.setattr(app_module, "build_config_from_store", build_cfg)

    metrics = pd.DataFrame({"ret": [0.1, 0.2]})
    monkeypatch.setattr(app_module.pipeline, "run", lambda cfg: metrics)
    monkeypatch.setattr(
        app_module.pipeline,
        "run_full",
        lambda cfg: pytest.fail("run_full should not run"),
    )

    exported: list[tuple[str, dict[str, pd.DataFrame]]] = []
    monkeypatch.setattr(
        app_module.export,
        "EXPORTERS",
        {"json": lambda data, path, _: exported.append((path, data))},
    )
    monkeypatch.setattr(
        app_module.export,
        "export_to_excel",
        lambda *args, **kwargs: pytest.fail("excel exporter should not fire"),
    )

    saved: list[ParamStore] = []
    monkeypatch.setattr(
        app_module, "save_state", lambda store_obj: saved.append(store_obj)
    )

    container = app_module.launch()
    run_btn = container.children[-1]

    run_btn.click()

    assert exported and exported[0][0].endswith("payload")
    assert "metrics" in exported[0][1]
    assert saved == [store]
    assert store.dirty is False
