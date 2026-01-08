"""Additional coverage for the configuration fallback implementation.

These tests exercise the ``trend_analysis.config.models`` module when
``pydantic`` is unavailable so that the simplified ``SimpleBaseModel`` and
``_FallbackConfig`` code paths are covered.  The tests avoid mutating the
production import by loading the module under an isolated name with
``sys.modules['pydantic']`` temporarily set to ``None`` so the import raises
``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_config_module_without_pydantic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str = "tests.config_models_fallback",
) -> ModuleType:
    """Load ``config.models`` with ``pydantic`` forcibly unavailable."""

    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "trend_analysis" / "config" / "models.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with monkeypatch.context() as ctx:
        ctx.setitem(sys.modules, "pydantic", None)
        ctx.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)

    return module


@pytest.fixture
def fallback_models(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Provide the fallback configuration module and clean up afterwards."""

    module_name = "tests.config_models_fallback"
    module = _load_config_module_without_pydantic(monkeypatch, module_name=module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_simple_base_model_initialises_defaults(fallback_models: ModuleType) -> None:
    """Instantiating ``SimpleBaseModel`` exercises its default hooks."""

    base = fallback_models.SimpleBaseModel()  # type: ignore[attr-defined]
    assert base.__dict__ == {}


def test_fallback_config_provides_defaults(fallback_models: ModuleType) -> None:
    """``Config`` in fallback mode should surface sensible defaults."""

    Config = fallback_models.Config  # type: ignore[attr-defined]
    cfg = Config(version="1.2.3")

    assert cfg.version == "1.2.3"
    # Defaults from SimpleBaseModel should populate the dictionary sections.
    for field in [
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "metrics",
        "export",
        "performance",
        "run",
    ]:
        assert getattr(cfg, field) == {}
    # Optional/nullable defaults
    assert cfg.output is None


@pytest.mark.parametrize(
    "bad_kwargs, expected_message",
    [
        ({"version": ""}, "String should have at least 1 character"),
        ({"version": "   "}, "Version field cannot be empty"),
        ({"data": None}, "data section is required"),
        ({"portfolio": []}, "portfolio must be a dictionary"),
        (
            {"portfolio": {"transaction_cost_bps": -0.1}},
            "transaction_cost_bps must be >= 0",
        ),
        ({"portfolio": {"max_turnover": 3.5}}, "max_turnover must be <= 2.0"),
    ],
)
def test_fallback_config_validation_errors(
    fallback_models: ModuleType,
    bad_kwargs: dict[str, Any],
    expected_message: str,
) -> None:
    """Invalid inputs should trigger the fallback validator."""

    Config = fallback_models.Config  # type: ignore[attr-defined]
    kwargs = {"version": "1.0", **bad_kwargs}
    with pytest.raises(ValueError, match=expected_message):
        Config(**kwargs)


def test_preset_config_requires_name(fallback_models: ModuleType) -> None:
    """The simplified ``PresetConfig`` should reject a missing name."""

    PresetConfig = fallback_models.PresetConfig  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="Preset name must be specified"):
        PresetConfig(
            name="",
            description="",
            data={},
            preprocessing={},
            vol_adjust={},
            sample_split={},
            portfolio={},
            metrics={},
            export={},
            run={},
        )


def test_column_mapping_defaults_and_validation(fallback_models: ModuleType) -> None:
    """Column mapping should populate defaults and validate required fields."""

    ColumnMapping = fallback_models.ColumnMapping  # type: ignore[attr-defined]

    # Missing required values triggers validation errors.
    with pytest.raises(ValueError, match="Date column must be specified"):
        ColumnMapping()

    mapping = ColumnMapping(
        date_column="Date",
        return_columns=["FundA"],
        benchmark_column=None,
        risk_free_column=None,
    )
    assert mapping.column_display_names == {}
    assert mapping.column_tickers == {}


def test_configuration_state_defaults(fallback_models: ModuleType) -> None:
    """``ConfigurationState`` should expose sensible defaults."""

    ConfigurationState = fallback_models.ConfigurationState  # type: ignore[attr-defined]
    state = ConfigurationState()
    assert state.preset_name == ""
    assert state.column_mapping is None
    assert state.config_dict == {}
    assert state.uploaded_data is None
    assert state.analysis_results is None


def test_load_merges_output_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``load`` merges legacy ``output`` into the export settings."""

    module = _load_config_module_without_pydantic(
        monkeypatch, module_name="tests.config_models_fallback_load"
    )
    try:
        cfg_dict = {
            "version": "1.0",
            "data": {},
            "preprocessing": {},
            "vol_adjust": {},
            "sample_split": {},
            "portfolio": {},
            "benchmarks": {},
            "metrics": {},
            "export": {"formats": "csv"},
            "performance": {},
            "run": {},
            "output": {"format": ["CSV", "xlsx"], "path": "reports/summary.xlsx"},
        }

        loaded = module.load(cfg_dict)
        export_cfg = loaded.export
        # Formats should be merged case-insensitively and deduplicated.
        assert export_cfg["formats"] == ["csv", "xlsx"]
        # Path is split into directory and filename when present.
        assert export_cfg["directory"].endswith("reports")
        assert export_cfg["filename"] == "summary.xlsx"
    finally:
        sys.modules.pop("tests.config_models_fallback_load", None)


def test_load_without_pydantic_when_model_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback load should succeed even if the real model module is cached."""

    import trend_analysis.config.model  # noqa: F401 - populate sys.modules

    module = _load_config_module_without_pydantic(
        monkeypatch, module_name="tests.config_models_fallback_preloaded"
    )
    try:
        payload = {
            "version": "1.0",
            "data": {},
            "preprocessing": {},
            "vol_adjust": {},
            "sample_split": {},
            "portfolio": {},
            "benchmarks": {},
            "metrics": {},
            "export": {},
            "performance": {},
            "run": {},
        }

        cfg = module.load(payload)
        assert cfg.version == "1.0"
        assert cfg.export == {}
    finally:
        sys.modules.pop("tests.config_models_fallback_preloaded", None)


def test_load_preset_missing_file_raises(
    fallback_models: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_preset`` should raise when the requested file is absent."""

    fake_dir = tmp_path / "config"
    fake_dir.mkdir()
    (fake_dir / "defaults.yml").write_text("version: 1", encoding="utf-8")

    monkeypatch.setattr(
        fallback_models,
        "_find_config_directory",
        lambda: fake_dir,
        raising=False,
    )

    with pytest.raises(FileNotFoundError, match="Preset file not found"):
        fallback_models.load_preset("missing")


def test_list_available_presets_handles_empty_directory(
    fallback_models: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preset lister should return an empty list when none exist."""

    fake_dir = tmp_path / "config"
    fake_dir.mkdir()
    (fake_dir / "defaults.yml").write_text("version: 1", encoding="utf-8")

    monkeypatch.setattr(
        fallback_models,
        "_find_config_directory",
        lambda: fake_dir,
        raising=False,
    )

    assert fallback_models.list_available_presets() == []


def test_pydantic_dict_field_detection_handles_item_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the Pydantic helper tolerates ``model_fields`` that lack
    ``items``."""

    import trend_analysis.config.models as models  # type: ignore[import-not-found]

    class BrokenFields:
        def items(self):  # pragma: no cover - exercised via the except path
            raise RuntimeError("boom")

    original = getattr(models._PydanticConfigImpl, "model_fields")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        models._PydanticConfigImpl,  # type: ignore[attr-defined]
        "model_fields",
        BrokenFields(),
        raising=False,
    )

    try:
        assert models._PydanticConfigImpl._dict_field_names() == []  # type: ignore[attr-defined]
    finally:
        monkeypatch.setattr(
            models._PydanticConfigImpl,  # type: ignore[attr-defined]
            "model_fields",
            original,
            raising=False,
        )
