from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Generator

import pytest
import yaml


@pytest.fixture
def fallback_models(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[ModuleType, None, None]:
    """Load ``trend_analysis.config.models`` with pydantic forcibly
    unavailable."""

    module_name = "trend_analysis.config.models_fallback_test"
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "trend_analysis"
        / "config"
        / "models.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    validate_calls: list[tuple[dict[str, Any], Path | None]] = []

    def fake_validate(data: dict[str, Any], *, base_path: Path | None = None) -> None:
        validate_calls.append((copy.deepcopy(data), base_path))

    stub_model = ModuleType("trend_analysis.config.model")
    stub_model.validate_trend_config = fake_validate  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "trend_analysis.config.model", stub_model)
    monkeypatch.setitem(sys.modules, "pydantic", None)
    monkeypatch.setitem(sys.modules, module_name, module)

    spec.loader.exec_module(module)

    module._validate_calls = validate_calls  # type: ignore[attr-defined]
    try:
        yield module
    finally:
        validate_calls.clear()


def _base_config_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "1.0",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "metrics": {},
        "export": {},
        "performance": {},
        "run": {},
    }
    payload.update(overrides)
    return payload


def test_simple_base_model_initialises_defaults(fallback_models: ModuleType) -> None:
    base = fallback_models.SimpleBaseModel()  # type: ignore[attr-defined]
    assert base.__dict__ == {}


def test_find_config_directory_locates_defaults(fallback_models: ModuleType) -> None:
    config_dir = fallback_models._find_config_directory()  # type: ignore[attr-defined]
    assert (config_dir / "defaults.yml").exists()


def test_validate_version_helper_handles_errors(fallback_models: ModuleType) -> None:
    with pytest.raises(ValueError, match="version must be a string"):
        fallback_models._validate_version_value(123)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="String should have at least 1 character"):
        fallback_models._validate_version_value("")  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="Version field cannot be empty"):
        fallback_models._validate_version_value("   ")  # type: ignore[attr-defined]
    assert fallback_models._validate_version_value("3.0") == "3.0"  # type: ignore[attr-defined]


def test_fallback_config_coerces_portfolio_controls(
    fallback_models: ModuleType,
) -> None:
    Config = fallback_models.Config  # type: ignore[attr-defined]
    cfg = Config(
        **_base_config_payload(
            portfolio={
                "transaction_cost_bps": "2.5",
                "max_turnover": "1.25",
            }
        )
    )
    assert cfg.portfolio["transaction_cost_bps"] == pytest.approx(2.5)
    assert cfg.portfolio["max_turnover"] == pytest.approx(1.25)
    assert cfg.output is None
    assert cfg.multi_period is None


def test_fallback_config_rejects_invalid_portfolio_values(
    fallback_models: ModuleType,
) -> None:
    Config = fallback_models.Config  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="portfolio must be a dictionary"):
        Config(**_base_config_payload(portfolio=[]))

    with pytest.raises(ValueError, match="transaction_cost_bps must be >= 0"):
        Config(**_base_config_payload(portfolio={"transaction_cost_bps": -0.1}))

    with pytest.raises(ValueError, match="max_turnover must be <= 2.0"):
        Config(**_base_config_payload(portfolio={"max_turnover": 3.0}))


def test_fallback_config_requires_dict_sections(fallback_models: ModuleType) -> None:
    Config = fallback_models.Config  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="data section is required"):
        Config(**_base_config_payload(data=None))


def test_preset_config_requires_name(fallback_models: ModuleType) -> None:
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
    ColumnMapping = fallback_models.ColumnMapping  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="Date column must be specified"):
        ColumnMapping()

    with pytest.raises(
        ValueError, match="At least one return column must be specified"
    ):
        ColumnMapping(date_column="Date", return_columns=[])

    mapping = ColumnMapping(
        date_column="Date",
        return_columns=["FundA"],
        benchmark_column=None,
        risk_free_column=None,
    )
    assert mapping.column_display_names == {}
    assert mapping.column_tickers == {}


def test_configuration_state_defaults(fallback_models: ModuleType) -> None:
    ConfigurationState = fallback_models.ConfigurationState  # type: ignore[attr-defined]
    state = ConfigurationState()
    assert state.preset_name == ""
    assert state.column_mapping is None
    assert state.config_dict == {}
    assert state.uploaded_data is None
    assert state.analysis_results is None


def test_list_available_presets_and_load_preset(
    fallback_models: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "defaults.yml").write_text("version: 1\n", encoding="utf-8")
    preset_payload = _base_config_payload(description="demo")
    (config_dir / "alpha.yml").write_text(
        yaml.safe_dump({**preset_payload, "name": "alpha"}),
        encoding="utf-8",
    )
    (config_dir / "beta.yml").write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(
        fallback_models,
        "_find_config_directory",
        lambda: config_dir,
        raising=False,
    )

    assert fallback_models.list_available_presets() == ["alpha", "beta"]  # type: ignore[attr-defined]

    preset = fallback_models.load_preset("alpha")  # type: ignore[attr-defined]
    assert preset.name == "alpha"
    assert preset.description == "demo"

    with pytest.raises(TypeError, match="Preset file must contain a mapping"):
        fallback_models.load_preset("beta")  # type: ignore[attr-defined]


def test_load_config_fallback_handles_validation_failure(
    fallback_models: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def boom(data: dict[str, Any], *, base_path: Path) -> None:
        calls.append(copy.deepcopy(data))
        raise RuntimeError("validator failure")

    monkeypatch.setattr(
        fallback_models,
        "validate_trend_config",
        boom,
        raising=False,
    )

    payload = _base_config_payload()
    cfg = fallback_models.load_config(payload)

    assert isinstance(cfg, fallback_models.Config)  # type: ignore[attr-defined]
    assert cfg.version == payload["version"]
    # Ensure validation was attempted even though the error was swallowed.
    assert calls and calls[0]["version"] == payload["version"]


def test_list_available_presets_handles_missing_directory(
    fallback_models: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "config"
    monkeypatch.setattr(
        fallback_models,
        "_find_config_directory",
        lambda: missing,
        raising=False,
    )
    assert fallback_models.list_available_presets() == []  # type: ignore[attr-defined]


def _write_config_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_merges_output_settings_and_calls_validator(
    fallback_models: ModuleType, tmp_path: Path
) -> None:
    cfg_file = tmp_path / "config.yml"
    payload = _base_config_payload(
        export={"formats": "csv"},
        output={
            "format": ["CSV", "xlsx"],
            "path": str(tmp_path / "exports" / "demo.xlsx"),
        },
    )
    _write_config_file(cfg_file, payload)

    cfg = fallback_models.load(cfg_file)  # type: ignore[attr-defined]
    assert sorted(cfg.export["formats"]) == ["csv", "xlsx"]
    assert cfg.export["directory"].endswith("exports")
    assert cfg.export["filename"] == "demo.xlsx"

    assert fallback_models._validate_calls[-1][1] == tmp_path  # type: ignore[attr-defined]


def test_load_uses_environment_variable_when_path_missing(
    fallback_models: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_file = tmp_path / "env_config.yml"
    payload = _base_config_payload(output={"format": "json"})
    _write_config_file(cfg_file, payload)

    monkeypatch.setenv("TREND_CFG", str(cfg_file))

    cfg = fallback_models.load()  # type: ignore[attr-defined]
    assert "json" in cfg.export["formats"]
    assert fallback_models._validate_calls[-1][1] == cfg_file.parent  # type: ignore[attr-defined]


def test_load_accepts_direct_mapping(fallback_models: ModuleType) -> None:
    payload = _base_config_payload()
    cfg = fallback_models.load(payload)  # type: ignore[attr-defined]
    assert cfg.version == "1.0"
    assert fallback_models._validate_calls[-1][1] == Path.cwd()  # type: ignore[attr-defined]


def test_load_config_accepts_mapping_and_path(
    fallback_models: ModuleType, tmp_path: Path
) -> None:
    payload = _base_config_payload()
    cfg = fallback_models.load_config(payload)  # type: ignore[attr-defined]
    assert cfg.version == "1.0"
    assert fallback_models._validate_calls[-1][1] == Path.cwd()  # type: ignore[attr-defined]

    cfg_file = tmp_path / "config.yml"
    _write_config_file(cfg_file, payload)
    cfg = fallback_models.load_config(str(cfg_file))  # type: ignore[attr-defined]
    assert cfg.version == "1.0"
    assert fallback_models._validate_calls[-1][1] == tmp_path  # type: ignore[attr-defined]


def test_load_config_rejects_unsupported_types(fallback_models: ModuleType) -> None:
    with pytest.raises(TypeError, match="cfg must be a mapping or path"):
        fallback_models.load_config(3.14)  # type: ignore[attr-defined]
