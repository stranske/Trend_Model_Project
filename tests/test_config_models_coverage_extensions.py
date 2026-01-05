"""Additional coverage tests for ``trend_analysis.config.models``."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from trend_analysis.config import models

MODULE_PATH = Path(models.__file__)


def _load_models_without_pydantic(
    monkeypatch: pytest.MonkeyPatch, name: str = "tests.config_models_fallback_cov"
):
    """Load a fresh copy of the `trend_analysis.config.models` module with
    `pydantic` unavailable.

    This function simulates an environment where the `pydantic` package is not present,
    by setting `sys.modules["pydantic"]` to `None` using the provided `monkeypatch`.
    It then loads the module from disk under the given `name`, allowing tests to verify
    fallback behavior when `pydantic` cannot be imported.

    Parameters:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching `sys.modules`.
        name (str): The name to assign to the loaded module in `sys.modules`. Defaults to
            "tests.config_models_fallback_cov".

    Returns:
        types.ModuleType: The loaded module object with `pydantic` unavailable.
    """
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    monkeypatch.setitem(sys.modules, "pydantic", None)

    spec.loader.exec_module(module)
    return module


def _base_config_mapping() -> dict[str, Any]:
    return {
        "version": "1.0",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "metrics": {},
        "export": {},
        "run": {},
    }


def test_find_config_directory_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDefaults:
        def exists(self) -> bool:
            return False

    class _FakeCandidate:
        def __init__(self, label: str) -> None:
            self.label = label

        def __truediv__(self, item: str):  # type: ignore[override]
            if item == "defaults.yml":
                return _FakeDefaults()
            return _FakeCandidate(f"{self.label}/{item}")

        def is_dir(self) -> bool:
            return False

    class _FakeResolved:
        def __init__(self) -> None:
            self.parents = [_FakeCandidate("one"), _FakeCandidate("two")]

        def __truediv__(self, item: str):  # type: ignore[override]
            return _FakeCandidate(f"resolved/{item}")

    class _FakePath:
        def __init__(self, *_: Any, **__: Any) -> None:
            self._resolved = _FakeResolved()

        def resolve(self) -> _FakeResolved:
            return self._resolved

    monkeypatch.setattr(models, "Path", _FakePath)

    with pytest.raises(FileNotFoundError):
        models._find_config_directory()


def test_dict_field_names_handles_varied_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = models._PydanticConfigImpl  # type: ignore[attr-defined]

    field_plain = types.SimpleNamespace(annotation=dict)
    field_outer = types.SimpleNamespace(annotation=None, outer_type_=dict[str, Any])
    field_specific = types.SimpleNamespace(annotation=dict[str, str])

    monkeypatch.setattr(
        cls,
        "model_fields",
        {"plain": field_plain, "outer": field_outer, "specific": field_specific},
        raising=False,
    )
    monkeypatch.setattr(cls, "OPTIONAL_DICT_FIELDS", set())

    names = cls._dict_field_names()
    assert "plain" in names  # generic dict triggers len(args) == 0 path
    assert "outer" in names  # annotation fallback to outer_type_
    assert "specific" not in names  # value type not Any → excluded


def test_ensure_dict_rejects_none_and_non_mappings() -> None:
    info = types.SimpleNamespace(field_name="metrics")
    validator = models._PydanticConfigImpl._ensure_dict.__func__  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="metrics section is required"):
        validator(models._PydanticConfigImpl, None, info)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="metrics must be a dictionary"):
        validator(models._PydanticConfigImpl, [1, 2, 3], info)  # type: ignore[attr-defined]


def test_validate_portfolio_controls_handles_edge_cases() -> None:
    validator = models._PydanticConfigImpl._validate_portfolio_controls.__func__  # type: ignore[attr-defined]

    assert validator(models._PydanticConfigImpl, "skip") == "skip"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="transaction_cost_bps must be >= 0"):
        validator(models._PydanticConfigImpl, {"transaction_cost_bps": "-1"})  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="max_turnover must be >= 0"):
        validator(models._PydanticConfigImpl, {"max_turnover": "-0.5"})  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="max_turnover must be <= 2.0"):
        validator(models._PydanticConfigImpl, {"max_turnover": "2.5"})  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="lambda_tc must be >= 0"):
        validator(models._PydanticConfigImpl, {"lambda_tc": "-0.1"})  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="lambda_tc must be <= 1"):
        validator(models._PydanticConfigImpl, {"lambda_tc": "1.5"})  # type: ignore[attr-defined]

    validated = validator(
        models._PydanticConfigImpl,
        {"transaction_cost_bps": "15", "max_turnover": "1.5", "lambda_tc": "0.3"},
    )  # type: ignore[attr-defined]
    assert validated["transaction_cost_bps"] == pytest.approx(15.0)
    assert validated["max_turnover"] == pytest.approx(1.5)
    assert validated["lambda_tc"] == pytest.approx(0.3)


def test_pydantic_config_cache_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    if hasattr(builtins, "_TREND_CONFIG_CLASS"):
        delattr(builtins, "_TREND_CONFIG_CLASS")

    module_name = "tests.config_models_cache_probe"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(builtins, "_TREND_CONFIG_CLASS")
    assert builtins._TREND_CONFIG_CLASS is module.Config  # type: ignore[attr-defined]

    monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_load_config_validates_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_validate(cfg: dict[str, Any], *, base_path: Path) -> dict[str, Any]:
        return cfg

    monkeypatch.setattr(models, "validate_trend_config", fake_validate)

    config = _base_config_mapping()
    config["data"] = None
    with pytest.raises(ValueError, match="data section is required"):
        models.load_config(config)

    config = _base_config_mapping()
    config["metrics"] = []
    with pytest.raises(ValueError, match="metrics must be a dictionary"):
        models.load_config(config)

    result = models.load_config(_base_config_mapping())
    assert isinstance(result, models.Config)


def test_load_merges_output_formats_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_validate(cfg: dict[str, Any], *, base_path: Path) -> dict[str, Any]:
        return cfg

    monkeypatch.setattr(models, "validate_trend_config", fake_validate)

    data = _base_config_mapping()
    data["export"] = {"formats": ("json", "txt")}
    data["output"] = {
        "format": ["csv", "JSON"],
        "path": tmp_path / "reports" / "summary.xlsx",
    }

    cfg = models.load(data)
    export_cfg = cfg.export  # type: ignore[attr-defined]
    assert {fmt.lower() for fmt in export_cfg["formats"]} == {"json", "txt", "csv"}
    assert export_cfg["directory"].endswith("reports")
    assert export_cfg["filename"] == "summary.xlsx"


def test_load_applies_validator_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _base_config_mapping()

    class DummyModel:
        def model_dump(self) -> dict[str, Any]:
            return {"export": {"formats": ["csv"]}, "extra": {"enabled": True}}

    monkeypatch.setattr(models, "validate_trend_config", lambda *_args, **_kwargs: DummyModel())
    cfg = models.load(dict(data))
    assert cfg.export["formats"] == ["csv"]
    dumped = cfg.model_dump()
    assert "export" in dumped and dumped["export"]["formats"] == ["csv"]

    monkeypatch.setattr(
        models,
        "validate_trend_config",
        lambda *_args, **_kwargs: {"version": data["version"], "metrics": {"alpha": 1}},
    )
    cfg = models.load(dict(data))
    assert cfg.metrics["alpha"] == 1

    config_instance = models.Config(**data)  # type: ignore[arg-type]
    monkeypatch.setattr(models, "validate_trend_config", lambda *_args, **_kwargs: config_instance)
    cfg = models.load(dict(data))
    assert cfg is config_instance


def test_column_mapping_defaults_cover_branches() -> None:
    mapping = models.ColumnMapping(
        date_column="Date",
        return_columns=["ret"],
        column_display_names=None,
        column_tickers=None,
    )
    assert mapping.column_display_names == {}
    assert mapping.column_tickers == {}


def test_fallback_config_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback = _load_models_without_pydantic(monkeypatch, "tests.config_models_fallback_cov1")

    with pytest.raises(ValueError, match="version field is required"):
        fallback.Config()

    with pytest.raises(ValueError, match="version must be a string"):
        fallback.Config(version=123)

    with pytest.raises(ValueError, match="String should have at least 1 character"):
        fallback.Config(version="", data={})

    with pytest.raises(ValueError, match="Version field cannot be empty"):
        fallback.Config(version="   ", data={})

    kwargs = _base_config_mapping()
    cfg = fallback.Config(**kwargs)
    assert cfg.model_dump()["seed"] == 42

    with pytest.raises(ValueError, match="transaction_cost_bps must be >= 0"):
        fallback.Config(**{**kwargs, "portfolio": {"transaction_cost_bps": -1}})

    with pytest.raises(ValueError, match="max_turnover must be <= 2.0"):
        fallback.Config(**{**kwargs, "portfolio": {"max_turnover": 3}})


def test_fallback_load_enforces_version(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback = _load_models_without_pydantic(monkeypatch, "tests.config_models_fallback_cov2")

    with pytest.raises(ValueError, match="version must be a string"):
        fallback.load({"version": 123})

    data = _base_config_mapping()
    fallback.validate_trend_config = lambda *_args, **_kwargs: {"version": data["version"]}
    cfg = fallback.load(dict(data))
    assert isinstance(cfg, fallback.Config)


def test_fallback_load_respects_validator_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = _load_models_without_pydantic(monkeypatch, "tests.config_models_fallback_cov3")

    data = _base_config_mapping()

    class DummyModel:
        def model_dump(self) -> dict[str, Any]:
            return {"portfolio": {"weights": [0.5, 0.5]}}

    fallback.validate_trend_config = lambda *_args, **_kwargs: DummyModel()
    cfg = fallback.load(dict(data))
    assert cfg.portfolio["weights"] == [0.5, 0.5]

    fallback.validate_trend_config = lambda *_args, **_kwargs: {
        "version": data["version"],
        "metrics": {"beta": 2},
    }
    cfg = fallback.load(dict(data))
    assert cfg.metrics["beta"] == 2
