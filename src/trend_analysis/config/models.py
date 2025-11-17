"""Configuration models for Streamlit Configure page validation.

This module supports environments with or without Pydantic installed.
Tests import the module twice (with and without Pydantic) and expect the
symbol ``_HAS_PYDANTIC`` to reflect availability.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Protocol, cast

import yaml

# ``models.py`` is executed under different module names in the test suite so we
# import ``validate_trend_config`` via its fully-qualified path to avoid
# relative-import resolution against the temporary alias (for example when the
# module is loaded as ``tests.config_models_fallback``).
# Import primary validator; define a lightweight fallback only if initial import fails.
try:  # pragma: no cover - normal path
    from trend_analysis.config.model import validate_trend_config

    _fallback_validate_trend_config_ref: _ValidateConfigFn | None = None
except Exception:  # pragma: no cover - fallback when model unavailable

    def _fallback_validate_trend_config(
        data: dict[str, Any], *, base_path: Path
    ) -> Any:  # pragma: no cover - exercised only in absence
        version = data.get("version")
        if not isinstance(version, str):
            raise ValueError("version must be a string")
        return {
            "version": version,
            "data": data.get("data", {}),
            "portfolio": data.get("portfolio", {}),
            "vol_adjust": data.get("vol_adjust", {}),
        }

    validate_trend_config = _fallback_validate_trend_config


class _ValidateConfigFn(Protocol):
    def __call__(self, data: dict[str, Any], *, base_path: Path) -> Any:
        """Validate configuration data and optionally return a model."""


class ConfigProtocol(Protocol):
    """Type protocol for Config class that works in both Pydantic and fallback
    modes."""

    version: str
    data: dict[str, Any]
    preprocessing: dict[str, Any]
    vol_adjust: dict[str, Any]
    sample_split: dict[str, Any]
    portfolio: dict[str, Any]
    benchmarks: dict[str, str]
    metrics: dict[str, Any]
    regime: dict[str, Any]
    export: dict[str, Any]
    output: dict[str, Any] | None
    run: dict[str, Any]
    multi_period: dict[str, Any] | None
    jobs: int | None
    checkpoint_dir: str | None
    seed: int

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str: ...


ConfigType = ConfigProtocol

# Pydantic import (optional in tests)
# Use temporary underscored names within the branch, then export public names
# Provide explicit annotations so static type checkers accept reassignment
# across the try/except fallback paths.
_BaseModel: Any
_ValidationInfo: Any
try:  # pragma: no cover - exercised via tests toggling availability
    import pydantic as _pyd

    _BaseModel = cast(Any, _pyd.BaseModel)
    _Field = cast(Any, _pyd.Field)
    _ValidationInfo = cast(Any, _pyd.ValidationInfo)
    _field_validator = cast(Any, _pyd.field_validator)

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _BaseModel = object

    def _Field(*_args: Any, **_kwargs: Any) -> None:  # noqa: D401 - simple fallback
        """Fallback Field when Pydantic is unavailable."""
        return None

    _ValidationInfo = object

    def _field_validator(*_args: Any, **_kwargs: Any) -> Any:
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    _HAS_PYDANTIC = False

# Export names with broad Any types for static checkers
BaseModel: Any = cast(Any, _BaseModel)
Field: Any = cast(Any, _Field)
ValidationInfo: Any = cast(Any, _ValidationInfo)
field_validator: Any = cast(Any, _field_validator)


# Simple BaseModel that works without pydantic (used by fallback Config)
class SimpleBaseModel:
    """Simple base model for configuration validation."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with validation."""
        defaults = self._get_defaults()
        for key, value in defaults.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._validate()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default values for this model."""
        return {}

    def _validate(self) -> None:
        """Validate the configuration."""
        pass


def _find_config_directory() -> Path:
    """Locate the project's configuration directory.

    Starting from this file's location, walk up the directory tree until a
    ``config`` directory containing ``defaults.yml`` is found. If no suitable
    directory is discovered, a :class:`FileNotFoundError` is raised.
    """

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "config"
        if candidate.is_dir() and (candidate / "defaults.yml").exists():
            return candidate

    raise FileNotFoundError("Could not find 'config' directory")


def _validate_version_value(v: Any) -> str:
    """Validate the ``version`` field for both pydantic and fallback modes."""
    if not isinstance(v, str):
        # Tests expect a ``ValueError`` for wrong types when pydantic is not
        # available.  Using ``ValueError`` keeps behaviour consistent between
        # the pydantic-backed model (which raises ``ValidationError``) and the
        # simple fallback model used in this repository.
        raise ValueError("version must be a string")
    if len(v) == 0:
        # Match pydantic's wording for empty strings
        raise ValueError("String should have at least 1 character")
    if not v.strip():
        raise ValueError("Version field cannot be empty")
    return v


if _HAS_PYDANTIC:
    # Cache class identity across re-imports to keep isinstance checks stable
    import builtins as _bi

    _cached = getattr(_bi, "_TREND_CONFIG_CLASS", None)
    _runtime_base = BaseModel if _cached is None else _cached
    if _cached is None:
        setattr(_bi, "_TREND_CONFIG_CLASS", _runtime_base)

    if TYPE_CHECKING:  # pragma: no cover - typing aid only
        from pydantic import BaseModel as _TypedPydanticBaseModel

        PydanticConfigBase = _TypedPydanticBaseModel
    else:
        PydanticConfigBase = cast(type[BaseModel], _runtime_base)

    # Provide a typed decorator wrapper to satisfy mypy in strict mode
    from typing import Callable, TypeVar

    F = TypeVar("F")

    def _fv_typed(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        return cast(Callable[[F], F], field_validator(*args, **kwargs))

    class _PydanticConfigImpl(PydanticConfigBase):
        """Typed access to the YAML configuration (Pydantic mode)."""

        # Field lists generated dynamically from model fields to prevent maintenance burden
        OPTIONAL_DICT_FIELDS: ClassVar[set[str]] = {"performance", "signals", "regime"}

        @classmethod
        def _dict_field_names(cls) -> List[str]:
            """Return names of fields whose type is dict[str, Any] (or
            compatible)."""
            # Support both Pydantic v2 (model_fields) and v1 (__fields__)
            fields_map = getattr(cls, "model_fields", {})

            # items() for both dict-like types
            def _items(obj: Any) -> list[tuple[str, Any]]:
                try:
                    return list(obj.items())
                except Exception:
                    return []

            items: list[tuple[str, Any]] = _items(fields_map)

            def _is_dict_type(tp: Any) -> bool:
                # Python 3.8+ typing origin helper
                try:
                    from typing import get_origin as _get_origin

                    origin = cast(Any, _get_origin(tp))
                except Exception:  # pragma: no cover - fallback
                    origin = getattr(tp, "__origin__", None)
                if not (origin is dict or tp is dict):
                    return False
                # Prefer to include only dict[str, Any]-like annotations
                try:
                    from typing import get_args as _get_args

                    args = _get_args(tp)
                except Exception:  # pragma: no cover - fallback
                    args = getattr(tp, "__args__", ())
                if len(args) == 2:
                    _key_t, val_t = args
                    # Exclude specific concrete types (e.g., str) for value
                    if (
                        getattr(val_t, "__module__", "") == "typing"
                        and getattr(val_t, "__qualname__", "") == "Any"
                    ):
                        return True
                    # If value annotation is 'Any' from typing, above returns True.
                    # Otherwise, do not include (filters out dict[str, str])
                    return False
                # If no args, fall back to including
                return True

            optional_fields = cast(
                set[str], getattr(cls, "OPTIONAL_DICT_FIELDS", set())
            )
            result: List[str] = []
            for name, field in items:
                tp = getattr(field, "annotation", None)
                if tp is None:
                    tp = getattr(field, "outer_type_", None)
                if _is_dict_type(tp) and name not in optional_fields:
                    result.append(name)
            return result

        # Placeholders; computed after class creation for reliability
        REQUIRED_DICT_FIELDS: ClassVar[List[str]] = []
        ALL_FIELDS: ClassVar[List[str]] = []

        # Use a plain dict for model_config to avoid type-checker issues when
        # Pydantic is not installed (tests toggle availability).
        model_config = {"extra": "ignore"}
        # ``version`` must be a non-empty string. ``min_length`` handles the empty
        # string case and produces the standard pydantic error message
        # "String should have at least 1 character". A separate validator below
        # ensures the field isn't composed solely of whitespace.
        version: str = Field(min_length=1)
        data: dict[str, Any] = Field(default_factory=dict)
        preprocessing: dict[str, Any] = Field(default_factory=dict)
        vol_adjust: dict[str, Any] = Field(default_factory=dict)
        sample_split: dict[str, Any] = Field(default_factory=dict)
        portfolio: dict[str, Any] = Field(default_factory=dict)
        benchmarks: dict[str, str] = Field(default_factory=dict)
        metrics: dict[str, Any] = Field(default_factory=dict)
        regime: dict[str, Any] = Field(default_factory=dict)
        signals: dict[str, Any] = Field(default_factory=dict)
        export: dict[str, Any] = Field(default_factory=dict)
        performance: dict[str, Any] = Field(default_factory=dict)
        output: dict[str, Any] | None = None
        run: dict[str, Any] = Field(default_factory=dict)
        multi_period: dict[str, Any] | None = None
        jobs: int | None = None
        checkpoint_dir: str | None = None
        seed: int = 42

        @_fv_typed("version", mode="before")
        def _validate_version(cls, v: Any) -> str:  # noqa: N805 - pydantic validator
            """Reject strings that consist only of whitespace."""
            return _validate_version_value(v)

        @_fv_typed(
            "data",
            "preprocessing",
            "vol_adjust",
            "sample_split",
            "portfolio",
            "metrics",
            "regime",
            "export",
            "run",
            mode="before",
        )
        def _ensure_dict(cls, v: Any, info: Any) -> dict[str, Any]:
            field_name = getattr(info, "field_name", "field")
            if v is None:
                # Maintain backwards-compatible error message checked in tests.
                raise ValueError(f"{field_name} section is required")
            if not isinstance(v, dict):
                raise ValueError(f"{field_name} must be a dictionary")
            return v

        @_fv_typed("portfolio", mode="after")
        def _validate_portfolio_controls(
            cls, v: dict[str, Any]
        ) -> dict[str, Any]:  # noqa: N805 - pydantic validator
            """Validate and normalise turnover / transaction cost controls.

            Backwards compatible: silently coerces numeric strings and ignores
            missing keys. Only raises when values are present but invalid.
            """
            if not isinstance(v, dict):  # defensive (already checked above)
                return v
            # Transaction cost (basis points per 1 unit turnover)
            if "transaction_cost_bps" in v:
                raw = v["transaction_cost_bps"]
                try:
                    tc = float(raw)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError("transaction_cost_bps must be numeric") from exc
                if tc < 0:
                    raise ValueError("transaction_cost_bps must be >= 0")
                v["transaction_cost_bps"] = tc
            if "slippage_bps" in v:
                raw = v["slippage_bps"]
                try:
                    slip = float(raw)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError("slippage_bps must be numeric") from exc
                if slip < 0:
                    raise ValueError("slippage_bps must be >= 0")
                v["slippage_bps"] = slip
            # Max turnover cap (fraction of portfolio; 1.0 = effectively uncapped)
            if "max_turnover" in v:
                raw = v["max_turnover"]
                try:
                    mt = float(raw)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError("max_turnover must be numeric") from exc
                if mt < 0:
                    raise ValueError("max_turnover must be >= 0")
                # Allow values >1.0 (full liquidation + rebuild = 2.0 theoretical upper)
                if mt > 2.0:
                    raise ValueError("max_turnover must be <= 2.0")
                v["max_turnover"] = mt
            return v

    # Field constants are already defined as class variables above

    # Only cache when creating a fresh class
    if _cached is None:
        setattr(_bi, "_TREND_CONFIG_CLASS", _PydanticConfigImpl)

    # Compute class-level field lists post definition (works for v1/v2)
    _fields_map = getattr(_PydanticConfigImpl, "model_fields", {})
    try:
        _field_names = list(_fields_map.keys())
    except Exception:  # pragma: no cover
        _field_names = list(_fields_map)
    setattr(_PydanticConfigImpl, "ALL_FIELDS", _field_names)
    setattr(
        _PydanticConfigImpl,
        "REQUIRED_DICT_FIELDS",
        _PydanticConfigImpl._dict_field_names(),
    )

else:  # Fallback mode for tests without pydantic

    class _FallbackConfig(SimpleBaseModel):
        """Simplified Config for environments without Pydantic."""

        # Field lists as class constants to prevent maintenance burden
        REQUIRED_DICT_FIELDS: ClassVar[List[str]] = [
            "data",
            "preprocessing",
            "vol_adjust",
            "sample_split",
            "portfolio",
            "metrics",
            "export",
            "run",
        ]

        ALL_FIELDS: ClassVar[List[str]] = [
            "version",
            "data",
            "preprocessing",
            "vol_adjust",
            "sample_split",
            "portfolio",
            "benchmarks",
            "metrics",
            "regime",
            "signals",
            "export",
            "performance",
            "output",
            "run",
            "multi_period",
            "jobs",
            "checkpoint_dir",
            "seed",
        ]

        OPTIONAL_DICT_FIELDS: ClassVar[set[str]] = {"performance", "signals", "regime"}

        # Attribute declarations for linters/type-checkers
        version: str
        data: Dict[str, Any]
        preprocessing: Dict[str, Any]
        vol_adjust: Dict[str, Any]
        sample_split: Dict[str, Any]
        portfolio: Dict[str, Any]
        benchmarks: Dict[str, str]
        metrics: Dict[str, Any]
        regime: Dict[str, Any]
        signals: Dict[str, Any]
        export: Dict[str, Any]
        performance: Dict[str, Any]
        output: Dict[str, Any] | None
        run: Dict[str, Any]
        multi_period: Dict[str, Any] | None
        jobs: int | None
        checkpoint_dir: str | None
        seed: int

        def _get_defaults(self) -> Dict[str, Any]:
            return {
                "data": {},
                "preprocessing": {},
                "vol_adjust": {},
                "sample_split": {},
                "portfolio": {},
                "benchmarks": {},
                "metrics": {},
                "regime": {},
                "signals": {},
                "export": {},
                "performance": {},
                "output": None,
                "run": {},
                "multi_period": None,
                "jobs": None,
                "checkpoint_dir": None,
                "seed": 42,
            }

        def _validate(self) -> None:  # Simple runtime validation
            if getattr(self, "version", None) is None:
                raise ValueError("version field is required")
            if not isinstance(self.version, str):
                raise ValueError("version must be a string")
            if len(self.version) == 0:
                raise ValueError("String should have at least 1 character")
            if not self.version.strip():
                raise ValueError("Version field cannot be empty")

            for field in [
                "data",
                "preprocessing",
                "vol_adjust",
                "sample_split",
                "portfolio",
                "metrics",
                "regime",
                "export",
                "performance",
                "run",
            ]:
                value = getattr(self, field, None)
                if value is None:
                    raise ValueError(f"{field} section is required")
                if not isinstance(value, dict):
                    raise ValueError(f"{field} must be a dictionary")
            # Light-weight validation for turnover / cost controls
            port = getattr(self, "portfolio", {})
            if isinstance(port, dict):
                if "transaction_cost_bps" in port:
                    try:
                        tc = float(port["transaction_cost_bps"])
                    except Exception as exc:  # pragma: no cover - defensive
                        raise ValueError(
                            "transaction_cost_bps must be numeric"
                        ) from exc
                    if tc < 0:
                        raise ValueError("transaction_cost_bps must be >= 0")
                    port["transaction_cost_bps"] = tc
                if "slippage_bps" in port:
                    try:
                        slip = float(port["slippage_bps"])
                    except Exception as exc:  # pragma: no cover - defensive
                        raise ValueError("slippage_bps must be numeric") from exc
                    if slip < 0:
                        raise ValueError("slippage_bps must be >= 0")
                    port["slippage_bps"] = slip
                if "max_turnover" in port:
                    try:
                        mt = float(port["max_turnover"])
                    except Exception as exc:  # pragma: no cover - defensive
                        raise ValueError("max_turnover must be numeric") from exc
                    if mt < 0:
                        raise ValueError("max_turnover must be >= 0")
                    if mt > 2.0:
                        raise ValueError("max_turnover must be <= 2.0")
                    port["max_turnover"] = mt

        # Provide a similar API surface to pydantic for callers
        def model_dump(self) -> Dict[str, Any]:
            return {k: getattr(self, k) for k in self.ALL_FIELDS}

        def model_dump_json(self, *, indent: int | None = None) -> str:
            return json.dumps(self.model_dump(), indent=indent, sort_keys=True)

    # Fallback does not modify package attributes at runtime

# Public alias selected at runtime for callers
if _HAS_PYDANTIC:
    Config = cast("type[ConfigType]", globals().get("_PydanticConfigImpl"))
else:
    Config = cast("type[ConfigType]", globals().get("_FallbackConfig"))


class PresetConfig(SimpleBaseModel):
    """Configuration preset with validation."""

    # Field lists as class constants to prevent maintenance burden
    PRESET_DICT_FIELDS: ClassVar[List[str]] = [
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "metrics",
        "export",
        "run",
    ]

    OPTIONAL_DICT_FIELDS: ClassVar[set[str]] = {"signals"}

    name: str
    description: str
    data: Dict[str, Any]
    preprocessing: Dict[str, Any]
    vol_adjust: Dict[str, Any]
    sample_split: Dict[str, Any]
    portfolio: Dict[str, Any]
    metrics: Dict[str, Any]
    signals: Dict[str, Any]
    export: Dict[str, Any]
    run: Dict[str, Any]

    def _get_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {field: {} for field in self.PRESET_DICT_FIELDS}
        for optional_field in self.OPTIONAL_DICT_FIELDS:
            defaults.setdefault(optional_field, {})
        return defaults

    def _validate(self) -> None:
        """Validate preset configuration."""
        if not self.name or not self.name.strip():
            raise ValueError("Preset name must be specified")


class ColumnMapping(SimpleBaseModel):
    """Column mapping configuration for uploaded data."""

    # Attribute declarations for type checkers
    date_column: str
    return_columns: List[str]
    benchmark_column: str | None
    risk_free_column: str | None
    column_display_names: Dict[str, str]
    column_tickers: Dict[str, str]

    def __init__(
        self,
        date_column: str = "",
        return_columns: List[str] | None = None,
        benchmark_column: str | None = None,
        risk_free_column: str | None = None,
        column_display_names: Dict[str, str] | None = None,
        column_tickers: Dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        if return_columns is None:
            return_columns = []
        if column_display_names is None:
            column_display_names = {}
        if column_tickers is None:
            column_tickers = {}
        super().__init__(
            date_column=date_column,
            return_columns=return_columns,
            benchmark_column=benchmark_column,
            risk_free_column=risk_free_column,
            column_display_names=column_display_names,
            column_tickers=column_tickers,
            **kwargs,
        )

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "date_column": "",
            "return_columns": [],
            "benchmark_column": None,
            "risk_free_column": None,
            "column_display_names": {},
            "column_tickers": {},
        }

    def _validate(self) -> None:
        """Validate column mapping."""
        if not self.date_column or not self.date_column.strip():
            raise ValueError("Date column must be specified")

        if not self.return_columns:
            raise ValueError("At least one return column must be specified")


class ConfigurationState(SimpleBaseModel):
    """Complete configuration state for the Streamlit app."""

    preset_name: str
    column_mapping: ColumnMapping | None
    config_dict: Dict[str, Any]
    uploaded_data: Any
    analysis_results: Any

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "preset_name": "",
            "column_mapping": None,
            "config_dict": {},
            "uploaded_data": None,
            "analysis_results": None,
        }

    def _validate(self) -> None:
        """Validate configuration state."""
        pass


def load_preset(preset_name: str) -> PresetConfig:
    """Load a preset configuration from file."""
    # Find the config directory relative to this file
    config_dir = _find_config_directory()
    preset_file = config_dir / f"{preset_name}.yml"

    if not preset_file.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_file}")

    with preset_file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise TypeError("Preset file must contain a mapping")

    data["name"] = preset_name
    return PresetConfig(**data)


def list_available_presets() -> List[str]:
    """List all available preset names."""
    config_dir = _find_config_directory()

    if not config_dir.exists():
        return []

    presets = []
    for yml_file in config_dir.glob("*.yml"):
        if yml_file.name not in ["defaults.yml"]:  # Exclude defaults
            presets.append(yml_file.stem)

    return sorted(presets)


DEFAULTS = Path(__file__).resolve().parents[3] / "config" / "defaults.yml"


def load_config(cfg: Mapping[str, Any] | str | Path) -> ConfigProtocol:
    """Load configuration from a mapping or file path."""
    if isinstance(cfg, (str, Path)):
        return load(cfg)
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping or path")
    cfg_dict = dict(cfg)
    # Early version validation for mapping-based load to surface version
    # errors directly (tests accept ValueError here) regardless of Pydantic.
    if "version" in cfg_dict:
        _validate_version_value(cfg_dict["version"])  # raises ValueError on failure
    # Defer generic required-section checks to Pydantic so tests see
    # field-specific messages unless the user explicitly set a section to None.
    required_sections = [
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "metrics",
        "export",
        "run",
    ]
    for section in required_sections:
        if section in cfg_dict and cfg_dict[section] is None:
            # Preserve classic message for explicit null
            raise ValueError(f"{section} section is required")
        # If section missing entirely, let Pydantic raise (when available).
        if section in cfg_dict and not isinstance(cfg_dict[section], dict):
            # Preserve type-specific message
            raise ValueError(f"{section} must be a dictionary")
    pydantic_present = sys.modules.get("pydantic") is not None
    if _HAS_PYDANTIC:
        # Allow ValidationError to propagate (tests expect this)
        validate_trend_config(cfg_dict, base_path=Path.cwd())
    else:
        validator_module = str(getattr(validate_trend_config, "__module__", ""))
        if (not pydantic_present) or validator_module.startswith(
            "trend_analysis.config"
        ):
            try:
                validate_trend_config(cfg_dict, base_path=Path.cwd())
            except Exception:
                pass
    return Config(**cfg_dict)


def load(path: str | Path | None = None) -> ConfigProtocol:
    """Load configuration from ``path`` or ``DEFAULTS``.

    If ``path`` is ``None``, the ``TREND_CFG`` environment variable is
    consulted before falling back to ``DEFAULTS``.
    If ``path`` is a dict, it is used directly as configuration data.
    """
    base_dir = Path.cwd()
    if isinstance(path, dict):
        data = path.copy()
    elif path is None:
        env = os.environ.get("TREND_CONFIG") or os.environ.get("TREND_CFG")
        cfg_path = Path(env) if env else DEFAULTS
        base_dir = cfg_path.parent
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise TypeError("Config file must contain a mapping")
    else:
        cfg_path = Path(path)
        base_dir = cfg_path.parent
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise TypeError("Config file must contain a mapping")

    out_cfg = data.pop("output", None)
    if isinstance(out_cfg, dict):
        export_cfg = data.setdefault("export", {})
        fmt = out_cfg.get("format")
        if fmt:
            fmt_list = [fmt] if isinstance(fmt, str) else list(fmt)
            existing = export_cfg.get("formats")
            if isinstance(existing, str):
                combined = [str(existing)]
            elif isinstance(existing, (list, tuple, set)):
                combined = [str(item) for item in existing]
            else:
                combined = []
            seen = {item.lower() for item in combined}
            for item in fmt_list:
                item_str = str(item)
                key = item_str.lower()
                if key not in seen:
                    combined.append(item_str)
                    seen.add(key)
            export_cfg["formats"] = combined if combined else [str(v) for v in fmt_list]
        path_val = out_cfg.get("path")
        if path_val:
            p = Path(path_val)
            export_cfg.setdefault("directory", str(p.parent) if p.parent else ".")
            export_cfg.setdefault("filename", p.name)

    # Version validation (type / whitespace). Let ValueError propagate in fallback.
    if "version" in data and not _HAS_PYDANTIC:
        _validate_version_value(data["version"])  # fallback mode only

    required_sections = [
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "metrics",
        "export",
        "run",
    ]
    for section in required_sections:
        if section in data and data[section] is None:
            raise ValueError(f"{section} section is required")
        if section in data and not isinstance(data[section], dict):
            raise ValueError(f"{section} must be a dictionary")
    validated: Any | None = None
    pydantic_present = sys.modules.get("pydantic") is not None
    if _HAS_PYDANTIC:
        validated = validate_trend_config(data, base_path=base_dir)
    else:
        validator_module = str(getattr(validate_trend_config, "__module__", ""))
        if (not pydantic_present) or validator_module.startswith(
            "trend_analysis.config"
        ):
            try:
                validated = validate_trend_config(data, base_path=base_dir)
            except Exception:
                validated = None

    if isinstance(validated, Config):
        return validated
    if hasattr(validated, "model_dump"):
        dumped = cast(Any, validated).model_dump()
        if isinstance(dumped, Mapping):
            merged: dict[str, Any] = dict(data)
            for key, value in dumped.items():
                merged[key] = value
            if "version" not in merged and "version" in data:
                merged["version"] = data["version"]
            return Config(**merged)
    if isinstance(validated, Mapping):
        return Config(**dict(validated))
    return Config(**data)


__all__ = [
    "Config",
    "ConfigType",
    "load",
    "PresetConfig",
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
    "load_config",
    "DEFAULTS",
]
