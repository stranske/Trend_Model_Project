"""Generate JSON Schema artifacts for configuration files."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from utils.paths import proj_path

_DEFAULTS_FILE = proj_path() / "config" / "defaults.yml"
_CONFIG_DIR = proj_path() / "config"
_CONFIG_MAP = proj_path() / "docs" / "ConfigMap.md"
_SCHEMA_FILE = proj_path() / "config.schema.json"
_COMPACT_SCHEMA_FILE = proj_path() / "config.schema.compact.json"

# Manual overrides for optional types when defaults are null or empty.
_TYPE_OVERRIDES: dict[str, list[str] | str] = {
    "data.csv_path": ["string", "null"],
    "data.universe_membership_path": ["string", "null"],
    "data.managers_glob": ["string", "null"],
    "data.indices_glob": ["string", "null"],
    "data.risk_free_column": ["string", "null"],
    "data.missing_limit": ["integer", "null"],
    "data.missing_fill_limit": ["integer", "null"],
    "data.columns": ["array", "null"],
    "preprocessing.resample.target": ["string", "null"],
    "preprocessing.missing_data.limit": ["integer", "null"],
    "portfolio.rebalance_freq": ["string", "null"],
    "portfolio.manual_list": ["array"],
    "portfolio.rank.pct": ["number", "null"],
    "portfolio.rank.threshold": ["number", "null"],
    "portfolio.max_turnover": ["number", "null"],
    "output": ["object", "null"],
    "multi_period": ["object", "null"],
    "jobs": ["integer", "null"],
    "checkpoint_dir": ["string", "null"],
}

# Known constraints for validation and prompting.
_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "data.frequency": {"enum": ["D", "W", "M", "ME"]},
    "data.missing_policy": {"enum": ["drop", "ffill", "zero"]},
    "preprocessing.winsorise.limits": {"minItems": 2, "maxItems": 2},
    "preprocessing.resample.method": {"enum": ["last", "mean", "sum", "pad"]},
    "preprocessing.missing_data.policy": {"enum": ["drop", "ffill", "zero"]},
    "vol_adjust.window.decay": {"enum": ["ewma", "simple"]},
    "sample_split.method": {"enum": ["date", "ratio"]},
    "portfolio.selection_mode": {"enum": ["all", "random", "manual", "rank"]},
    "portfolio.weighting_scheme": {
        "enum": [
            "equal",
            "risk_parity",
            "hrp",
            "erc",
            "robust_mv",
            "robust_risk_parity",
            "custom",
        ]
    },
    "portfolio.rebalance_freq": {"enum": ["M", "Q", "A", None]},
    "portfolio.rank.inclusion_approach": {"enum": ["top_n", "top_pct", "threshold"]},
    "portfolio.constraints.max_weight": {"minimum": 0, "maximum": 1},
    "portfolio.leverage_cap": {"minimum": 0},
    "portfolio.transaction_cost_bps": {"minimum": 0},
    "portfolio.max_turnover": {"minimum": 0, "maximum": 2},
    "metrics.bootstrap_ci.ci_level": {"minimum": 0, "maximum": 1},
    "metrics.rf_rate_annual": {"minimum": 0},
    "vol_adjust.target_vol": {"minimum": 0},
    "vol_adjust.window.lambda": {"minimum": 0, "maximum": 1},
    "sample_split.ratio": {"minimum": 0, "maximum": 1},
}

# Free-form mapping sections allow dynamic keys with known value types.
_FREEFORM_MAPS: dict[str, dict[str, Any]] = {
    "benchmarks": {"type": "string"},
    "preprocessing.missing_data.per_asset": {"type": "string"},
    "preprocessing.missing_data.per_asset_limit": {"type": ["integer", "null"]},
    "portfolio.constraints.group_caps": {"type": "number"},
    "portfolio.rank.blended_weights": {"type": "number"},
    "portfolio.custom_weights": {"type": "number"},
    "portfolio.selector.params": {"type": ["number", "string", "boolean", "array", "object", "null"]},
    "portfolio.weighting.params": {"type": ["number", "string", "boolean", "array", "object", "null"]},
    "strategy.grid": {"type": ["array", "number", "string", "boolean", "object", "null"]},
}

# Manual descriptions for common fields that lack inline comments.
_MANUAL_DESCRIPTIONS: dict[str, str] = {
    "data.csv_path": "Path to a single returns CSV file.",
    "data.indices_glob": "Glob for benchmark index CSV inputs.",
    "data.universe_membership_path": "Optional CSV with universe membership overrides.",
    "data.columns": "Subset of columns to load from the returns CSV.",
    "data.missing_fill_limit": "Maximum consecutive periods to forward-fill missing data.",
    "preprocessing.steps": "Optional list of preprocessing steps to run in order.",
    "sample_split.in_start": "Start period for in-sample window.",
    "sample_split.in_end": "End period for in-sample window.",
    "sample_split.out_start": "Start period for out-of-sample window.",
    "sample_split.out_end": "End period for out-of-sample window.",
    "portfolio.custom_weights": "Explicit weight overrides keyed by manager name.",
    "output.format": "Output file format for legacy single-file runners.",
    "output.path": "Output path or prefix for legacy single-file runners.",
    "run.seed": "Random seed used for deterministic runs.",
    "run.jobs": "Parallel worker count for a run.",
    "run.checkpoint_dir": "Directory for saving checkpoints.",
    "jobs": "Legacy top-level worker count override.",
    "checkpoint_dir": "Legacy top-level checkpoint directory override.",
    "multi_period.frequency": "Frequency for multi-period windows.",
    "multi_period.in_sample_len": "Length of the in-sample window.",
    "multi_period.out_sample_len": "Length of the out-of-sample window.",
    "multi_period.start": "Start period for multi-period runs.",
    "multi_period.end": "End period for multi-period runs.",
    "walk_forward.train": "Walk-forward in-sample window length.",
    "walk_forward.test": "Walk-forward out-of-sample window length.",
    "walk_forward.step": "Walk-forward step length.",
    "strategy.top_n": "Top-N selection used by walk-forward strategies.",
    "strategy.defaults": "Default strategy parameters for walk-forward runs.",
    "strategy.grid": "Grid search parameter ranges for walk-forward runs.",
}


def collect_config_sources(config_dir: Path) -> list[Path]:
    """Return a stable list of config YAML files to scan for keys."""

    paths = sorted(config_dir.glob("*.yml"))
    preset_paths = sorted((config_dir / "presets").glob("*.yml"))
    return paths + preset_paths


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must contain a mapping at the root.")
    return data


def merge_defaults(base: Any, extra: Any) -> Any:
    """Merge ``extra`` into ``base`` without overriding existing defaults."""

    if isinstance(base, dict) and isinstance(extra, dict):
        merged: dict[str, Any] = {key: merge_defaults(value, extra.get(key)) for key, value in base.items()}
        for key, value in extra.items():
            if key not in merged:
                merged[key] = value
        return merged
    if base is None and extra is not None:
        return base
    return base if base is not None else extra


def gather_samples(configs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Capture first non-null sample values for each path."""

    samples: dict[str, Any] = {}

    def _walk(value: Any, path: list[str]) -> None:
        path_key = ".".join(path)
        if isinstance(value, dict):
            if path_key and path_key not in samples and value:
                samples[path_key] = value
            for key, child in value.items():
                _walk(child, path + [key])
            return
        if isinstance(value, list):
            if path_key and path_key not in samples and value:
                samples[path_key] = value
            if value:
                _walk(value[0], path + ["[]"])
            return
        if path_key and path_key not in samples and value is not None:
            samples[path_key] = value

    for cfg in configs:
        _walk(cfg, [])
    return samples


def _find_inline_comment(value: str) -> str | None:
    """Return the inline comment text for a YAML line (if any)."""

    in_single = False
    in_double = False
    for idx, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[idx + 1 :].strip()
    return None


def extract_inline_comments(path: Path) -> dict[str, str]:
    """Extract inline YAML comments keyed by dotted path."""

    comments: dict[str, str] = {}
    stack: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            continue
        match = re.match(r"^(\s*)([^:#]+):(.*)$", line)
        if not match:
            continue
        indent = len(match.group(1))
        key = match.group(2).strip().strip("'\"")
        level = indent // 2
        stack = stack[:level]
        stack.append(key)
        comment = _find_inline_comment(match.group(3))
        if comment:
            comments[".".join(stack)] = _sanitize_description(comment)
    return comments


def _sanitize_description(value: str) -> str:
    cleaned = value.strip().replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("≈", "~")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned


def _schema_root_description(config_map_path: Path) -> str:
    if not config_map_path.exists():
        return "Configuration schema generated from defaults and templates."
    for line in config_map_path.read_text(encoding="utf-8").splitlines():
        if "`config/defaults.yml`" in line:
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 3:
                return _sanitize_description(parts[2])
    return "Configuration schema generated from defaults and templates."


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _infer_array_items(values: list[Any], sample_item: Any | None) -> dict[str, Any]:
    if not values and sample_item is not None:
        return {"type": _infer_type(sample_item)}
    if not values:
        return {"type": "string"}
    item_types = sorted(
        {"integer" if isinstance(v, int) and not isinstance(v, bool) else _infer_type(v) for v in values}
    )
    if len(item_types) == 1:
        return {"type": item_types[0]}
    return {"type": item_types}


def _infer_schema_type(path_key: str, default_value: Any, sample: Any | None) -> list[str] | str:
    if path_key in _TYPE_OVERRIDES:
        return _TYPE_OVERRIDES[path_key]
    if default_value is None and sample is not None:
        return sorted({"null", _infer_type(sample)})
    return _infer_type(default_value)


def _description_for(path_key: str, comment_map: dict[str, str]) -> str:
    if path_key in comment_map:
        return comment_map[path_key]
    if path_key in _MANUAL_DESCRIPTIONS:
        return _MANUAL_DESCRIPTIONS[path_key]
    return f"Config option for {path_key}."


def _constraints_for(path_key: str) -> dict[str, Any]:
    return _CONSTRAINTS.get(path_key, {})


def _nl_editable(path_key: str) -> bool:
    lowered = path_key.lower()
    if any(token in lowered for token in ("path", "glob", "directory", "file", "csv", "log_file")):
        return False
    return True


def build_schema(
    value: Any,
    *,
    path: list[str],
    comment_map: dict[str, str],
    samples: dict[str, Any],
) -> dict[str, Any]:
    path_key = ".".join(path)
    schema: dict[str, Any] = {
        "description": _description_for(path_key, comment_map),
        "constraints": _constraints_for(path_key),
        "nl_editable": _nl_editable(path_key),
    }

    if isinstance(value, dict):
        schema["type"] = "object"
        schema["properties"] = {}
        schema["additionalProperties"] = False
        for key, child in value.items():
            schema["properties"][key] = build_schema(
                child,
                path=path + [key],
                comment_map=comment_map,
                samples=samples,
            )
        if path_key in _FREEFORM_MAPS:
            schema["additionalProperties"] = _FREEFORM_MAPS[path_key]
        schema["default"] = value
        return schema

    if isinstance(value, list):
        schema["type"] = "array"
        sample_item = samples.get(f"{path_key}.[]") if path_key else None
        schema["items"] = _infer_array_items(value, sample_item)
        schema["default"] = value
        return _apply_constraints(schema, path_key)

    if value is None and isinstance(samples.get(path_key), dict):
        sample_dict = samples[path_key]
        schema["type"] = ["null", "object"]
        schema["properties"] = {}
        schema["additionalProperties"] = False
        for key, child in sample_dict.items():
            schema["properties"][key] = build_schema(
                child,
                path=path + [key],
                comment_map=comment_map,
                samples=samples,
            )
        if path_key in _FREEFORM_MAPS:
            schema["additionalProperties"] = _FREEFORM_MAPS[path_key]
        schema["default"] = None
        return schema

    schema["type"] = _infer_schema_type(path_key, value, samples.get(path_key))
    schema["default"] = value
    return _apply_constraints(schema, path_key)


def _apply_constraints(schema: dict[str, Any], path_key: str) -> dict[str, Any]:
    constraints = _constraints_for(path_key)
    if "enum" in constraints:
        schema["enum"] = constraints["enum"]
    if "minimum" in constraints:
        schema["minimum"] = constraints["minimum"]
    if "maximum" in constraints:
        schema["maximum"] = constraints["maximum"]
    if "minItems" in constraints:
        schema["minItems"] = constraints["minItems"]
    if "maxItems" in constraints:
        schema["maxItems"] = constraints["maxItems"]
    return schema


def generate_schema(
    *,
    defaults_path: Path = _DEFAULTS_FILE,
    config_dir: Path = _CONFIG_DIR,
    config_map_path: Path = _CONFIG_MAP,
) -> dict[str, Any]:
    """Generate a JSON schema dictionary for config files."""

    defaults = load_yaml(defaults_path)
    comment_map = extract_inline_comments(defaults_path)

    configs = [load_yaml(path) for path in collect_config_sources(config_dir)]
    merged = defaults
    for cfg in configs:
        merged = merge_defaults(merged, cfg)

    samples = gather_samples(configs)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Trend Model configuration",
        "type": "object",
        "description": _schema_root_description(config_map_path),
        "additionalProperties": False,
        "properties": {},
        "default": merged,
    }
    for key, value in merged.items():
        schema["properties"][key] = build_schema(
            value,
            path=[key],
            comment_map=comment_map,
            samples=samples,
        )
    return schema


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Create a compact schema for prompt injection."""

    allowed_keys = {"type", "description", "default", "nl_editable", "properties", "items"}
    compact: dict[str, Any] = {k: v for k, v in schema.items() if k in allowed_keys}
    if "properties" in schema:
        compact["properties"] = {key: _compact_schema(val) for key, val in schema["properties"].items()}
    if "items" in schema:
        compact["items"] = schema["items"]
    return compact


def write_schema_files(
    *,
    schema_path: Path = _SCHEMA_FILE,
    compact_path: Path = _COMPACT_SCHEMA_FILE,
) -> tuple[Path, Path]:
    """Generate and write schema files to disk."""

    schema = generate_schema()
    compact = _compact_schema(schema)

    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return schema_path, compact_path


__all__ = [
    "collect_config_sources",
    "generate_schema",
    "write_schema_files",
]
