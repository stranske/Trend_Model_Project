from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from utils.paths import proj_path, repo_root

__all__ = [
    "MonteCarloScenario",
    "ScenarioRegistryEntry",
    "get_scenario_path",
    "list_scenarios",
    "load_scenario",
]

_REGISTRY_PATH = proj_path("config", "scenarios", "monte_carlo", "index.yml")
_SUPPORTED_SUFFIXES = (".yml", ".yaml")


@dataclass(frozen=True)
class ScenarioRegistryEntry:
    name: str
    path: Path
    description: str | None
    tags: tuple[str, ...]


def _ensure_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{label} must be a mapping")


def _resolve_registry_path(registry_path: Path | None) -> Path:
    resolved = registry_path or _REGISTRY_PATH
    return resolved.resolve()


def _registry_label(registry_path: Path | None) -> str:
    resolved = _resolve_registry_path(registry_path)
    try:
        return str(resolved.relative_to(repo_root()))
    except ValueError:
        return resolved.name or str(resolved)


def _load_yaml(path: Path) -> Mapping[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Scenario config '{path}' is empty")
    return _ensure_mapping(raw, label=f"Scenario config '{path}'")


def _coerce_tags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        return ()
    cleaned: list[str] = []
    for tag in values:
        label = str(tag).strip()
        if label:
            cleaned.append(label)
    return tuple(cleaned)


def _load_registry(registry_path: Path | None = None) -> list[ScenarioRegistryEntry]:
    path = _resolve_registry_path(registry_path)
    if not path.exists():
        registry_label = _registry_label(registry_path)
        raise FileNotFoundError(f"Scenario registry '{registry_label}' does not exist")
    raw = _load_yaml(path)
    entries = raw.get("scenarios")
    if not isinstance(entries, list):
        raise ValueError("Scenario registry must define 'scenarios' as a list")

    scenarios: list[ScenarioRegistryEntry] = []
    seen_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Scenario registry entries must be mappings")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("Scenario registry entry missing 'name'")
        if name in seen_names:
            raise ValueError(f"Scenario registry entry '{name}' is duplicated")
        path_value = entry.get("path")
        if not path_value:
            raise ValueError(f"Scenario registry entry '{name}' missing 'path'")
        resolved_path = _resolve_path(
            str(path_value),
            base_dir=path.parent,
            search_dirs=[proj_path()],
        )
        scenarios.append(
            ScenarioRegistryEntry(
                name=name,
                path=resolved_path,
                description=str(entry.get("description") or "") or None,
                tags=_coerce_tags(entry.get("tags")),
            )
        )
        seen_names.add(name)

    return scenarios


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_path(value: str, *, base_dir: Path, search_dirs: Sequence[Path] | None = None) -> Path:
    raw = Path(value).expanduser()
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            (base_dir / raw).resolve(),
            (base_dir.parent / raw).resolve(),
        ]
        if search_dirs:
            candidates.extend((Path(root) / raw).resolve() for root in search_dirs)
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                raise IsADirectoryError(f"Path '{candidate}' must be a file")
            if candidate.suffix not in _SUPPORTED_SUFFIXES:
                allowed = ", ".join(_SUPPORTED_SUFFIXES)
                raise ValueError(f"Scenario config '{candidate}' must use one of: {allowed}")
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{value}'. Checked: {', '.join(str(c) for c in candidates)}"
    )


def _resolve_base_config(value: str, *, source_path: Path) -> Path:
    raw = Path(value).expanduser()
    allowed_roots = [
        source_path.parent.resolve(),
        proj_path().resolve(),
    ]

    candidates: list[Path] = []
    if raw.is_absolute():
        candidate = raw.resolve()
        if not any(_is_within(candidate, root) for root in allowed_roots):
            allowed = ", ".join(str(root) for root in allowed_roots)
            raise ValueError(f"base_config must resolve under: {allowed}")
        candidates = [candidate]
    else:
        candidates = [(root / raw).resolve() for root in allowed_roots]

    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                raise IsADirectoryError(f"Path '{candidate}' must be a file")
            return candidate

    raise FileNotFoundError(
        f"Could not locate base_config '{value}'. Checked: {', '.join(str(c) for c in candidates)}"
    )


def _matches_any_tag(entry_tags: Iterable[str], tag_set: set[str]) -> bool:
    return bool({tag.lower() for tag in entry_tags}.intersection(tag_set))


def list_scenarios(
    *, tags: Iterable[str] | None = None, registry_path: Path | None = None
) -> list[ScenarioRegistryEntry]:
    """Return registered Monte Carlo scenarios.

    Parameters
    ----------
    tags:
        Optional tag filter. When provided, only scenarios that share at least
        one tag (logical OR) are returned.
    registry_path:
        Optional override for the registry location (useful in tests).
    """

    scenarios = _load_registry(registry_path)
    if not tags:
        return scenarios
    tag_set = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    if not tag_set:
        return scenarios
    filtered: list[ScenarioRegistryEntry] = []
    for entry in scenarios:
        if _matches_any_tag(entry.tags, tag_set):
            filtered.append(entry)
    return filtered


def _format_missing(
    name: str,
    scenarios: Sequence[ScenarioRegistryEntry],
    *,
    registry_label: str,
) -> str:
    available = ", ".join(sorted(entry.name for entry in scenarios))
    if available:
        return f"Unknown scenario '{name}' in registry '{registry_label}'. Available: {available}"
    return f"Unknown scenario '{name}' in registry '{registry_label}'. No scenarios registered."


def get_scenario_path(name: str, *, registry_path: Path | None = None) -> Path:
    """Return the resolved path for a registered scenario name."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("Scenario name is required")
    scenarios = _load_registry(registry_path)
    registry_label = _registry_label(registry_path)
    for entry in scenarios:
        if entry.name == normalized:
            return entry.path
    raise ValueError(_format_missing(normalized, scenarios, registry_label=registry_label))


def _extract_scenario_metadata(
    name: str, raw: Mapping[str, object], *, source_path: Path
) -> tuple[str, str | None, str | None]:
    scenario_block = raw.get("scenario")
    scenario_map: Mapping[str, object] | None = None
    if "scenario" in raw and scenario_block is None:
        raise ValueError("Scenario config 'scenario' must be a mapping (null provided)")
    if scenario_block is not None:
        scenario_map = _ensure_mapping(scenario_block, label="Scenario config 'scenario'")

    top_level = {
        "name": raw.get("name"),
        "description": raw.get("description"),
        "version": raw.get("version"),
    }

    if scenario_map is None:
        merged = dict(top_level)
    else:
        merged = dict(scenario_map)
        for key, value in top_level.items():
            if value is None:
                continue
            if key in merged and merged.get(key) not in (None, ""):
                if str(merged[key]).strip() != str(value).strip():
                    raise ValueError(
                        f"Scenario config has conflicting '{key}' values between scenario block "
                        f"and top-level in '{source_path}'"
                    )
                continue
            merged[key] = value

    scenario_name = str(merged.get("name") or "").strip()
    if not scenario_name:
        raise ValueError("Scenario config must define scenario.name")
    if scenario_name != name:
        raise ValueError(f"Scenario name mismatch: registry '{name}' vs config '{scenario_name}'")

    description_value = merged.get("description")
    description = str(description_value) if description_value is not None else None

    version_value = merged.get("version")
    version = None
    if version_value is not None:
        version = str(version_value).strip()
        if not version:
            raise ValueError("Scenario config must define scenario.version as a non-empty string")

    return scenario_name, description, version


def _parse_scenario(
    name: str, raw: Mapping[str, object], *, source_path: Path
) -> MonteCarloScenario:
    scenario_name, description, version = _extract_scenario_metadata(
        name, raw, source_path=source_path
    )

    base_config_value = raw.get("base_config")
    if not base_config_value:
        raise ValueError("Scenario config must define base_config")
    base_config = _resolve_base_config(str(base_config_value), source_path=source_path)

    monte_carlo = raw.get("monte_carlo")
    if monte_carlo is None:
        raise ValueError("Scenario config must define monte_carlo")
    monte_carlo_map = _ensure_mapping(monte_carlo, label="Scenario config 'monte_carlo'")

    strategy_set = None
    if "strategy_set" in raw:
        strategy_set = _ensure_mapping(
            raw.get("strategy_set"), label="Scenario config 'strategy_set'"
        )

    outputs = None
    if "outputs" in raw:
        outputs = _ensure_mapping(raw.get("outputs"), label="Scenario config 'outputs'")

    scenario_kwargs: dict[str, Any] = {
        "name": scenario_name,
        "description": description,
        "version": version,
        "base_config": base_config,
        "monte_carlo": monte_carlo_map,
        "strategy_set": strategy_set,
        "outputs": outputs,
        "path": source_path,
        "raw": raw,
    }

    if "folds" in raw:
        folds_value = raw.get("folds")
        if folds_value is None:
            raise ValueError("Scenario config 'folds' must be a mapping (null provided)")
        scenario_kwargs["folds"] = _ensure_mapping(folds_value, label="Scenario config 'folds'")

    if "return_model" in raw:
        return_model_value = raw.get("return_model")
        if return_model_value is None:
            raise ValueError("Scenario config 'return_model' must be a mapping (null provided)")
        scenario_kwargs["return_model"] = _ensure_mapping(
            return_model_value, label="Scenario config 'return_model'"
        )

    return MonteCarloScenario(**scenario_kwargs)


def load_scenario(name: str, *, registry_path: Path | None = None) -> MonteCarloScenario:
    """Load and validate a scenario definition by name."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("Scenario name is required")
    scenarios = _load_registry(registry_path)
    registry_label = _registry_label(registry_path)
    entry = next((item for item in scenarios if item.name == normalized), None)
    if entry is None:
        raise ValueError(_format_missing(normalized, scenarios, registry_label=registry_label))

    raw = _load_yaml(entry.path)
    return _parse_scenario(normalized, raw, source_path=entry.path)
