from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from utils.paths import proj_path

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


@dataclass(frozen=True)
class MonteCarloScenario:
    name: str
    description: str | None
    version: str
    base_config: Path
    monte_carlo: Mapping[str, object]
    strategy_set: Mapping[str, object] | None
    outputs: Mapping[str, object] | None
    path: Path
    raw: Mapping[str, object]


def _ensure_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{label} must be a mapping")


def _resolve_registry_path(registry_path: Path | None) -> Path:
    resolved = registry_path or _REGISTRY_PATH
    return resolved.resolve()


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
        raise FileNotFoundError(f"Scenario registry '{path}' does not exist")
    raw = _load_yaml(path)
    entries = raw.get("scenarios")
    if not isinstance(entries, Sequence):
        raise ValueError("Scenario registry must define a list under 'scenarios'")

    scenarios: list[ScenarioRegistryEntry] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Scenario registry entries must be mappings")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("Scenario registry entry missing 'name'")
        path_value = entry.get("path")
        if not path_value:
            raise ValueError(f"Scenario registry entry '{name}' missing 'path'")
        resolved_path = _resolve_path(str(path_value), base_dir=path.parent)
        scenarios.append(
            ScenarioRegistryEntry(
                name=name,
                path=resolved_path,
                description=str(entry.get("description") or "") or None,
                tags=_coerce_tags(entry.get("tags")),
            )
        )

    return scenarios


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    raw = Path(value).expanduser()
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            (base_dir / raw).resolve(),
            (base_dir.parent / raw).resolve(),
            Path.cwd().resolve() / raw,
        ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                raise IsADirectoryError(f"Path '{candidate}' must be a file")
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{value}'. Checked: {', '.join(str(c) for c in candidates)}"
    )


def list_scenarios(
    *, tags: Iterable[str] | None = None, registry_path: Path | None = None
) -> list[ScenarioRegistryEntry]:
    """Return registered Monte Carlo scenarios.

    Parameters
    ----------
    tags:
        Optional tag filter. When provided, only scenarios that share at least
        one tag are returned.
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
        entry_tags = {tag.lower() for tag in entry.tags}
        if entry_tags.intersection(tag_set):
            filtered.append(entry)
    return filtered


def _format_missing(name: str, scenarios: Sequence[ScenarioRegistryEntry]) -> str:
    available = ", ".join(sorted(entry.name for entry in scenarios))
    if available:
        return f"Unknown scenario '{name}'. Available: {available}"
    return f"Unknown scenario '{name}'. No scenarios registered."


def get_scenario_path(name: str, *, registry_path: Path | None = None) -> Path:
    """Return the resolved path for a registered scenario name."""

    scenarios = _load_registry(registry_path)
    for entry in scenarios:
        if entry.name == name:
            return entry.path
    raise ValueError(_format_missing(name, scenarios))


def _parse_scenario(
    name: str, raw: Mapping[str, object], *, source_path: Path
) -> MonteCarloScenario:
    scenario = raw.get("scenario")
    scenario_map = _ensure_mapping(scenario, label="Scenario config 'scenario'")

    scenario_name = str(scenario_map.get("name") or "").strip()
    if not scenario_name:
        raise ValueError("Scenario config must define scenario.name")
    if scenario_name != name:
        raise ValueError(f"Scenario name mismatch: registry '{name}' vs config '{scenario_name}'")

    version = scenario_map.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Scenario config must define scenario.version as a string")

    description_value = scenario_map.get("description")
    description = str(description_value) if description_value is not None else None

    base_config_value = raw.get("base_config")
    if not base_config_value:
        raise ValueError("Scenario config must define base_config")
    base_config = _resolve_path(str(base_config_value), base_dir=source_path.parent)

    monte_carlo = raw.get("monte_carlo")
    if monte_carlo is None:
        raise ValueError("Scenario config must define monte_carlo")
    monte_carlo_map = _ensure_mapping(monte_carlo, label="Scenario config 'monte_carlo'")

    strategy_set_value = raw.get("strategy_set")
    strategy_set = None
    if strategy_set_value is not None:
        strategy_set = _ensure_mapping(strategy_set_value, label="Scenario config 'strategy_set'")

    outputs_value = raw.get("outputs")
    outputs = None
    if outputs_value is not None:
        outputs = _ensure_mapping(outputs_value, label="Scenario config 'outputs'")

    return MonteCarloScenario(
        name=scenario_name,
        description=description,
        version=version,
        base_config=base_config,
        monte_carlo=monte_carlo_map,
        strategy_set=strategy_set,
        outputs=outputs,
        path=source_path,
        raw=raw,
    )


def load_scenario(name: str, *, registry_path: Path | None = None) -> MonteCarloScenario:
    """Load and validate a scenario definition by name."""

    scenarios = _load_registry(registry_path)
    entry = next((item for item in scenarios if item.name == name), None)
    if entry is None:
        raise ValueError(_format_missing(name, scenarios))

    raw = _load_yaml(entry.path)
    return _parse_scenario(name, raw, source_path=entry.path)
