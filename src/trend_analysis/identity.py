"""Deterministic entity identity resolution for run manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class EntityId:
    canonical_id: str
    display_name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    resolved: bool = True

    def to_manifest(self, *, label: str, labels: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": label,
            "canonical_id": self.canonical_id,
            "display_name": self.display_name,
            "resolved": self.resolved,
        }
        if labels is not None:
            payload["labels"] = labels
        if self.aliases:
            payload["aliases"] = list(self.aliases)
        return payload


def normalize_identity_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().casefold())


class IdentityMap:
    def __init__(self, entities: list[EntityId] | None = None) -> None:
        self._entities = list(entities or [])
        self._exact_lookup: dict[str, EntityId] = {}
        self._lookup: dict[str, EntityId] = {}
        for entity in self._entities:
            for alias in (entity.display_name, entity.canonical_id, *entity.aliases):
                raw_alias = str(alias)
                if raw_alias:
                    self._exact_lookup.setdefault(raw_alias, entity)
                normalized = normalize_identity_label(alias)
                if normalized:
                    self._lookup.setdefault(normalized, entity)

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any] | None, *, base_path: Path | None = None
    ) -> "IdentityMap":
        if not isinstance(cfg, Mapping):
            return cls()
        identity_cfg = cfg.get("identity")
        entities: list[EntityId] = []
        if isinstance(identity_cfg, Mapping):
            entities.extend(_entities_from_identity_block(identity_cfg))
            universe_paths = identity_cfg.get("universes") or identity_cfg.get("universe")
            entities.extend(_entities_from_universe_paths(universe_paths, base_path=base_path))
        if not entities:
            entities.extend(_entities_from_universe_paths(cfg.get("universe"), base_path=base_path))
        return cls(entities)

    def resolve(self, label: str) -> EntityId:
        raw = str(label)
        entity = self._exact_lookup.get(raw)
        if entity is not None:
            return entity
        normalized = normalize_identity_label(label)
        entity = self._lookup.get(normalized)
        if entity is not None:
            return entity
        return EntityId(
            canonical_id=f"unknown:{raw}",
            display_name=raw,
            aliases=(),
            resolved=False,
        )


def _entities_from_identity_block(identity_cfg: Mapping[str, Any]) -> list[EntityId]:
    entries = identity_cfg.get("entities") or identity_cfg.get("entries") or []
    if isinstance(entries, Mapping):
        entries = [
            {"canonical_id": key, **value} if isinstance(value, Mapping) else {"canonical_id": key}
            for key, value in entries.items()
        ]
    if not isinstance(entries, list):
        return []
    entities: list[EntityId] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        canonical_id = str(entry.get("canonical_id") or entry.get("id") or "").strip()
        if not canonical_id:
            continue
        display_name = str(entry.get("display_name") or entry.get("name") or canonical_id).strip()
        aliases_raw = entry.get("aliases") or []
        aliases = tuple(str(alias).strip() for alias in aliases_raw if str(alias).strip()) if isinstance(aliases_raw, list) else ()
        entities.append(
            EntityId(
                canonical_id=canonical_id,
                display_name=display_name or canonical_id,
                aliases=aliases,
                resolved=True,
            )
        )
    return entities


def _entities_from_universe_paths(
    universe_paths: Any, *, base_path: Path | None
) -> list[EntityId]:
    if universe_paths is None:
        return []
    paths = universe_paths if isinstance(universe_paths, list) else [universe_paths]
    entities: list[EntityId] = []
    for raw_path in paths:
        if not isinstance(raw_path, (str, Path)):
            continue
        path = Path(raw_path)
        if not path.is_absolute() and base_path is not None:
            path = base_path / path
        if not path.exists() or path.suffix.lower() not in {".yml", ".yaml"}:
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        members = payload.get("members") or []
        if not isinstance(members, list):
            continue
        for member in members:
            if not isinstance(member, str) or not member.strip():
                continue
            canonical_id = "fund:" + re.sub(r"[^a-z0-9]+", "-", member.casefold()).strip("-")
            entities.append(EntityId(canonical_id=canonical_id, display_name=member))
    return entities


__all__ = ["EntityId", "IdentityMap", "normalize_identity_label"]
