from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParamStore:
    """Mutable GUI state shared across view layers."""

    cfg: dict[str, Any] = field(default_factory=dict)
    theme: str = "system"
    dirty: bool = False
    weight_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.cfg

    @classmethod
    def from_yaml(cls, path: Path) -> ParamStore:
        return cls(cfg=yaml.safe_load(path.read_text()))


__all__ = ["ParamStore"]
