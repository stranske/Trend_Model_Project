from __future__ import annotations

import re
from typing import Any

__all__ = ["normalize_regime_key", "alias_regime_key"]


def normalize_regime_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def alias_regime_key(value: str) -> str | None:
    aliases = {
        "riskon": "calm",
        "riskoff": "stress",
        "calm": "riskon",
        "stress": "riskoff",
    }
    return aliases.get(value)
