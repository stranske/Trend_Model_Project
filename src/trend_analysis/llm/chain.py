"""LangChain wrapper for ConfigPatch generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

PromptBuilder = Callable[..., str]


@dataclass(slots=True)
class ConfigPatchChain:
    """Container for the ConfigPatch LangChain pipeline."""

    llm: Any
    prompt_builder: PromptBuilder
    schema: dict[str, Any]
    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 1
