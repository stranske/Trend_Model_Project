"""LLM utilities for Trend Model."""

from .chain import ConfigPatchChain
from .nl_logging import NLOperationLog
from .prompts import (
    DEFAULT_SAFETY_RULES,
    DEFAULT_SYSTEM_PROMPT,
    SECTION_CONFIG,
    SECTION_SAFETY,
    SECTION_SCHEMA,
    SECTION_SYSTEM,
    SECTION_USER,
    build_config_patch_prompt,
    format_config_for_prompt,
)
from .providers import LLMProviderConfig, create_llm
from .replay import ReplayResult, load_nl_log_entry, render_prompt, replay_nl_entry

__all__ = [
    "ConfigPatchChain",
    "DEFAULT_SAFETY_RULES",
    "DEFAULT_SYSTEM_PROMPT",
    "SECTION_CONFIG",
    "SECTION_SAFETY",
    "SECTION_SCHEMA",
    "SECTION_SYSTEM",
    "SECTION_USER",
    "build_config_patch_prompt",
    "format_config_for_prompt",
    "NLOperationLog",
    "ReplayResult",
    "LLMProviderConfig",
    "create_llm",
    "load_nl_log_entry",
    "replay_nl_entry",
    "render_prompt",
]
