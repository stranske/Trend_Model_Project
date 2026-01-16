"""LLM utilities for Trend Model."""

from .chain import ConfigPatchChain
from .nl_logging import NLOperationLog
from .prompts import (
    DEFAULT_RESULT_RULES,
    DEFAULT_RESULT_SYSTEM_PROMPT,
    DEFAULT_SAFETY_RULES,
    DEFAULT_SYSTEM_PROMPT,
    SECTION_CONFIG,
    SECTION_RESULT_METRICS,
    SECTION_RESULT_OUTPUT,
    SECTION_RESULT_QUESTIONS,
    SECTION_RESULT_RULES,
    SECTION_RESULT_SYSTEM,
    SECTION_SAFETY,
    SECTION_SCHEMA,
    SECTION_SYSTEM,
    SECTION_USER,
    build_config_patch_prompt,
    build_result_summary_prompt,
    format_config_for_prompt,
)
from .providers import LLMProviderConfig, create_llm
from .replay import ReplayResult, load_nl_log_entry, render_prompt, replay_nl_entry
from .result_metrics import MetricEntry, extract_metric_catalog, format_metric_catalog

__all__ = [
    "ConfigPatchChain",
    "DEFAULT_RESULT_RULES",
    "DEFAULT_RESULT_SYSTEM_PROMPT",
    "DEFAULT_SAFETY_RULES",
    "DEFAULT_SYSTEM_PROMPT",
    "SECTION_CONFIG",
    "SECTION_RESULT_METRICS",
    "SECTION_RESULT_OUTPUT",
    "SECTION_RESULT_QUESTIONS",
    "SECTION_RESULT_RULES",
    "SECTION_RESULT_SYSTEM",
    "SECTION_SAFETY",
    "SECTION_SCHEMA",
    "SECTION_SYSTEM",
    "SECTION_USER",
    "build_config_patch_prompt",
    "build_result_summary_prompt",
    "format_config_for_prompt",
    "NLOperationLog",
    "ReplayResult",
    "LLMProviderConfig",
    "create_llm",
    "MetricEntry",
    "extract_metric_catalog",
    "format_metric_catalog",
    "load_nl_log_entry",
    "replay_nl_entry",
    "render_prompt",
]
