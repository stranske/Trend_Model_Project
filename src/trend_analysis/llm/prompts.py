"""Prompt templates for NL config patch generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import yaml

SECTION_SYSTEM = "SYSTEM PROMPT"
SECTION_CONFIG = "CURRENT CONFIG"
SECTION_SCHEMA = "ALLOWED SCHEMA"
SECTION_SAFETY = "SAFETY RULES"
SECTION_USER = "USER INSTRUCTION"
SECTION_RETRY_ERROR = "PREVIOUS ERROR"
SECTION_RESULT_SYSTEM = "RESULT SUMMARY SYSTEM PROMPT"
SECTION_RESULT_OUTPUT = "ANALYSIS OUTPUT"
SECTION_RESULT_METRICS = "METRIC CATALOG"
SECTION_RESULT_QUESTIONS = "RESULT QUESTIONS"
SECTION_RESULT_RULES = "RESULT SAFETY RULES"
SECTION_COMPARISON_SYSTEM = "COMPARISON SYSTEM PROMPT"
SECTION_COMPARISON_RESULT_A = "SIMULATION A OUTPUT"
SECTION_COMPARISON_RESULT_B = "SIMULATION B OUTPUT"
SECTION_COMPARISON_METRICS_A = "SIMULATION A METRICS"
SECTION_COMPARISON_METRICS_B = "SIMULATION B METRICS"
SECTION_COMPARISON_RULES = "COMPARISON SAFETY RULES"
SECTION_COMPARISON_QUESTIONS = "COMPARISON QUESTIONS"

DEFAULT_SYSTEM_PROMPT = """You are a configuration assistant for Trend Model.
Your task is to read the user instruction and current configuration, then emit
a ConfigPatch JSON object that updates the config safely and minimally.

Return ONLY a valid JSON object that conforms exactly to the ConfigPatch schema.
Include only operations that are necessary to implement the instruction.
Never add keys outside the ConfigPatch schema or output non-JSON content.
Do not invent keys; if the instruction or config mentions unknown or extraneous
keys, flag them explicitly in the summary and return empty operations.
If asked to target unknown keys or unsafe content, return empty operations and
explain the refusal in the summary without echoing the unsafe request.
"""

DEFAULT_RESULT_SYSTEM_PROMPT = """You are a quantitative investment analyst reviewing a trend-following manager selection backtest.
The purpose of this tool is to simulate typical allocator decision-making when using systematic selection heuristics.

ANALYSIS FRAMEWORK - Focus on manager selection dynamics:

1. **Selection Rationale**
   - Why were certain managers selected over others?
   - What metrics drove their inclusion (Sharpe, drawdown, combination)?
   - Were selections driven by genuine outperformance or favorable timing?

2. **Portfolio Persistence Analysis**
   - Which managers stayed in the portfolio across multiple periods? Why?
   - Identify any "lukewarm" performers that survived simply by not underperforming badly
   - Flag managers that entered during benign periods and may not have been stress-tested

3. **Turnover and Replacement Decisions**
   - What triggered manager exits - underperformance or mechanical rules?
   - Were replacements upgrades or lateral moves?
   - Could different selection parameters have avoided unnecessary churn?

4. **Out-of-Sample Reality Check**
   - Do IS selection metrics predict OS performance?
   - Which managers showed consistent behavior vs. regime-dependent results?
   - Flag any concerning divergence that suggests overfitting to history

5. **Portfolio Construction Insights**
   - Is the weighting scheme producing expected concentration/diversification?
   - How does the selected portfolio compare to equal-weight baseline?
   - Are position sizes aligned with conviction from the metrics?

6. **Actionable Observations**
   - What parameter adjustments might improve selection quality?
   - Should lookback periods, selection counts, or thresholds change?
   - Note any managers that warrant manual review outside systematic rules

STYLE GUIDELINES:
- Lead with decision-making insights, not metric recitation
- Use comparative language: "Manager X was selected over Y because..."
- Be direct about weak selections that may have slipped through
- Focus on what an allocator would want to know for the next decision

Include this disclaimer verbatim at the end:
"This is analytical output, not financial advice. Always verify metrics independently."
"""

DEFAULT_COMPARISON_SYSTEM_PROMPT = """You are a quantitative investment analyst comparing two trend-following manager selection backtests.
Your goal is to explain *why* the outcomes differ, grounding your reasoning in the parameter differences
and the observed metrics for each simulation.

ANALYSIS FRAMEWORK - Focus on differences and drivers:

1. **Parameter Drivers**
    - Which configuration changes most plausibly drive the outcome differences?
    - Highlight the mechanisms: selection thresholds, ranking, constraints, or weighting.

2. **Selection & Turnover Impact**
    - How did the changes alter manager selection, turnover, and replacements?
    - Are differences driven by entry/exit timing or persistent holdings?

3. **Risk/Return Trade-offs**
    - Compare how risk metrics (drawdown, volatility, turnover, transaction costs) shifted.
    - Explain whether the changes improved risk-adjusted performance or just raw returns.

4. **In-Sample vs Out-of-Sample Effects**
    - Identify where in-sample gains failed to translate to out-of-sample results (or vice versa).
    - Flag possible overfitting or regime sensitivity.

5. **Next Tests**
    - Recommend follow-up parameter tweaks or diagnostics to validate the observed differences.

STYLE GUIDELINES:
- Prioritize causal reasoning tied to configuration differences.
- Do not restate full metric tables; cite only key values that support your conclusions.
- Use clear A/B framing (Simulation A vs Simulation B).

Include this disclaimer verbatim at the end:
This is analytical output, not financial advice. Always verify metrics independently.
"""

DEFAULT_SAFETY_RULES = (
    "Use only keys that exist in the allowed schema or current config.",
    "Do not invent new keys or aliases; unknown keys must be explicitly flagged in the summary.",
    "Reject and flag extraneous keys in the instruction or config; do not pass them through.",
    "Do not include any keys beyond operations, risk_flags, and summary.",
    "If the instruction asks for unknown keys or unsafe content, return no operations and explain why in summary.",
    "Never output markdown, prose, or any content that is not a single JSON object.",
    "Flag risky changes in risk_flags when appropriate (constraints, leverage, validations).",
    "Keep patch operations minimal and ordered in the sequence they should apply.",
    "Never include secrets, credentials, or unsafe content in any field.",
)

DEFAULT_RESULT_RULES = (
    "Ground all claims in metrics from the analysis output - cite sparingly for key points.",
    "Focus on analytical insights and comparisons, not restating individual numbers.",
    "Flag potential overfitting when in-sample metrics significantly exceed out-of-sample.",
    "Compare against equal-weight baseline when available to assess selection value.",
    "If critical data is missing, note what additional analysis would be needed.",
    "Prioritize actionable observations over comprehensive metric listing.",
)

DEFAULT_COMPARISON_RULES = (
    "Ground all claims in the provided metrics and configuration differences.",
    "Cite only key values that explain deltas; avoid listing every number.",
    "Focus on causal explanations tied to parameter changes.",
    "Call out uncertainty when differences cannot be attributed confidently.",
    "If critical data is missing, note what additional analysis would be needed.",
)


def format_config_for_prompt(config: Any) -> str:
    """Render a config mapping or excerpt as YAML for prompt injection."""

    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False).strip()


def build_config_patch_prompt(
    *,
    current_config: str,
    allowed_schema: str,
    instruction: str,
    system_prompt: str | None = None,
    safety_rules: Iterable[str] | None = None,
) -> str:
    """Build the prompt text for ConfigPatch generation."""

    system_text = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    rules = list(safety_rules or DEFAULT_SAFETY_RULES)
    safety_text = "\n".join(f"- {rule}" for rule in rules)
    sections = [
        _format_section(SECTION_SYSTEM, system_text),
        _format_section(SECTION_CONFIG, current_config.strip()),
        _format_section(SECTION_SCHEMA, allowed_schema.strip()),
        _format_section(SECTION_SAFETY, safety_text),
        _format_section(SECTION_USER, instruction.strip()),
    ]
    return "\n\n".join(sections).strip()


def build_retry_prompt(
    *,
    current_config: str,
    allowed_schema: str,
    instruction: str,
    error_message: str,
    system_prompt: str | None = None,
    safety_rules: Iterable[str] | None = None,
) -> str:
    """Build the retry prompt with previous parsing error context."""

    base_prompt = build_config_patch_prompt(
        current_config=current_config,
        allowed_schema=allowed_schema,
        instruction=instruction,
        system_prompt=system_prompt,
        safety_rules=safety_rules,
    )
    retry_note = (
        f"{error_message}\n\n"
        "Return ONLY a valid JSON object that matches the ConfigPatch schema."
    )
    sections = [
        base_prompt,
        _format_section(SECTION_RETRY_ERROR, retry_note),
    ]
    return "\n\n".join(sections).strip()


def build_result_summary_prompt(
    *,
    analysis_output: str,
    metric_catalog: str,
    questions: str,
    system_prompt: str | None = None,
    safety_rules: Iterable[str] | None = None,
) -> str:
    """Build the prompt text for analysis result summaries and Q&A."""

    system_text = (system_prompt or DEFAULT_RESULT_SYSTEM_PROMPT).strip()
    rules = list(safety_rules or DEFAULT_RESULT_RULES)
    safety_text = "\n".join(f"- {rule}" for rule in rules)
    sections = [
        _format_section(SECTION_RESULT_SYSTEM, system_text),
        _format_section(SECTION_RESULT_OUTPUT, analysis_output.strip()),
        _format_section(SECTION_RESULT_METRICS, metric_catalog.strip()),
        _format_section(SECTION_RESULT_RULES, safety_text),
        _format_section(SECTION_RESULT_QUESTIONS, questions.strip()),
    ]
    return "\n\n".join(sections).strip()


def build_comparison_prompt(
    *,
    analysis_output: str | tuple[str, str] | list[str],
    metric_catalog: str | tuple[str, str] | list[str],
    questions: str,
    system_prompt: str | None = None,
    safety_rules: Iterable[str] | None = None,
) -> str:
    """Build the prompt text for comparing two simulation outputs."""

    system_text = (system_prompt or DEFAULT_COMPARISON_SYSTEM_PROMPT).strip()
    rules = list(safety_rules or DEFAULT_COMPARISON_RULES)
    safety_text = "\n".join(f"- {rule}" for rule in rules)
    output_a, output_b = _coerce_pair(analysis_output)
    metrics_a, metrics_b = _coerce_pair(metric_catalog)
    sections = [
        _format_section(SECTION_COMPARISON_SYSTEM, system_text),
        _format_section(SECTION_COMPARISON_RESULT_A, output_a.strip()),
        _format_section(SECTION_COMPARISON_RESULT_B, output_b.strip()),
        _format_section(SECTION_COMPARISON_METRICS_A, metrics_a.strip()),
        _format_section(SECTION_COMPARISON_METRICS_B, metrics_b.strip()),
        _format_section(SECTION_COMPARISON_RULES, safety_text),
        _format_section(SECTION_COMPARISON_QUESTIONS, questions.strip()),
    ]
    return "\n\n".join(sections).strip()


def _coerce_pair(value: str | tuple[str, str] | list[str]) -> tuple[str, str]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0]), str(value[1])
    text = str(value)
    return text, ""


def _format_section(title: str, body: str) -> str:
    return f"## {title}\n{body}".strip()


__all__ = [
    "SECTION_SYSTEM",
    "SECTION_CONFIG",
    "SECTION_SCHEMA",
    "SECTION_SAFETY",
    "SECTION_USER",
    "SECTION_RETRY_ERROR",
    "SECTION_RESULT_SYSTEM",
    "SECTION_RESULT_OUTPUT",
    "SECTION_RESULT_METRICS",
    "SECTION_RESULT_QUESTIONS",
    "SECTION_RESULT_RULES",
    "SECTION_COMPARISON_SYSTEM",
    "SECTION_COMPARISON_RESULT_A",
    "SECTION_COMPARISON_RESULT_B",
    "SECTION_COMPARISON_METRICS_A",
    "SECTION_COMPARISON_METRICS_B",
    "SECTION_COMPARISON_RULES",
    "SECTION_COMPARISON_QUESTIONS",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_RESULT_SYSTEM_PROMPT",
    "DEFAULT_COMPARISON_SYSTEM_PROMPT",
    "DEFAULT_SAFETY_RULES",
    "DEFAULT_RESULT_RULES",
    "DEFAULT_COMPARISON_RULES",
    "format_config_for_prompt",
    "build_config_patch_prompt",
    "build_retry_prompt",
    "build_result_summary_prompt",
    "build_comparison_prompt",
]
