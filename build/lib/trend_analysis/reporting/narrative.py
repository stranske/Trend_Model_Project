"""Narrative section templates for report exports."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, cast
from typing import OrderedDict as OrderedDictType

import pandas as pd

from trend_analysis.reporting.portfolio_series import select_primary_portfolio_series


@dataclass(frozen=True)
class NarrativeTemplateSection:
    """Template metadata for a narrative section."""

    key: str
    title: str
    template: str
    placeholders: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeQualityIssue:
    """Describe a narrative quality issue."""

    kind: str
    message: str
    section: str | None = None
    detail: dict[str, str] | None = None


FORWARD_LOOKING_PHRASES = [
    "will",
    "forecast",
    "projected",
    "expect",
    "expected",
    "anticipate",
    "anticipated",
    "estimate",
    "estimated",
    "target",
    "targeted",
    "aim",
    "aimed",
    "plan",
    "plans",
    "planned",
    "going forward",
    "next month",
    "next quarter",
    "next year",
    "future",
]

STANDARD_NARRATIVE_DISCLAIMER = (
    "This narrative is auto-generated from observed results, excludes forward-looking "
    "statements, and should be treated as a descriptive summary rather than investment advice."
)


def default_narrative_templates() -> OrderedDictType[str, NarrativeTemplateSection]:
    """Return the default narrative templates in display order."""

    templates: OrderedDictType[str, NarrativeTemplateSection] = OrderedDictType()
    templates["executive_summary"] = NarrativeTemplateSection(
        key="executive_summary",
        title="Executive Summary",
        template=(
            "During {analysis_period}, the {portfolio_label} portfolio delivered "
            "{out_total_return}% total return with {out_cagr}% CAGR and "
            "{out_volatility}% volatility. Risk-adjusted performance measured "
            "{out_sharpe} Sharpe and {out_sortino} Sortino. Maximum drawdown over "
            "the period was {out_max_drawdown}%."
        ),
        placeholders=(
            "analysis_period",
            "portfolio_label",
            "out_total_return",
            "out_cagr",
            "out_volatility",
            "out_sharpe",
            "out_sortino",
            "out_max_drawdown",
        ),
    )
    templates["key_findings"] = NarrativeTemplateSection(
        key="key_findings",
        title="Key Findings",
        template=(
            "- Largest contributor: {top_contributor} with {top_contributor_return}% "
            "total return.\n"
            "- Portfolio breadth: {manager_count} managers with an average weight of "
            "{avg_weight}%.\n"
            "- Results consistency: {positive_months} positive months out of "
            "{observations} observed."
        ),
        placeholders=(
            "top_contributor",
            "top_contributor_return",
            "manager_count",
            "avg_weight",
            "positive_months",
            "observations",
        ),
    )
    templates["risk_highlights"] = NarrativeTemplateSection(
        key="risk_highlights",
        title="Risk Highlights",
        template=(
            "- Maximum drawdown reached {out_max_drawdown}% and volatility was "
            "{out_volatility}%.\n"
            "- Concentration: top {top_weight_count} allocations represent "
            "{top_weight_share}% of total weight.\n"
            "- Turnover averaged {turnover_avg}% with transaction cost impact of "
            "{transaction_cost_avg}%."
        ),
        placeholders=(
            "out_max_drawdown",
            "out_volatility",
            "top_weight_count",
            "top_weight_share",
            "turnover_avg",
            "transaction_cost_avg",
        ),
    )
    templates["methodology_note"] = NarrativeTemplateSection(
        key="methodology_note",
        title="Methodology Note",
        template=(
            "Metrics are computed from {return_frequency} returns between "
            "{analysis_start} and {analysis_end} using the configured rebalancing "
            f"schedule. {STANDARD_NARRATIVE_DISCLAIMER}"
        ),
        placeholders=(
            "return_frequency",
            "analysis_start",
            "analysis_end",
        ),
    )
    return templates


DEFAULT_NARRATIVE_TEMPLATES = default_narrative_templates()


__all__ = [
    "DEFAULT_NARRATIVE_TEMPLATES",
    "NarrativeTemplateSection",
    "NarrativeQualityIssue",
    "FORWARD_LOOKING_PHRASES",
    "STANDARD_NARRATIVE_DISCLAIMER",
    "narrative_generation_enabled",
    "extract_narrative_metrics",
    "generate_narrative_sections",
    "build_narrative_sections",
    "validate_narrative_quality",
    "default_narrative_templates",
]
_DISCLAIMER_SNIPPETS = (
    "auto-generated",
    "excludes forward-looking statements",
    "investment advice",
)


def narrative_generation_enabled(config: Any | None) -> bool:
    """Return True when narrative generation is enabled."""
    if config is None:
        return True
    export_cfg: Any = None
    if isinstance(config, Mapping):
        export_cfg = config.get("export")
        if export_cfg is None:
            export_cfg = config
    elif hasattr(config, "export"):
        export_cfg = getattr(config, "export")
    if not isinstance(export_cfg, Mapping):
        export_cfg = {}

    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    disable = _truthy(export_cfg.get("disable_narrative_generation"))
    disable = disable or _truthy(export_cfg.get("DISABLE_NARRATIVE_GENERATION"))
    return not disable


def _contains_forward_looking(content: str) -> str | None:
    lowered = content.lower()
    for phrase in FORWARD_LOOKING_PHRASES:
        phrase_l = phrase.lower()
        if " " in phrase_l:
            if phrase_l in lowered:
                return phrase
            continue
        pattern = r"\b" + re.escape(phrase_l) + r"\b"
        if re.search(pattern, lowered):
            return phrase
    return None


def _has_standard_disclaimer(sections: Mapping[str, str]) -> bool:
    for text in sections.values():
        content = str(text or "").lower()
        if STANDARD_NARRATIVE_DISCLAIMER.lower() in content:
            return True
    return False


def validate_narrative_quality(sections: Mapping[str, str]) -> list[NarrativeQualityIssue]:
    """Validate narrative sections for forward-looking language and disclaimers."""

    issues: list[NarrativeQualityIssue] = []
    if not sections:
        issues.append(
            NarrativeQualityIssue(
                kind="missing_sections",
                message="Narrative sections are empty.",
            )
        )
        return issues

    for key, text in sections.items():
        content = str(text or "").strip()
        if not content:
            issues.append(
                NarrativeQualityIssue(
                    kind="empty_section",
                    message="Narrative section is empty.",
                    section=str(key),
                )
            )
            continue
        phrase = _contains_forward_looking(content)
        if phrase:
            issues.append(
                NarrativeQualityIssue(
                    kind="forward_looking",
                    message="Narrative contains forward-looking language.",
                    section=str(key),
                    detail={"phrase": phrase},
                )
            )

    if not sections:
        return issues
    has_disclaimer = _has_standard_disclaimer(sections)
    methodology = sections.get("methodology_note")
    if not has_disclaimer:
        if methodology is None:
            issues.append(
                NarrativeQualityIssue(
                    kind="missing_disclaimer_section",
                    message="Methodology note section is missing.",
                )
            )
        else:
            content = str(methodology).lower()
            missing = [snippet for snippet in _DISCLAIMER_SNIPPETS if snippet not in content]
            if missing:
                issues.append(
                    NarrativeQualityIssue(
                        kind="missing_disclaimer_text",
                        message="Methodology note lacks required disclaimer text.",
                        section="methodology_note",
                        detail={"missing_snippets": ", ".join(missing)},
                    )
                )

    return issues


def _coerce_stats(stats_obj: Any) -> dict[str, float | None]:
    if stats_obj is None:
        return {}
    if isinstance(stats_obj, tuple) and len(stats_obj) >= 6:
        values = [float(val) if val is not None else float("nan") for val in stats_obj[:6]]
        return {
            "cagr": values[0],
            "vol": values[1],
            "sharpe": values[2],
            "sortino": values[3],
            "information_ratio": values[4],
            "max_drawdown": values[5],
        }
    for attr in ("cagr", "vol", "sharpe", "sortino", "information_ratio", "max_drawdown"):
        if not hasattr(stats_obj, attr):
            return {}
    return {
        "cagr": getattr(stats_obj, "cagr"),
        "vol": getattr(stats_obj, "vol"),
        "sharpe": getattr(stats_obj, "sharpe"),
        "sortino": getattr(stats_obj, "sortino"),
        "information_ratio": getattr(stats_obj, "information_ratio"),
        "max_drawdown": getattr(stats_obj, "max_drawdown"),
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}"


def _total_return(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return float((1.0 + s.astype(float)).prod() - 1.0)


def _normalise_weights(weights: Mapping[str, float]) -> pd.Series:
    series = pd.Series({str(k): float(v) for k, v in weights.items()})
    series = series.replace([math.inf, -math.inf], math.nan).dropna()
    series = series[series.abs() > 0]
    total = float(series.sum())
    if total:
        series = series / total
    return series


def _extract_frequency_label(res: Mapping[str, Any]) -> str:
    freq = res.get("input_frequency") or res.get("input_frequency_details")
    if not freq:
        freq = cast(Mapping[str, Any], res.get("preprocessing", {})).get(
            "input_frequency_details", {}
        )
    if not freq:
        metadata = res.get("metadata")
        if isinstance(metadata, Mapping):
            freq = metadata.get("frequency", {})
    if isinstance(freq, Mapping):
        label = freq.get("target_label") or freq.get("label")
        if label:
            return str(label)
    return "unspecified"


def _extract_lookbacks(res: Mapping[str, Any]) -> tuple[str | None, str | None]:
    metadata = res.get("metadata")
    if isinstance(metadata, Mapping):
        lookbacks = metadata.get("lookbacks")
        if isinstance(lookbacks, Mapping):
            out_sample = lookbacks.get("out_sample")
            if isinstance(out_sample, Mapping):
                start = out_sample.get("start")
                end = out_sample.get("end")
                return (str(start) if start else None, str(end) if end else None)
            in_sample = lookbacks.get("in_sample")
            if isinstance(in_sample, Mapping):
                start = in_sample.get("start")
                end = in_sample.get("end")
                return (str(start) if start else None, str(end) if end else None)
    return (None, None)


def extract_narrative_metrics(res: Mapping[str, Any]) -> dict[str, str]:
    """Extract narrative-ready metrics from the report payload."""
    out_df = res.get("out_sample_scaled")
    out_df = out_df if isinstance(out_df, pd.DataFrame) else None
    weights = res.get("fund_weights")
    weights_map = weights if isinstance(weights, Mapping) else None
    stats_obj = res.get("out_user_stats") or res.get("out_ew_stats")
    stats = _coerce_stats(stats_obj)
    port_series = select_primary_portfolio_series(res)

    total_return = _total_return(port_series)
    out_cagr = _to_float(stats.get("cagr"))
    out_vol = _to_float(stats.get("vol"))
    out_sharpe = _to_float(stats.get("sharpe"))
    out_sortino = _to_float(stats.get("sortino"))
    out_max_dd = _to_float(stats.get("max_drawdown"))

    start, end = _extract_lookbacks(res)
    analysis_period = f"{start} to {end}" if start and end else "the analysis period"

    top_contributor = "N/A"
    top_contributor_return = "N/A"
    if out_df is not None and not out_df.empty:
        totals = out_df.apply(_total_return)
        if weights_map:
            weights_series = _normalise_weights(weights_map)
            if not weights_series.empty:
                totals = totals.mul(weights_series.reindex(totals.index, fill_value=0.0))
        if not totals.empty:
            name = totals.idxmax()
            top_contributor = str(name)
            raw = _total_return(out_df[name]) if name in out_df else None
            top_contributor_return = _format_pct(raw)

    manager_count = None
    avg_weight = None
    top_weight_count = None
    top_weight_share = None
    if weights_map:
        weights_series = _normalise_weights(weights_map)
        if not weights_series.empty:
            manager_count = int(weights_series.shape[0])
            avg_weight = float(weights_series.mean())
            top_weight_count = min(3, int(weights_series.shape[0]))
            top_weight_share = float(
                weights_series.sort_values(ascending=False).head(top_weight_count).sum()
            )
    elif out_df is not None:
        manager_count = int(out_df.shape[1])
        if manager_count:
            avg_weight = 1.0 / float(manager_count)
            top_weight_count = min(3, manager_count)
            top_weight_share = float(top_weight_count) / float(manager_count)

    positive_months = None
    observations = None
    if port_series is not None and not port_series.empty:
        series_clean = port_series.dropna()
        observations = int(series_clean.shape[0])
        positive_months = int((series_clean > 0).sum()) if observations else 0

    risk_diag = res.get("risk_diagnostics")
    turnover_avg = None
    if isinstance(risk_diag, Mapping):
        turnover_val = risk_diag.get("turnover_value")
        turnover_avg = _to_float(turnover_val)
    if turnover_avg is None:
        turnover_avg = _to_float(res.get("turnover"))

    transaction_cost_avg = _to_float(res.get("transaction_cost"))

    return {
        "analysis_period": analysis_period,
        "portfolio_label": "Portfolio",
        "out_total_return": _format_pct(total_return),
        "out_cagr": _format_pct(out_cagr),
        "out_volatility": _format_pct(out_vol),
        "out_sharpe": _format_number(out_sharpe),
        "out_sortino": _format_number(out_sortino),
        "out_max_drawdown": _format_pct(out_max_dd),
        "top_contributor": top_contributor,
        "top_contributor_return": top_contributor_return,
        "manager_count": _format_number(manager_count, digits=0),
        "avg_weight": _format_pct(avg_weight),
        "positive_months": _format_number(positive_months, digits=0),
        "observations": _format_number(observations, digits=0),
        "top_weight_count": _format_number(top_weight_count, digits=0),
        "top_weight_share": _format_pct(top_weight_share),
        "turnover_avg": _format_pct(turnover_avg),
        "transaction_cost_avg": _format_pct(transaction_cost_avg),
        "return_frequency": _extract_frequency_label(res),
        "analysis_start": start or "unspecified start",
        "analysis_end": end or "unspecified end",
    }


class _SafeMetrics(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "N/A"


def _stringify_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str) and not value.strip():
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    return str(value)


def _append_disclaimer(sections: OrderedDictType[str, str]) -> None:
    if _has_standard_disclaimer(sections):
        return
    sections["disclaimer"] = STANDARD_NARRATIVE_DISCLAIMER


def generate_narrative_sections(
    metrics: Mapping[str, str],
    templates: OrderedDictType[str, NarrativeTemplateSection] | None = None,
) -> OrderedDictType[str, str]:
    """Render narrative sections using templates and provided metrics."""
    sections: OrderedDictType[str, str] = OrderedDictType()
    safe = _SafeMetrics({str(k): _stringify_metric(v) for k, v in metrics.items()})
    for key, template in (templates or DEFAULT_NARRATIVE_TEMPLATES).items():
        sections[key] = template.template.format_map(safe)
    _append_disclaimer(sections)
    return sections


def build_narrative_sections(
    res: Mapping[str, Any],
    templates: OrderedDictType[str, NarrativeTemplateSection] | None = None,
) -> OrderedDictType[str, str]:
    """End-to-end metric extraction and rendering."""
    metrics = extract_narrative_metrics(res)
    return generate_narrative_sections(metrics, templates=templates)
