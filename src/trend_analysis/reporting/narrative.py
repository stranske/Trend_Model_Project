"""Narrative section templates for report exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import OrderedDict as OrderedDictType


@dataclass(frozen=True)
class NarrativeTemplateSection:
    """Template metadata for a narrative section."""

    key: str
    title: str
    template: str
    placeholders: tuple[str, ...]


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
            "schedule. This narrative is auto-generated from observed results, "
            "excludes forward-looking statements, and should be treated as a "
            "descriptive summary rather than investment advice."
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
    "default_narrative_templates",
]
