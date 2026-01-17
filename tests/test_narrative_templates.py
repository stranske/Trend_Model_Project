from __future__ import annotations

from trend_analysis.reporting.narrative import DEFAULT_NARRATIVE_TEMPLATES


def test_default_narrative_templates_order_and_keys():
    assert list(DEFAULT_NARRATIVE_TEMPLATES) == [
        "executive_summary",
        "key_findings",
        "risk_highlights",
        "methodology_note",
    ]


def test_default_narrative_templates_placeholders_present():
    for section in DEFAULT_NARRATIVE_TEMPLATES.values():
        for placeholder in section.placeholders:
            token = f"{{{placeholder}}}"
            assert token in section.template
