import pandas as pd

from trend_analysis.reporting.narrative import (
    STANDARD_NARRATIVE_DISCLAIMER,
    build_narrative_sections,
    extract_narrative_metrics,
    validate_narrative_quality,
)


def _sample_report_payload():
    out_sample_scaled = pd.DataFrame(
        {
            "Alpha": [0.02, -0.01, 0.03],
            "Beta": [0.01, 0.0, 0.01],
        },
        index=pd.date_range("2020-01-31", periods=3, freq="M"),
    )
    return {
        "out_user_stats": (0.12, 0.2, 1.25, 1.8, 0.5, -0.1),
        "out_sample_scaled": out_sample_scaled,
        "fund_weights": {"Alpha": 0.7, "Beta": 0.3},
        "risk_diagnostics": {"turnover_value": 0.1},
        "transaction_cost": 0.002,
        "metadata": {
            "lookbacks": {"out_sample": {"start": "2020-01", "end": "2020-03"}},
            "frequency": {"label": "Monthly", "target_label": "Monthly"},
        },
    }


def test_validate_narrative_quality_passes_for_generated_sections():
    res = _sample_report_payload()
    sections = build_narrative_sections(res)
    issues = validate_narrative_quality(sections)

    metrics = extract_narrative_metrics(res)
    assert metrics["out_total_return"] in sections["executive_summary"]
    assert any(
        STANDARD_NARRATIVE_DISCLAIMER in section for section in sections.values()
    )
    assert issues == []


def test_validate_narrative_quality_flags_forward_looking_text():
    sections = build_narrative_sections(_sample_report_payload())
    sections["key_findings"] = "We will improve next year based on these results."

    issues = validate_narrative_quality(sections)

    assert any(issue.kind == "forward_looking" for issue in issues)


def test_validate_narrative_quality_flags_missing_disclaimer_text():
    sections = build_narrative_sections(_sample_report_payload())
    sections["methodology_note"] = "Metrics are computed from monthly returns."

    issues = validate_narrative_quality(sections)

    assert any(issue.kind == "missing_disclaimer_text" for issue in issues)
