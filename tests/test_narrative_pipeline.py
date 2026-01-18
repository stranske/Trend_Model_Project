import pandas as pd

from trend_analysis.reporting.narrative import (
    DEFAULT_NARRATIVE_TEMPLATES,
    STANDARD_NARRATIVE_DISCLAIMER,
    build_narrative_sections,
    extract_narrative_metrics,
    generate_narrative_sections,
)


def test_extract_narrative_metrics_from_report_payload():
    out_sample_scaled = pd.DataFrame(
        {
            "Alpha": [0.02, -0.01, 0.03],
            "Beta": [0.01, 0.0, 0.01],
        },
        index=pd.date_range("2020-01-31", periods=3, freq="M"),
    )
    res = {
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

    metrics = extract_narrative_metrics(res)

    assert metrics["analysis_period"] == "2020-01 to 2020-03"
    assert metrics["analysis_start"] == "2020-01"
    assert metrics["analysis_end"] == "2020-03"
    assert metrics["out_total_return"] == "3.41"
    assert metrics["out_cagr"] == "12.00"
    assert metrics["out_volatility"] == "20.00"
    assert metrics["out_sharpe"] == "1.25"
    assert metrics["out_sortino"] == "1.80"
    assert metrics["out_max_drawdown"] == "-10.00"
    assert metrics["top_contributor"] == "Alpha"
    assert metrics["top_contributor_return"] == "4.01"
    assert metrics["manager_count"] == "2"
    assert metrics["avg_weight"] == "50.00"
    assert metrics["positive_months"] == "2"
    assert metrics["observations"] == "3"
    assert metrics["top_weight_count"] == "2"
    assert metrics["top_weight_share"] == "100.00"
    assert metrics["turnover_avg"] == "10.00"
    assert metrics["transaction_cost_avg"] == "0.20"
    assert metrics["return_frequency"] == "Monthly"


def test_build_narrative_sections_formats_templates():
    res = {
        "out_user_stats": (0.1, 0.15, 1.1, 1.4, 0.4, -0.08),
        "out_sample_scaled": pd.DataFrame(
            {"Alpha": [0.01, 0.0], "Beta": [0.02, -0.01]},
            index=pd.date_range("2021-01-31", periods=2, freq="M"),
        ),
        "fund_weights": {"Alpha": 0.6, "Beta": 0.4},
        "metadata": {
            "lookbacks": {"out_sample": {"start": "2021-01", "end": "2021-02"}},
            "frequency": {"label": "Monthly"},
        },
    }

    metrics = extract_narrative_metrics(res)
    sections = generate_narrative_sections(metrics, templates=DEFAULT_NARRATIVE_TEMPLATES)
    built = build_narrative_sections(res)

    expected_keys = list(DEFAULT_NARRATIVE_TEMPLATES)
    assert list(sections) == expected_keys
    assert list(built) == expected_keys
    for section_text in sections.values():
        assert "{" not in section_text
        assert "}" not in section_text
    assert "Monthly returns" in sections["methodology_note"]
    assert STANDARD_NARRATIVE_DISCLAIMER in sections["methodology_note"]


def test_generate_narrative_sections_handles_sparse_metrics():
    res = {
        "out_user_stats": (None, float("nan"), None, None, None, None),
        "out_sample_scaled": pd.DataFrame(),
        "metadata": {},
    }

    sections = build_narrative_sections(res)

    assert "nan" not in sections["executive_summary"].lower()
    assert "N/A" in sections["executive_summary"]
