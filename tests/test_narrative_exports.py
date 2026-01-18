from collections import OrderedDict
import logging

import pandas as pd

from trend_analysis import export
from trend_analysis.reporting.narrative import STANDARD_NARRATIVE_DISCLAIMER


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


def test_export_includes_narrative_across_formats(tmp_path):
    res = _sample_report_payload()
    data = {"metrics": pd.DataFrame({"Metric": [1]})}
    export.append_narrative_section(data, res, config={"export": {}})

    assert "narrative" in data

    out = tmp_path / "analysis"
    export.export_data(data, str(out), formats=["csv", "txt", "xlsx"])

    csv_path = tmp_path / "analysis_narrative.csv"
    txt_path = tmp_path / "analysis_narrative.txt"
    xlsx_path = tmp_path / "analysis.xlsx"

    assert csv_path.exists()
    assert txt_path.exists()
    assert xlsx_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    txt_text = txt_path.read_text(encoding="utf-8")
    assert STANDARD_NARRATIVE_DISCLAIMER in csv_text
    assert STANDARD_NARRATIVE_DISCLAIMER in txt_text

    narrative_sheet = pd.read_excel(xlsx_path, sheet_name="narrative")
    assert any(
        STANDARD_NARRATIVE_DISCLAIMER in text for text in narrative_sheet["Narrative"].to_list()
    )


def test_export_respects_disable_narrative_generation():
    res = _sample_report_payload()
    data = {"metrics": pd.DataFrame({"Metric": [1]})}
    export.append_narrative_section(
        data,
        res,
        config={"export": {"disable_narrative_generation": True}},
    )

    assert "narrative" not in data


def test_narrative_frame_logs_quality_issues(monkeypatch, caplog):
    def fake_sections(_: object) -> OrderedDict[str, str]:
        return OrderedDict(
            [
                ("key_findings", "We will improve next year based on these results."),
                ("methodology_note", "Metrics are computed from monthly returns."),
            ]
        )

    monkeypatch.setattr(export, "build_narrative_sections", fake_sections)
    with caplog.at_level(logging.WARNING):
        frame = export.narrative_frame_from_result({}, config={"export": {}})

    assert frame is not None
    messages = " ".join(caplog.messages)
    assert "forward_looking" in messages
    assert "missing_disclaimer_text" in messages
