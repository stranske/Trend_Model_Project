from __future__ import annotations

import json
import zipfile
from io import BytesIO

import plotly.graph_objects as go

from trend_analysis.monte_carlo.export_bundle import save, save_to_tempfile


def _sample_charts() -> dict[str, go.Figure]:
    return {
        "Equity Curve": go.Figure(
            data=[go.Scatter(x=["2025-01-01", "2025-01-02"], y=[100.0, 102.0], mode="lines")]
        ),
        "Risk Return": go.Figure(data=[go.Scatter(x=[0.08, 0.12], y=[0.10, 0.14], mode="markers")]),
    }


def test_save_writes_plotly_json_and_html_to_bytesio() -> None:
    buffer = save(_sample_charts())
    assert isinstance(buffer, BytesIO)
    assert buffer.tell() == 0

    with zipfile.ZipFile(buffer) as bundle:
        names = set(bundle.namelist())
        assert names == {
            "Equity_Curve.json",
            "Equity_Curve.html",
            "Risk_Return.json",
            "Risk_Return.html",
        }
        chart_payload = json.loads(bundle.read("Equity_Curve.json").decode("utf-8"))
        assert "data" in chart_payload
        html_payload = bundle.read("Risk_Return.html").decode("utf-8")
        assert "<html" in html_payload.lower()
        assert "plotly" in html_payload.lower()


def test_save_writes_plotly_json_and_html_to_temp_path(tmp_path) -> None:
    out_path = tmp_path / "charts_bundle.zip"
    returned = save(_sample_charts().items(), destination=out_path)
    assert returned == out_path
    assert out_path.exists()

    with zipfile.ZipFile(out_path) as bundle:
        names = set(bundle.namelist())
        assert "Equity_Curve.json" in names
        assert "Equity_Curve.html" in names
        assert "Risk_Return.json" in names
        assert "Risk_Return.html" in names


def test_save_to_tempfile_creates_zip_path() -> None:
    out_path = save_to_tempfile(_sample_charts())
    try:
        assert out_path.exists()
        with zipfile.ZipFile(out_path) as bundle:
            names = set(bundle.namelist())
            assert "Equity_Curve.json" in names
            assert "Equity_Curve.html" in names
    finally:
        out_path.unlink(missing_ok=True)
