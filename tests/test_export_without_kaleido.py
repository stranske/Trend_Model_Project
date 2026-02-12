from __future__ import annotations

import zipfile

import plotly.graph_objects as go
import pytest

from trend_analysis.monte_carlo import export_bundle


def test_export_without_kaleido_skips_png_and_records_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(export_bundle, "kaleido_available", lambda: False)

    def _to_image_should_not_run(*_args, **_kwargs):  # pragma: no cover - guard assertion
        raise AssertionError("pio.to_image should not be called when Kaleido is unavailable")

    monkeypatch.setattr(export_bundle.pio, "to_image", _to_image_should_not_run)

    charts = {"paths": go.Figure(data=[go.Scatter(x=[1, 2], y=[1.0, 1.1])])}
    warnings: list[str] = []
    buffer = export_bundle.save(charts, include_png=True, warnings=warnings)

    with zipfile.ZipFile(buffer) as bundle:
        names = set(bundle.namelist())
        assert "paths.png" not in names
        assert "paths.json" in names
        assert "paths.html" in names

    assert warnings
    assert "Kaleido is not installed" in warnings[0]
