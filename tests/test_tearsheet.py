from pathlib import Path

import pandas as pd

from analysis import Results
from analysis.tearsheet import DEFAULT_OUTPUT, load_results_payload, render


def _dummy_results() -> Results:
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    returns = pd.Series(0.01, index=dates)
    exposures = pd.Series(0.1, index=[f"Fund{i}" for i in range(1, 11)])
    turnover = pd.Series(0.0, index=dates)
    return Results(
        returns=returns,
        weights=exposures,
        exposures=exposures,
        turnover=turnover,
        costs={"turnover_applied": 0.001},
        metadata={"fingerprint": "abc123"},
    )


def test_render_writes_markdown_and_plot(tmp_path: Path) -> None:
    md_path = tmp_path / "tearsheet.md"
    results = _dummy_results()

    out, plot = render(results, out=md_path)

    text = out.read_text(encoding="utf-8")
    assert "Headline statistics" in text
    assert "Sharpe" in text
    assert plot.exists()


def test_load_results_payload_bootstrap(tmp_path: Path, monkeypatch) -> None:
    sentinel = tmp_path / "missing.json"
    monkeypatch.chdir(tmp_path)

    results = load_results_payload(sentinel)

    assert not results.returns.empty
    assert results.fingerprint() is not None
    # Should persist for subsequent calls
    assert sentinel.exists()


def test_default_output_constant() -> None:
    assert DEFAULT_OUTPUT.name == "tearsheet.md"
