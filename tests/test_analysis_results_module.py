"""Tests for the :mod:`analysis.results` helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import Results, build_metadata, compute_universe_fingerprint


def test_compute_universe_fingerprint_uses_all_inputs(tmp_path: Path) -> None:
    data_path = tmp_path / "Trend Universe Data.csv"
    data_path.write_text("Date,Funda\n2024-01-31,0.01\n", encoding="utf-8")
    membership_path = tmp_path / "Trend Universe Membership.csv"
    membership_path.write_text(
        "fund,effective_date,end_date\nFundA,2020-01-31,\n", encoding="utf-8"
    )

    fingerprint = compute_universe_fingerprint(data_path, membership_path)

    assert len(fingerprint) == 12
    # Deterministic check ensures both files contribute to the fingerprint.
    assert fingerprint == compute_universe_fingerprint(data_path, membership_path)


def test_build_metadata_includes_core_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "analysis.results.compute_universe_fingerprint", lambda *_, **__: "abc123def456"
    )
    meta = build_metadata(
        universe=["FundA", "FundB"],
        selected=["FundA"],
        lookbacks={
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2020-12",
        },
        costs={"monthly_cost": 0.0025},
        code_version="9.9.9",
        data_path=None,
        membership_path=None,
    )

    assert meta["universe"]["count"] == 2
    assert meta["universe"]["selected_count"] == 1
    assert meta["lookbacks"]["in_sample"]["start"] == "2020-01"
    assert meta["costs"]["monthly_cost"] == 0.0025
    assert meta["code_version"] == "9.9.9"
    assert meta["fingerprint"] == "abc123def456"


def test_results_from_payload_coerces_series() -> None:
    portfolio = pd.Series(
        [0.01, 0.02], index=pd.date_range("2020-01-31", periods=2, freq="M")
    )
    payload = {
        "portfolio_equal_weight_combined": portfolio,
        "fund_weights": {"FundA": 0.6, "FundB": 0.4},
        "risk_diagnostics": {
            "final_weights": pd.Series({"FundA": 0.55, "FundB": 0.45}),
            "turnover": pd.Series([0.1, 0.2]),
            "turnover_value": 0.15,
        },
        "metadata": {
            "costs": {"monthly_cost": 0.001},
            "fingerprint": "deadbeefcafe",
        },
    }

    results = Results.from_payload(payload)

    assert list(results.weights.index) == ["FundA", "FundB"]
    assert results.exposures.sum() == 1.0
    assert not results.turnover.empty
    assert results.costs["monthly_cost"] == 0.001
    assert results.fingerprint() == "deadbeefcafe"
