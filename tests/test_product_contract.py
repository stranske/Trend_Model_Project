"""Regression checks for the analyst-facing product contract."""

from pathlib import Path


def test_product_contract_lists_core_function_c4() -> None:
    contract = (Path(__file__).resolve().parents[1] / "docs/PRODUCT_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    rows = [line.split("|")[1:-1] for line in contract.splitlines() if line.startswith("|")]
    c4_rows = [row for row in rows if row[0].strip() == "C4"]
    assert len(c4_rows) == 1, "The core-functions table must list C4 exactly once"
    row = [cell.strip() for cell in c4_rows[0]]
    assert len(row) == 5
    assert "Performance by Regime" in row[1]
    assert "export" in row[1].lower()
    assert "trend report" in row[2]
    assert "regime.enabled" in row[2]
    assert "tests/test_multi_period_regime_exports.py" in row[3]
    assert row[4].startswith("WORKS")
