from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_user_guide_matches_regime_annualisation_contract() -> None:
    guide = (ROOT / "docs" / "UserGuide.md").read_text(encoding="utf-8")
    defaults = (ROOT / "config" / "defaults.yml").read_text(encoding="utf-8")

    assert "classification invariant when annualisation is toggled" in guide
    assert "pre-annualise either boundary" in guide
    assert "threshold, and neutral band together" in guide
    assert "per-period volatility boundary; scaled with the signal" in defaults
    assert "threshold is interpreted as\nannualised volatility" not in guide
