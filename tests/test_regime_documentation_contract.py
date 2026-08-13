from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_user_guide_matches_regime_annualisation_contract() -> None:
    guide = (ROOT / "docs" / "UserGuide.md").read_text(encoding="utf-8")
    defaults = (ROOT / "config" / "defaults.yml").read_text(encoding="utf-8")
    demo = (ROOT / "config" / "demo.yml").read_text(encoding="utf-8")
    trend_toml = (ROOT / "config" / "trend.toml").read_text(encoding="utf-8")

    assert "classification invariant when annualisation is toggled" in guide
    assert "pre-annualise either boundary" in guide
    assert "threshold, and neutral band together" in guide
    for config in (defaults, demo, trend_toml):
        assert "rolling-return boundary in compounded-signal units" in config
        assert "rolling-return neutral buffer in compounded-signal units" in config
        assert "only for method: volatility" in config

    rolling_return_block = guide.split('method: "rolling_return"', 1)[1].split("```", 1)[0]
    assert "rolling-return boundary in compounded-signal units" in rolling_return_block
    assert "rolling-return neutral buffer in compounded-signal units" in rolling_return_block
    assert "per-period cut-over" not in rolling_return_block
    assert "per-period neutral buffer" not in rolling_return_block
    assert "threshold is interpreted as\nannualised volatility" not in guide
