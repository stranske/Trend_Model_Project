from __future__ import annotations

from pathlib import Path

import yaml

from trend_analysis.monte_carlo.strategy.validation import validate_strategy_pack


def test_validate_strategy_pack_accepts_hf_equity_curated() -> None:
    pack_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    errors = validate_strategy_pack(pack_path)
    assert errors == []


def test_validate_strategy_pack_rejects_non_list_curated(tmp_path: Path) -> None:
    pack_path = tmp_path / "invalid_pack.yml"
    pack_path.write_text(yaml.safe_dump({"curated": "not-a-list"}), encoding="utf-8")

    errors = validate_strategy_pack(pack_path)

    assert errors == ["strategy_pack.curated must be a list"]


def test_validate_strategy_pack_rejects_missing_curated(tmp_path: Path) -> None:
    pack_path = tmp_path / "missing_curated.yml"
    pack_path.write_text(yaml.safe_dump({"foo": "bar"}), encoding="utf-8")

    errors = validate_strategy_pack(pack_path)

    assert errors == ["strategy_pack.curated is required"]


def test_validate_strategy_pack_rejects_invalid_entry_type(tmp_path: Path) -> None:
    pack_path = tmp_path / "invalid_entry.yml"
    pack_path.write_text(yaml.safe_dump({"curated": [123]}), encoding="utf-8")

    errors = validate_strategy_pack(pack_path)

    assert errors == ["strategy_pack.curated[0] invalid: entry must be a mapping or string"]


def test_validate_strategy_pack_rejects_missing_name(tmp_path: Path) -> None:
    pack_path = tmp_path / "missing_name.yml"
    pack_path.write_text(yaml.safe_dump({"curated": [{"overrides": {}}]}), encoding="utf-8")

    errors = validate_strategy_pack(pack_path)

    assert errors == ["strategy_pack.curated[0] invalid: name is required"]


def test_validate_strategy_pack_rejects_non_mapping_overrides(tmp_path: Path) -> None:
    pack_path = tmp_path / "invalid_overrides.yml"
    pack_path.write_text(
        yaml.safe_dump({"curated": [{"name": "BadOverrides", "overrides": "nope"}]}),
        encoding="utf-8",
    )

    errors = validate_strategy_pack(pack_path)

    assert errors == ["strategy_pack.curated[0] invalid: overrides must be a mapping"]


def test_validate_strategy_pack_rejects_invalid_weighting_scheme(tmp_path: Path) -> None:
    pack_path = tmp_path / "invalid_weighting.yml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "curated": [
                    {
                        "name": "BadWeighting",
                        "overrides": {"portfolio": {"weighting_scheme": "not_a_scheme"}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    errors = validate_strategy_pack(pack_path)

    assert any("portfolio.weighting_scheme" in error for error in errors)
