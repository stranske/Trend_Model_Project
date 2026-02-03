from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trend_analysis.monte_carlo.strategy.validation import validate_strategy_pack


def test_validate_strategy_pack_accepts_hf_equity_curated() -> None:
    pack_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    errors = validate_strategy_pack(pack_path)
    assert errors == []


def test_validate_strategy_pack_reports_base_config_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path = tmp_path / "defaults.yml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "max_turnover": 0.5,
                    "weighting_scheme": "equal",
                    "weighting": {"name": "equal", "params": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    pack_path = tmp_path / "pack.yml"
    pack_path.write_text(
        yaml.safe_dump({"curated": [{"name": "Mutate", "overrides": {}}]}),
        encoding="utf-8",
    )

    def _mutating_apply(self, base_config: dict[str, object]) -> dict[str, object]:
        base_config["portfolio"]["max_turnover"] = 0.9  # type: ignore[index]
        return base_config

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.strategy.validation.StrategyVariant.apply_to",
        _mutating_apply,
    )

    errors = validate_strategy_pack(pack_path, base_config_path=base_path)

    assert any("base_config mutated during validation" in error for error in errors)


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


def test_validate_strategy_pack_rejects_unsupported_keys(tmp_path: Path) -> None:
    pack_path = tmp_path / "invalid_keys.yml"
    pack_path.write_text(
        yaml.safe_dump({"curated": [{"name": "BadKeys", "extra": 123}]}),
        encoding="utf-8",
    )

    errors = validate_strategy_pack(pack_path)

    assert errors == ["strategy_pack.curated[0] invalid: unsupported keys: extra"]


def test_validate_strategy_pack_rejects_mapping_tags(tmp_path: Path) -> None:
    pack_path = tmp_path / "invalid_tags.yml"
    pack_path.write_text(
        yaml.safe_dump({"curated": [{"name": "BadTags", "tags": {"bad": True}}]}),
        encoding="utf-8",
    )

    errors = validate_strategy_pack(pack_path)

    assert errors == [
        "strategy_pack.curated[0] invalid: tags must be a string or sequence of strings"
    ]


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
