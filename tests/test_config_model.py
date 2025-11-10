"""Tests for trend_analysis.config.model validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest
from pydantic import ValidationError

pytest.importorskip("yaml")

from trend_analysis.config import model as config_model


class TestResolvePath:
    def test_prefers_base_directory_for_relative_file(self, tmp_path: Path) -> None:
        base = tmp_path / "config"
        base.mkdir()
        target = base / "data.csv"
        target.write_text("col\n1\n", encoding="utf-8")

        resolved = config_model._resolve_path("data.csv", base_dir=base)

        assert resolved == target.resolve()

    def test_rejects_missing_files(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            config_model._resolve_path(tmp_path / "missing.csv", base_dir=tmp_path)

    def test_rejects_directories(self, tmp_path: Path) -> None:
        directory = tmp_path / "folder"
        directory.mkdir()
        with pytest.raises(ValueError, match="points to a directory"):
            config_model._resolve_path(directory, base_dir=None)

    def test_rejects_glob_patterns(self, tmp_path: Path) -> None:
        file_path = tmp_path / "data.csv"
        file_path.write_text("col\n1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="contains wildcard characters"):
            config_model._resolve_path("*.csv", base_dir=tmp_path)

    def test_uses_current_working_directory_when_base_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        csv_path = tmp_path / "local.csv"
        csv_path.write_text("v\n1\n", encoding="utf-8")

        resolved = config_model._resolve_path("local.csv", base_dir=None)

        assert resolved == csv_path.resolve()


class TestGlobHelpers:
    def test_expand_pattern_includes_base_and_parent(self, tmp_path: Path) -> None:
        base = tmp_path / "configs" / "nested"
        base.mkdir(parents=True)

        candidates = config_model._expand_pattern("demo/*.yml", base_dir=base)

        expected = [
            base / "demo" / "*.yml",
            base.parent / "demo" / "*.yml",
            Path.cwd() / "demo" / "*.yml",
        ]
        assert candidates[:3] == expected

    def test_ensure_glob_matches_raises_when_no_csv(self, tmp_path: Path) -> None:
        base = tmp_path / "inputs"
        base.mkdir()
        pattern = str(base / "*.csv")

        with pytest.raises(ValueError, match="did not match any CSV files"):
            config_model._ensure_glob_matches(pattern, base_dir=base)

    def test_ensure_glob_matches_rejects_non_csv(self, tmp_path: Path) -> None:
        base = tmp_path / "inputs"
        base.mkdir()
        (base / "data.txt").write_text("content", encoding="utf-8")
        pattern = str(base / "*.txt")

        with pytest.raises(ValueError, match="must resolve to CSV files"):
            config_model._ensure_glob_matches(pattern, base_dir=base)

    def test_expand_pattern_deduplicates_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        candidates = config_model._expand_pattern("demo/*.yml", base_dir=tmp_path)

        # When base_dir equals cwd the first two entries should be identical and
        # therefore deduplicated, leaving the cwd entry last.
        assert candidates == [tmp_path / "demo" / "*.yml", tmp_path.parent / "demo" / "*.yml"]

    def test_candidate_roots_uses_current_directory_when_base_missing(self) -> None:
        roots = list(config_model._candidate_roots(None))

        assert roots == [Path.cwd()]


class TestDataSettings:
    def test_validates_relative_csv_path(self, tmp_path: Path) -> None:
        base = tmp_path / "conf"
        base.mkdir()
        csv_path = base / "returns.csv"
        csv_path.write_text("date,value\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": "returns.csv",
                "date_column": "date",
                "frequency": "D",
            },
            context={"base_path": base},
        )

        assert settings.csv_path == csv_path

    def test_csv_path_allows_blank_values(self, tmp_path: Path) -> None:
        managers_dir = tmp_path / "managers"
        managers_dir.mkdir()
        (managers_dir / "data.csv").write_text("id\n1\n", encoding="utf-8")
        pattern = str(managers_dir / "*.csv")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": "",
                "managers_glob": pattern,
                "date_column": "date",
                "frequency": "D",
            },
            context={"base_path": tmp_path},
        )

        assert settings.csv_path is None

    def test_csv_path_absolute_without_context(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "returns.csv"
        csv_path.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "date_column": "date",
                "frequency": "D",
            }
        )

        assert settings.csv_path == csv_path

    def test_requires_source_when_paths_missing(self) -> None:
        with pytest.raises(ValueError, match="data.csv_path must point"):
            config_model.DataSettings.model_validate(
                {
                    "date_column": "date",
                    "frequency": "W",
                },
                context={"base_path": Path.cwd()},
            )

    def test_managers_glob_accepts_resolved_path(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "managers.csv"
        csv_file.write_text("id\n1\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "managers_glob": str(csv_file),
                "date_column": "date",
                "frequency": "M",
            },
            context={"base_path": tmp_path},
        )

        assert settings.managers_glob == str(csv_file)

    def test_managers_glob_supports_pathlike_inputs(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "managers.csv"
        csv_file.write_text("id\n1\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "managers_glob": csv_file,
                "date_column": "date",
                "frequency": "D",
            },
            context={"base_path": tmp_path},
        )

        assert settings.managers_glob == str(csv_file)

    def test_managers_glob_absolute_without_context(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "managers.csv"
        csv_file.write_text("id\n1\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "managers_glob": str(csv_file),
                "date_column": "date",
                "frequency": "D",
            }
        )

        assert settings.managers_glob == str(csv_file)

    def test_managers_glob_supports_wildcard_patterns(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "inputs" / "managers.csv"
        csv_file.parent.mkdir()
        csv_file.write_text("id\n1\n", encoding="utf-8")
        pattern = str(csv_file.parent / "*.csv")

        settings = config_model.DataSettings.model_validate(
            {
                "managers_glob": pattern,
                "date_column": "date",
                "frequency": "D",
            },
            context={"base_path": tmp_path},
        )

        assert settings.managers_glob == pattern

    def test_managers_glob_rejects_invalid_types(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a string"):
            config_model.DataSettings.model_validate(
                {
                    "csv_path": str(csv_file),
                    "managers_glob": 42,
                    "date_column": "date",
                    "frequency": "D",
                },
                context={"base_path": tmp_path},
            )

    def test_managers_glob_allows_blank_string(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "managers_glob": "",
                "date_column": "date",
                "frequency": "D",
            },
            context={"base_path": tmp_path},
        )

        assert settings.managers_glob is None

    def test_frequency_validation(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            config_model.DataSettings.model_validate(
                {
                    "csv_path": "demo.csv",
                    "date_column": "date",
                    "frequency": "hourly",
                },
                context={"base_path": Path.cwd()},
            )

    def test_frequency_accepts_month_end_alias(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "ME",
            },
            context={"base_path": tmp_path},
        )

        assert settings.frequency == "ME"

    def test_frequency_requires_value(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be provided"):
            config_model.DataSettings.model_validate(
                {
                    "csv_path": str(csv_file),
                    "date_column": "date",
                    "frequency": None,
                },
                context={"base_path": tmp_path},
            )

    def test_missing_limit_accepts_integer_strings(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
                "missing_limit": "5",
            },
            context={"base_path": tmp_path},
        )

        assert settings.missing_limit == 5

    def test_missing_limit_rejects_objects(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be an integer"):
            config_model.DataSettings.model_validate(
                {
                    "csv_path": str(csv_file),
                    "date_column": "date",
                    "frequency": "D",
                    "missing_limit": object(),
                },
                context={"base_path": tmp_path},
            )

    def test_missing_policy_defaults_to_none(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
            },
            context={"base_path": tmp_path},
        )

        assert settings.missing_policy is None

    def test_missing_policy_empty_string(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
                "missing_policy": "",
            },
            context={"base_path": tmp_path},
        )

        assert settings.missing_policy is None

    def test_missing_policy_rejects_non_mapping(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a string or mapping"):
            config_model.DataSettings.model_validate(
                {
                    "csv_path": str(csv_file),
                    "date_column": "date",
                    "frequency": "D",
                    "missing_policy": ["oops"],
                },
                context={"base_path": tmp_path},
            )

    def test_missing_policy_accepts_mapping(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
                "missing_policy": {"asset": "ffill"},
                "missing_limit": {"asset": 5},
            },
            context={"base_path": tmp_path},
        )

        assert settings.missing_policy == {"asset": "ffill"}
        assert settings.missing_limit == {"asset": 5}

    def test_missing_limit_handles_null_strings(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        settings = config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
                "missing_limit": "null",
            },
            context={"base_path": tmp_path},
        )

        assert settings.missing_limit is None

    def test_date_column_requires_non_empty_string(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            config_model.DataSettings.model_validate(
                {
                    "csv_path": str(csv_file),
                    "date_column": "  ",
                    "frequency": "D",
                },
                context={"base_path": tmp_path},
            )


class TestPortfolioSettings:
    def test_rejects_blank_calendar(self) -> None:
        with pytest.raises(ValueError, match="must name a valid trading calendar"):
            config_model.PortfolioSettings.model_validate(
                {
                    "rebalance_calendar": "",
                    "max_turnover": 0.2,
                    "transaction_cost_bps": 5,
                }
            )

    def test_rejects_turnover_above_one(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1.0"):
            config_model.PortfolioSettings.model_validate(
                {
                    "rebalance_calendar": "NYSE",
                    "max_turnover": 1.1,
                    "transaction_cost_bps": 5,
                }
            )

    def test_rejects_negative_transaction_costs(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            config_model.PortfolioSettings.model_validate(
                {
                    "rebalance_calendar": "NYSE",
                    "max_turnover": 0.2,
                    "transaction_cost_bps": -1,
                }
            )

    def test_rejects_negative_turnover(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            config_model.PortfolioSettings.model_validate(
                {
                    "rebalance_calendar": "NYSE",
                    "max_turnover": -0.1,
                    "transaction_cost_bps": 5,
                }
            )


class TestRiskSettings:
    def test_requires_positive_target(self) -> None:
        with pytest.raises(ValueError, match="greater than zero"):
            config_model.RiskSettings.model_validate(
                {"target_vol": 0, "floor_vol": 0.01, "warmup_periods": 0}
            )

    def test_rejects_negative_floor(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            config_model.RiskSettings.model_validate(
                {"target_vol": 0.2, "floor_vol": -0.1, "warmup_periods": 0}
            )

    def test_rejects_negative_warmup(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            config_model.RiskSettings.model_validate(
                {"target_vol": 0.2, "floor_vol": 0.1, "warmup_periods": -5}
            )


class TestConfigLoading:
    def test_resolve_config_prefers_repo_defaults(self) -> None:
        resolved = config_model._resolve_config_path("demo")
        assert resolved.name == "demo.yml"
        assert resolved.exists()

    def test_resolve_config_uses_environment_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = tmp_path / "custom.yml"
        cfg.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("TREND_CONFIG", str(cfg))

        resolved = config_model._resolve_config_path(None)

        assert resolved == cfg.resolve()

    def test_resolve_config_defaults_to_demo_when_no_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TREND_CONFIG", raising=False)
        monkeypatch.delenv("TREND_CFG", raising=False)

        resolved = config_model._resolve_config_path(None)

        assert resolved.name == "demo.yml"

    def test_resolve_config_requires_existing_files(self) -> None:
        with pytest.raises(FileNotFoundError):
            config_model._resolve_config_path("missing-config")

    def test_validate_trend_config_returns_model(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "prices.csv"
        csv_file.write_text("col\n1\n", encoding="utf-8")
        payload: Mapping[str, object] = {
            "data": {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
            },
            "portfolio": {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.4,
                "transaction_cost_bps": 10,
            },
            "vol_adjust": {
                "target_vol": 0.2,
                "floor_vol": 0.01,
                "warmup_periods": 2,
            },
        }

        cfg = config_model.validate_trend_config(payload, base_path=tmp_path)

        assert cfg.data.csv_path == csv_file
        assert cfg.portfolio.max_turnover == 0.4
        assert cfg.vol_adjust.target_vol == 0.2

    def test_validate_trend_config_surfaces_first_error(self, tmp_path: Path) -> None:
        payload = {
            "data": {
                "date_column": "date",
                "frequency": "D",
            },
            "portfolio": {
                "rebalance_calendar": "",
                "max_turnover": 0.4,
                "transaction_cost_bps": 10,
            },
            "vol_adjust": {
                "target_vol": 0.0,
                "floor_vol": 0.01,
                "warmup_periods": -1,
            },
        }

        with pytest.raises(ValueError, match="data.csv_path"):
            config_model.validate_trend_config(payload, base_path=tmp_path)

    def test_validate_trend_config_reports_error_location(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")
        payload = {
            "data": {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
            },
            "portfolio": {
                "rebalance_calendar": "",
                "max_turnover": 0.2,
                "transaction_cost_bps": 5,
            },
            "vol_adjust": {
                "target_vol": 0.2,
                "floor_vol": 0.01,
                "warmup_periods": 1,
            },
        }

        with pytest.raises(ValueError, match="portfolio.rebalance_calendar"):
            config_model.validate_trend_config(payload, base_path=tmp_path)

    def test_validate_trend_config_reports_nested_location(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "returns.csv"
        csv_file.write_text("col\n", encoding="utf-8")
        payload = {
            "data": {
                "csv_path": str(csv_file),
                "date_column": "date",
                "frequency": "D",
            },
            "portfolio": {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.5,
                "transaction_cost_bps": 10,
            },
            "vol_adjust": {
                "target_vol": -0.1,
                "floor_vol": 0.01,
                "warmup_periods": 1,
            },
        }

        with pytest.raises(ValueError, match="vol_adjust.target_vol"):
            config_model.validate_trend_config(payload, base_path=tmp_path)

    def test_load_trend_config_reads_yaml_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "prices.csv"
        csv_file.write_text("date,value\n", encoding="utf-8")
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
            data:
              csv_path: prices.csv
              date_column: date
              frequency: D
            portfolio:
              rebalance_calendar: NYSE
              max_turnover: 0.5
              transaction_cost_bps: 12
            vol_adjust:
              target_vol: 0.15
              floor_vol: 0.01
              warmup_periods: 3
            """,
            encoding="utf-8",
        )

        cfg, path = config_model.load_trend_config(config_file)

        assert path == config_file.resolve()
        assert cfg.data.csv_path == csv_file
        assert cfg.portfolio.transaction_cost_bps == 12

    def test_load_trend_config_rejects_non_mapping_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "invalid.yml"
        config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

        with pytest.raises(TypeError, match="must contain a mapping"):
            config_model.load_trend_config(config_file)

    def test_validate_trend_config_handles_non_mapping_payload(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="valid dictionary"):
            config_model.validate_trend_config([], base_path=tmp_path)

    def test_validate_trend_config_handles_validation_error_without_details(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_model_validate(cls, data, context):
            raise ValidationError.from_exception_data("TrendConfig", [])

        monkeypatch.setattr(
            config_model.TrendConfig,
            "model_validate",
            classmethod(fake_model_validate),
        )

        with pytest.raises(ValueError, match="TrendConfig"):
            config_model.validate_trend_config({}, base_path=tmp_path)

    def test_validate_trend_config_error_includes_joined_location(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = ValidationError.from_exception_data(
            "TrendConfig",
            [
                {
                    "type": "value_error",
                    "loc": ("portfolio", "rebalance_calendar"),
                    "msg": "invalid calendar",
                    "input": None,
                    "ctx": {"error": ValueError("invalid calendar")},
                }
            ],
        )

        def fake_validate(cls, data, context):
            raise error

        monkeypatch.setattr(
            config_model.TrendConfig,
            "model_validate",
            classmethod(fake_validate),
        )

        with pytest.raises(ValueError, match="portfolio.rebalance_calendar: Value error, invalid calendar"):
            config_model.validate_trend_config({}, base_path=tmp_path)
