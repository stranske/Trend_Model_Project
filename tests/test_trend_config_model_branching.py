from pathlib import Path

import pytest

from trend_analysis.config import model as config_model


def _base_data(**overrides: object) -> dict[str, object]:
    data = {
        "csv_path": None,
        "managers_glob": None,
        "date_column": "Date",
        "frequency": "M",
        "missing_policy": None,
        "missing_limit": None,
    }
    data.update(overrides)
    return data


def test_candidate_roots_includes_base_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base_dir = tmp_path / "cfg"
    base_dir.mkdir()

    roots = list(config_model._candidate_roots(base_dir))

    assert roots[0] == base_dir
    assert roots[1] == base_dir.parent
    assert roots[-1] == tmp_path


def test_expand_pattern_deduplicates_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base_dir = Path.cwd()

    expanded = config_model._expand_pattern("data/*.csv", base_dir=base_dir)

    assert expanded[0] == base_dir / "data" / "*.csv"
    assert expanded.count(base_dir / "data" / "*.csv") == 1


def test_validate_csv_path_uses_context_base(tmp_path: Path) -> None:
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    csv_file = data_dir / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    result = config_model.DataSettings.model_validate(
        _base_data(csv_path="returns.csv"),
        context={"base_path": data_dir},
    )

    assert result.csv_path == csv_file.resolve()


def test_validate_managers_glob_respects_context(tmp_path: Path) -> None:
    managers_dir = tmp_path / "managers"
    managers_dir.mkdir()
    (managers_dir / "fund.csv").write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    result = config_model.DataSettings.model_validate(
        _base_data(managers_glob="managers/*.csv", csv_path=None),
        context={"base_path": tmp_path},
    )

    assert result.managers_glob == "managers/*.csv"


def test_date_column_rejects_blank_string(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        config_model.DataSettings.model_validate(
            _base_data(date_column=" ", csv_path=str(csv_file))
        )


@pytest.mark.parametrize("policy", [None, ""])  # cover nullish branch
def test_missing_policy_accepts_nullish(tmp_path: Path, policy: object) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    result = config_model.DataSettings.model_validate(
        _base_data(missing_policy=policy, csv_path=str(csv_file))
    )
    assert result.missing_policy is None


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("null", None),
        ({"A": 2}, {"A": 2}),
    ],
)
def test_missing_limit_coercions(
    tmp_path: Path, spec: object, expected: object
) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    result = config_model.DataSettings.model_validate(
        _base_data(missing_limit=spec, csv_path=str(csv_file))
    )
    assert result.missing_limit == expected


def test_portfolio_calendar_requires_name() -> None:
    with pytest.raises(ValueError):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": " ",
                "max_turnover": 0.5,
                "transaction_cost_bps": 0,
            }
        )


def test_portfolio_max_turnover_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": -0.1,
                "transaction_cost_bps": 0,
            }
        )


def test_risk_settings_negative_floor_rejected() -> None:
    with pytest.raises(ValueError):
        config_model.RiskSettings.model_validate(
            {
                "target_vol": 0.1,
                "floor_vol": -0.5,
                "warmup_periods": 0,
            }
        )


def test_risk_settings_negative_warmup_rejected() -> None:
    with pytest.raises(ValueError):
        config_model.RiskSettings.model_validate(
            {
                "target_vol": 0.1,
                "floor_vol": 0.0,
                "warmup_periods": -1,
            }
        )


def test_resolve_config_path_defaults_to_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TREND_CONFIG", raising=False)
    monkeypatch.delenv("TREND_CFG", raising=False)

    path = config_model._resolve_config_path(None)

    assert path.name == "demo.yml"
    assert path.exists()


def test_validate_trend_config_reports_field_location(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    config = {
        "version": "1",
        "data": {
            "csv_path": str(csv_file),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 2,  # invalid, should trigger validator
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        config_model.validate_trend_config(config, base_path=tmp_path)

    assert str(exc.value).startswith("portfolio.max_turnover")
