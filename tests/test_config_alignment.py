"""Regression tests keeping CoreConfig and TrendConfig aligned."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from trend.config_schema import CoreConfig, validate_core_config
from trend_analysis.config.coverage import (
    ConfigCoverageTracker,
    activate_config_coverage,
    deactivate_config_coverage,
    wrap_config_for_coverage,
)
from trend_analysis.config.model import validate_trend_config
from utils.paths import proj_path


def _load_canonical_config() -> tuple[dict, Path]:
    cfg_path = proj_path() / "config" / "demo.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "demo.yml must contain a mapping"
    return raw, cfg_path


def _build_trend_payload(core_cfg: CoreConfig, raw_cfg: dict) -> dict:
    """Inject CoreConfig's normalised sections back into the raw config."""

    payload = copy.deepcopy(raw_cfg)
    core_payload = core_cfg.to_payload()

    payload["data"] = core_payload["data"]
    portfolio = payload.setdefault("portfolio", {})
    portfolio["transaction_cost_bps"] = core_payload["portfolio"][
        "transaction_cost_bps"
    ]
    portfolio["cost_model"] = core_payload["portfolio"]["cost_model"]
    return payload


def test_core_config_round_trips_into_trend_config() -> None:
    raw_cfg, cfg_path = _load_canonical_config()

    core_cfg = validate_core_config(raw_cfg, base_path=cfg_path.parent)
    trend_payload = _build_trend_payload(core_cfg, raw_cfg)
    trend_cfg = validate_trend_config(trend_payload, base_path=cfg_path.parent)

    if core_cfg.data.csv_path:
        assert trend_cfg.data.csv_path == core_cfg.data.csv_path
    else:
        assert trend_cfg.data.csv_path is None

    if core_cfg.data.managers_glob:
        assert trend_cfg.data.managers_glob == core_cfg.data.managers_glob
    else:
        assert trend_cfg.data.managers_glob is None

    assert trend_cfg.data.date_column == core_cfg.data.date_column
    assert trend_cfg.data.frequency == core_cfg.data.frequency
    assert (
        trend_cfg.data.universe_membership_path
        == core_cfg.data.universe_membership_path
    )

    assert trend_cfg.portfolio.transaction_cost_bps == pytest.approx(
        core_cfg.costs.transaction_cost_bps
    )
    assert trend_cfg.portfolio.cost_model is not None
    assert trend_cfg.portfolio.cost_model.bps_per_trade == pytest.approx(
        core_cfg.costs.bps_per_trade
    )
    assert trend_cfg.portfolio.cost_model.slippage_bps == pytest.approx(
        core_cfg.costs.slippage_bps
    )
    assert trend_cfg.portfolio.cost_model.per_trade_bps == pytest.approx(
        core_cfg.costs.per_trade_bps
    )
    assert trend_cfg.portfolio.cost_model.half_spread_bps == pytest.approx(
        core_cfg.costs.half_spread_bps
    )


def test_core_and_trend_resolve_paths_consistently(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,Mgr_01\n2024-01-31,1.0\n", encoding="utf-8")

    membership = tmp_path / "universe.csv"
    membership.write_text("Mgr_01\n", encoding="utf-8")

    managers_dir = tmp_path / "managers"
    managers_dir.mkdir()
    manager_path = managers_dir / "mgr_01.csv"
    manager_path.write_text("Date,Mgr_01\n2024-01-31,1.0\n", encoding="utf-8")

    raw_cfg = {
        "data": {
            "csv_path": csv_file.name,
            "managers_glob": str(managers_dir.relative_to(tmp_path) / "*.csv"),
            "date_column": "Date",
            "frequency": "M",
            "universe_membership_path": membership.name,
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.25,
            "transaction_cost_bps": 12.5,
        },
        "vol_adjust": {"target_vol": 0.1, "floor_vol": 0.0, "warmup_periods": 0},
    }

    core_cfg = validate_core_config(raw_cfg, base_path=tmp_path)
    trend_payload = _build_trend_payload(core_cfg, raw_cfg)
    trend_cfg = validate_trend_config(trend_payload, base_path=tmp_path)

    assert trend_cfg.data.csv_path == core_cfg.data.csv_path
    assert trend_cfg.data.managers_glob == core_cfg.data.managers_glob
    assert (
        trend_cfg.data.universe_membership_path
        == core_cfg.data.universe_membership_path
    )
    assert trend_cfg.data.frequency == core_cfg.data.frequency
    assert trend_cfg.portfolio.transaction_cost_bps == pytest.approx(
        core_cfg.costs.transaction_cost_bps
    )
    assert trend_cfg.portfolio.cost_model is not None
    assert trend_cfg.portfolio.cost_model.per_trade_bps == pytest.approx(
        core_cfg.costs.per_trade_bps
    )


def test_config_coverage_report_flags_read_and_validation_gaps(
    tmp_path: Path,
) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,Mgr_01\n2024-01-31,1.0\n", encoding="utf-8")

    raw_cfg = {
        "data": {
            "csv_path": str(csv_file),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {"transaction_cost_bps": 0.0},
    }

    tracker = ConfigCoverageTracker()
    activate_config_coverage(tracker)
    try:
        validate_core_config(raw_cfg, base_path=tmp_path)
        cfg = type(
            "ConfigStub",
            (),
            {
                "data": {
                    "csv_path": str(csv_file),
                    "date_column": "Date",
                    "frequency": "M",
                    "unexpected_key": "unused",
                },
                "portfolio": {"transaction_cost_bps": 0.0},
            },
        )()
        wrap_config_for_coverage(cfg, tracker)
        _ = cfg.data.get("csv_path")
        _ = cfg.portfolio.get("transaction_cost_bps")
        _ = cfg.data.get("unexpected_key")
    finally:
        deactivate_config_coverage()

    report = tracker.generate_report()
    assert "data.date_column" in report.unread_validated
    assert "data.unexpected_key" in report.unvalidated_reads
