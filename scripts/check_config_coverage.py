"""Run a config coverage pass against a config file and enforce a threshold."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trend.cli import _ensure_dataframe, _resolve_returns_path
from trend.config_schema import load_core_config
from trend_analysis.api import run_simulation
from trend_analysis.config.coverage import (
    ConfigCoverageReport,
    ConfigCoverageTracker,
    activate_config_coverage,
    compute_schema_validity,
    deactivate_config_coverage,
    wrap_config_for_coverage,
)
from trend_analysis.config.models import load_config
from trend_model.spec import ensure_run_spec


def _run_coverage(
    config_path: Path,
) -> tuple[ConfigCoverageTracker, ConfigCoverageReport]:
    tracker = ConfigCoverageTracker()
    activate_config_coverage(tracker)
    try:
        load_core_config(config_path)
        cfg = load_config(config_path)
        wrap_config_for_coverage(cfg, tracker)
        ensure_run_spec(cfg, base_path=config_path.parent)
        returns_path = _resolve_returns_path(config_path, cfg, None)
        returns_df = _ensure_dataframe(returns_path)
        run_simulation(cfg, returns_df)
        report = tracker.generate_report()
        return tracker, report
    finally:
        deactivate_config_coverage()


def _normalize_threshold(value: float) -> float:
    if value <= 0:
        return 0.0
    if value > 1.0:
        return value / 100.0
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run config coverage on a config file and enforce a threshold."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/demo.yml"),
        help="Path to the config file to execute.",
    )
    parser.add_argument(
        "--ignored-threshold",
        type=int,
        default=0,
        help="Max allowed ignored keys before failing.",
    )
    parser.add_argument(
        "--min-validity",
        type=float,
        default=0.0,
        help="Minimum schema validity ratio (0-1) or percent (0-100) before failing.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Use CI defaults (min validity 95% unless overridden).",
    )
    args = parser.parse_args(argv)
    if args.ci and args.min_validity == parser.get_default("min_validity"):
        args.min_validity = 0.95

    config_path = args.config.expanduser().resolve()
    tracker, report = _run_coverage(config_path)
    print(tracker.format_report(report))
    validity = compute_schema_validity(report)
    print(f"Schema validity: {validity * 100:.1f}%")
    ignored_count = len(report.ignored)
    print(f"Ignored keys: {ignored_count}")
    if ignored_count > args.ignored_threshold:
        print(f"FAIL: ignored keys {ignored_count} exceeds threshold " f"{args.ignored_threshold}.")
        return 1
    min_validity = _normalize_threshold(args.min_validity)
    if min_validity and validity < min_validity:
        print(
            f"FAIL: schema validity {validity * 100:.1f}% below "
            f"minimum {min_validity * 100:.1f}%."
        )
        return 1
    print("OK: ignored keys within threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
