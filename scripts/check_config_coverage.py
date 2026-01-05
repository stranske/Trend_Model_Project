"""Run a config coverage pass against a config file and enforce a threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

from trend.cli import _ensure_dataframe, _resolve_returns_path
from trend.config_schema import load_core_config
from trend_analysis.api import run_simulation
from trend_analysis.config.coverage import (
    ConfigCoverageReport,
    ConfigCoverageTracker,
    activate_config_coverage,
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


def main() -> int:
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
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    tracker, report = _run_coverage(config_path)
    print(tracker.format_report(report))
    ignored_count = len(report.ignored)
    print(f"Ignored keys: {ignored_count}")
    if ignored_count > args.ignored_threshold:
        print(f"FAIL: ignored keys {ignored_count} exceeds threshold " f"{args.ignored_threshold}.")
        return 1
    print("OK: ignored keys within threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
