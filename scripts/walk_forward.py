"""CLI entry point for the walk-forward parameter sweep harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from trend_analysis.walk_forward import run_from_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward grid search")
    parser.add_argument(
        "--config",
        default="config/walk_forward.yml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = run_from_config(Path(args.config))
    print(f"Wrote walk-forward artifacts to {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
