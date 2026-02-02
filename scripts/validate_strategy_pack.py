"""Validate a strategy pack against the base configuration schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trend_analysis.monte_carlo.strategy.validation import validate_strategy_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Monte Carlo strategy pack against config defaults."
    )
    parser.add_argument("pack", type=Path, help="Path to the strategy pack YAML file")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/defaults.yml"),
        help="Path to the base config defaults",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_strategy_pack(args.pack, base_config_path=args.base_config)
    if errors:
        print("Strategy pack validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Strategy pack is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
