"""Validate a config file against the generated JSON schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from trend_analysis.config.schema_validation import validate_config_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a config YAML against config.schema.json")
    parser.add_argument("config", type=Path, help="Path to the config YAML file")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("config.schema.json"),
        help="Path to the JSON schema file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_config_file(args.config, args.schema)
    if errors:
        print("Config validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Config is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
