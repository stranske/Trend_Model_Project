from __future__ import annotations

import argparse

from trend_analysis.config import load
from trend_analysis import pipeline


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the trend analysis pipeline."""
    parser = argparse.ArgumentParser(prog="trend-analysis")
    parser.add_argument("-c", "--config", help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    result = pipeline.run(cfg)
    if result.empty:
        print("No results")
    else:
        # Show fund names (index) and column headers for clarity
        print(result.to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
