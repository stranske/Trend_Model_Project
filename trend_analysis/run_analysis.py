from __future__ import annotations

import argparse

from trend_analysis.config import load
from trend_analysis import pipeline, export


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the trend analysis pipeline."""
    parser = argparse.ArgumentParser(prog="trend-analysis")
    parser.add_argument("-c", "--config", help="Path to YAML config")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print comprehensive result dictionary",
    )
    args = parser.parse_args(argv)

    cfg = load(args.config)
    if args.detailed:
        result = pipeline.run_full(cfg)
        print(result if result else "No results")
    else:
        res = pipeline.run_full(cfg)
        if not res:
            print("No results")
        else:
            split = cfg.sample_split
            text = export.format_summary_text(
                res,
                split.get("in_start"),
                split.get("in_end"),
                split.get("out_start"),
                split.get("out_end"),
            )
            print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
