from __future__ import annotations

import argparse

from pathlib import Path

from typing import cast

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
                cast(str, split.get("in_start")),
                cast(str, split.get("in_end")),
                cast(str, split.get("out_start")),
                cast(str, split.get("out_end")),
            )
            print(text)
            export_cfg = cfg.export
            out_dir = export_cfg.get("directory")
            out_formats = export_cfg.get("formats")
            if out_dir and out_formats:
                data = {
                    "in_sample": res["in_sample_scaled"],
                    "out_sample": res["out_sample_scaled"],
                }
                prefix = Path(out_dir) / "analysis"
                export.export_data(data, str(prefix), formats=out_formats)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
