from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

from . import export
from .config import load
from .constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from .logging_setup import setup_logging
from .multi_period import run as run_mp


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for multi-period analysis."""
    parser = argparse.ArgumentParser(prog="trend-analysis-multi")
    parser.add_argument("-c", "--config", help="Path to YAML config")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print per-period summary tables",
    )
    args = parser.parse_args(argv)

    log_path = setup_logging(app_name="run_multi_analysis")
    logging.getLogger(__name__).info("Log file initialised at %s", log_path)

    cfg = load(args.config)
    results = run_mp(cfg)
    if not results:
        print("No results")  # pragma: no cover - trivial branch
        return 0

    if args.detailed:
        for res in results:  # pragma: no cover - human output
            period = cast(
                tuple[str, str, str, str], res.get("period", ("", "", "", ""))
            )
            text = export.format_summary_text(
                res,
                period[0],
                period[1],
                period[2],
                period[3],
            )
            print(text)
            print()

        summary = export.combined_summary_result(results)
        first_period = cast(
            tuple[str, str, str, str], results[0].get("period", ("", "", "", ""))
        )
        last_period = cast(
            tuple[str, str, str, str], results[-1].get("period", ("", "", "", ""))
        )
        sum_text = export.format_summary_text(
            summary,
            first_period[0],
            first_period[1],
            last_period[2],
            last_period[3],
        )
        print("Combined Summary")
        print(sum_text)
    export_cfg = cfg.export
    out_dir = export_cfg.get("directory")
    out_formats = export_cfg.get("formats")
    filename = export_cfg.get("filename", "analysis")
    if not out_dir and not out_formats:
        out_dir = DEFAULT_OUTPUT_DIRECTORY  # pragma: no cover - defaults
        out_formats = DEFAULT_OUTPUT_FORMATS
    if out_dir and out_formats:
        export.export_phase1_multi_metrics(
            results,
            str(Path(out_dir) / filename),
            formats=out_formats,
            include_metrics=True,
        )  # pragma: no cover - file I/O
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
