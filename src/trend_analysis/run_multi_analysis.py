from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from typing import cast

from trend_analysis import export
from trend_analysis.config import load
from trend_analysis.multi_period import run as run_mp


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

    export_cfg = cfg.export
    out_dir = export_cfg.get("directory")
    out_formats = export_cfg.get("formats")
    filename = export_cfg.get("filename", "analysis")
    if not out_dir and not out_formats:
        out_dir = "outputs"  # pragma: no cover - defaults
        out_formats = ["excel"]
    if out_dir and out_formats:
        export.export_multi_period_metrics(
            results,
            str(Path(out_dir) / filename),
            formats=out_formats,
            include_metrics=True,
        )  # pragma: no cover - file I/O

    # Prepare export
    export_cfg = getattr(cfg, "export", {}) or {}
    out_dir = export_cfg.get("directory")
    formats = export_cfg.get("formats", [])
    filename = export_cfg.get("filename", "analysis")

    if not out_dir or not formats:
        return 0

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write each period result to CSV if requested
    for i, res in enumerate(results):
        df = pd.DataFrame([res])
        if any(fmt.lower() == "csv" for fmt in formats):
            df.to_csv(out_path / f"{filename}_{i}.csv", index=False)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
