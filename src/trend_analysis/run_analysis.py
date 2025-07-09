from __future__ import annotations

import argparse

from pathlib import Path

from typing import cast

from trend_analysis.config import load
from trend_analysis import pipeline, export
import pandas as pd


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
    metrics_df = pipeline.run(cfg)
    if args.detailed:
        if metrics_df.empty:
            print("No results")  # pragma: no cover - trivial branch
        else:
            print(metrics_df.to_string())  # pragma: no cover - human output
    else:
        res = pipeline.run_full(cfg)
        if not res:
            print("No results")  # pragma: no cover - trivial branch
        else:
            split = cfg.sample_split
            text = export.format_summary_text(
                res,
                cast(str, split.get("in_start")),
                cast(str, split.get("in_end")),
                cast(str, split.get("out_start")),
                cast(str, split.get("out_end")),
            )
            print(text)  # pragma: no cover - human output
            export_cfg = cfg.export
            out_dir = export_cfg.get("directory")
            out_formats = export_cfg.get("formats")
            filename = export_cfg.get("filename", "analysis")
            if not out_dir and not out_formats:
                out_dir = "outputs"  # pragma: no cover - defaults
                out_formats = ["excel"]
            if out_dir and out_formats:  # pragma: no cover - file output
                data = {"metrics": metrics_df}
                if any(
                    f.lower() in {"excel", "xlsx"} for f in out_formats
                ):  # pragma: no cover - file I/O
                    sheet_formatter = export.make_summary_formatter(
                        res,
                        cast(str, split.get("in_start")),
                        cast(str, split.get("in_end")),
                        cast(str, split.get("out_start")),
                        cast(str, split.get("out_end")),
                    )
                    data["summary"] = pd.DataFrame()
                    export.export_to_excel(
                        data,
                        str(Path(out_dir) / f"{filename}.xlsx"),
                        default_sheet_formatter=sheet_formatter,
                    )
                    other = [
                        f for f in out_formats if f.lower() not in {"excel", "xlsx"}
                    ]
                    if other:
                        export.export_data(
                            data, str(Path(out_dir) / filename), formats=other
                        )  # pragma: no cover - file I/O
                else:
                    export.export_data(
                        data, str(Path(out_dir) / filename), formats=out_formats
                    )  # pragma: no cover - file I/O
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
