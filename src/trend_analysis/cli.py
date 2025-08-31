from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from . import export, pipeline
from .api import run_simulation
from .data import load_csv
from .config import load_config
from .constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS


APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``trend-model`` command."""

    parser = argparse.ArgumentParser(prog="trend-model")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gui", help="Launch Streamlit interface")

    run_p = sub.add_parser("run", help="Run analysis pipeline")
    run_p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    run_p.add_argument("-i", "--input", required=True, help="Path to returns CSV")

    args = parser.parse_args(argv)

    if args.command == "gui":
        result = subprocess.run(["streamlit", "run", str(APP_PATH)])
        return result.returncode

    if args.command == "run":
        cfg = load_config(args.config)
        df = load_csv(args.input)
        split = cfg.sample_split
        required_keys = {"in_start", "in_end", "out_start", "out_end"}
        if required_keys.issubset(split):
            result = run_simulation(cfg, df)
            metrics_df = result.metrics
            res = result.details
        else:  # pragma: no cover - legacy fallback
            metrics_df = pipeline.run(cfg)
            res = pipeline.run_full(cfg)
        if not res:
            print("No results")
            return 0

        split = cfg.sample_split
        text = export.format_summary_text(
            res,
            str(split.get("in_start")),
            str(split.get("in_end")),
            str(split.get("out_start")),
            str(split.get("out_end")),
        )
        print(text)

        export_cfg = cfg.export
        out_dir = export_cfg.get("directory")
        out_formats = export_cfg.get("formats")
        filename = export_cfg.get("filename", "analysis")
        if not out_dir and not out_formats:
            out_dir = DEFAULT_OUTPUT_DIRECTORY
            out_formats = DEFAULT_OUTPUT_FORMATS
        if out_dir and out_formats:
            data = {"metrics": metrics_df}
            if any(f.lower() in {"excel", "xlsx"} for f in out_formats):
                sheet_formatter = export.make_summary_formatter(
                    res,
                    str(split.get("in_start")),
                    str(split.get("in_end")),
                    str(split.get("out_start")),
                    str(split.get("out_end")),
                )
                data["summary"] = pd.DataFrame()
                export.export_to_excel(
                    data,
                    str(Path(out_dir) / f"{filename}.xlsx"),
                    default_sheet_formatter=sheet_formatter,
                )
                other = [f for f in out_formats if f.lower() not in {"excel", "xlsx"}]
                if other:
                    export.export_data(
                        data, str(Path(out_dir) / filename), formats=other
                    )
            else:
                export.export_data(
                    data, str(Path(out_dir) / filename), formats=out_formats
                )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
