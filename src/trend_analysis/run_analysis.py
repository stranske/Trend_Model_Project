from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, cast

import pandas as pd

from . import api, export
from .config import load
from .constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from .data import load_csv


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

    # Load CSV data from config
    csv_path = cfg.data.get("csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    csv_path = str(csv_path)

    data_settings = getattr(cfg, "data", {}) or {}
    missing_policy_cfg = data_settings.get("missing_policy")
    if missing_policy_cfg is None:
        missing_policy_cfg = data_settings.get("nan_policy")
    missing_limit_cfg = data_settings.get("missing_limit")
    if missing_limit_cfg is None:
        missing_limit_cfg = data_settings.get("nan_limit")

    load_csv_signature = inspect.signature(load_csv)
    load_csv_params = load_csv_signature.parameters

    load_kwargs: dict[str, Any] = {}
    if "errors" in load_csv_params:
        load_kwargs["errors"] = "raise"
    if missing_policy_cfg is not None:
        if "missing_policy" in load_csv_params:
            load_kwargs["missing_policy"] = missing_policy_cfg
        elif "nan_policy" in load_csv_params:
            load_kwargs["nan_policy"] = missing_policy_cfg
    if missing_limit_cfg is not None:
        if "missing_limit" in load_csv_params:
            load_kwargs["missing_limit"] = missing_limit_cfg
        elif "nan_limit" in load_csv_params:
            load_kwargs["nan_limit"] = missing_limit_cfg

    df = load_csv(csv_path, **cast(Any, load_kwargs))

    if df is None:
        raise FileNotFoundError(csv_path)

    # Use unified API instead of direct pipeline calls
    result = api.run_simulation(cfg, df)

    if args.detailed:
        if result.metrics.empty:
            print("No results")  # pragma: no cover - trivial branch
        else:
            print(result.metrics.to_string())  # pragma: no cover - human output
    else:
        if not result.details:
            print("No results")  # pragma: no cover - trivial branch
        else:
            split = cfg.sample_split
            text = export.format_summary_text(
                result.details,
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
                out_dir = DEFAULT_OUTPUT_DIRECTORY  # pragma: no cover - defaults
                out_formats = DEFAULT_OUTPUT_FORMATS
            if out_dir and out_formats:  # pragma: no cover - file output
                data = {"metrics": result.metrics}
                regime_table = result.details.get("performance_by_regime")
                if isinstance(regime_table, pd.DataFrame) and not regime_table.empty:
                    data["performance_by_regime"] = regime_table
                regime_notes = result.details.get("regime_notes")
                if regime_notes:
                    data["regime_notes"] = pd.DataFrame({"note": list(regime_notes)})
                if any(
                    f.lower() in {"excel", "xlsx"} for f in out_formats
                ):  # pragma: no cover - file I/O
                    sheet_formatter = export.make_summary_formatter(
                        result.details,
                        cast(str, split.get("in_start")),
                        cast(str, split.get("in_end")),
                        cast(str, split.get("out_start")),
                        cast(str, split.get("out_end")),
                    )
                    data["summary"] = export.summary_frame_from_result(result.details)
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
