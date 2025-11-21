from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from analysis.cv import export_report, walk_forward


def _load_returns(cfg: Mapping[str, Any], *, base_dir: Path) -> pd.DataFrame:
    data_path = cfg.get("csv_path")
    if not data_path:
        raise ValueError("data.csv_path must be provided in the config")
    csv_path = Path(data_path)
    if not csv_path.is_absolute():
        csv_path = (base_dir / csv_path).resolve()
    date_column = str(cfg.get("date_column", "Date"))
    columns = cfg.get("columns")

    frame = pd.read_csv(csv_path)
    if date_column not in frame.columns:
        raise ValueError(f"Date column '{date_column}' not found in {csv_path}")
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.sort_values(date_column)
    frame = frame.set_index(date_column)

    numeric = frame.select_dtypes(include=["number"]).astype(float)
    if columns:
        missing = [col for col in columns if col not in numeric.columns]
        if missing:
            raise ValueError(
                f"Missing columns in CSV: {', '.join(missing)} (from {csv_path})"
            )
        numeric = numeric.loc[:, list(columns)]
    if numeric.empty:
        raise ValueError("No numeric columns found in returns file")
    return numeric


def _load_cv_spec(cfg_path: Path) -> tuple[pd.DataFrame, Mapping[str, Any], int, bool, Path]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    data_cfg = raw.get("data", {})
    params = raw.get("params", {}) or {}
    cv_cfg = raw.get("cv", {}) or {}
    output_cfg = raw.get("output", {}) or {}

    folds = int(cv_cfg.get("folds", 3))
    expand = bool(cv_cfg.get("expand", True))
    output_dir = output_cfg.get("dir", "perf/cv")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (cfg_path.parent / output_path).resolve()

    data = _load_returns(data_cfg, base_dir=cfg_path.parent)
    return data, params, folds, expand, output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src", description="Utility CLI entrypoints")
    sub = parser.add_subparsers(dest="command", required=True)

    cv_p = sub.add_parser("cv", help="Run walk-forward cross-validation")
    cv_p.add_argument("--config", required=True, help="Path to YAML config file")
    cv_p.add_argument("--folds", type=int, help="Override folds from config")
    cv_p.add_argument(
        "--expand", dest="expand", action="store_true", help="Use expanding windows"
    )
    cv_p.add_argument(
        "--rolling", dest="expand", action="store_false", help="Use rolling windows"
    )
    cv_p.add_argument(
        "--output-dir", help="Directory for exported fold metrics and summary"
    )
    cv_p.set_defaults(expand=None)

    return parser


def _handle_cv(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    data, params, cfg_folds, cfg_expand, cfg_output = _load_cv_spec(cfg_path)
    folds = int(args.folds) if args.folds is not None else cfg_folds
    expand = cfg_expand if args.expand is None else bool(args.expand)
    output_dir = Path(args.output_dir) if args.output_dir else cfg_output
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    report = walk_forward(data, folds=folds, expand=expand, params=params)
    paths = export_report(report, output_dir)

    print(f"Folds written to: {paths['folds']}")
    print(f"Summary written to: {paths['summary']}")
    print(f"Markdown report: {paths['markdown']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "cv":
        return _handle_cv(args)
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
