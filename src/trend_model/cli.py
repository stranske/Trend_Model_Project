"""Console entry point for headless Trend Model runs."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, cast

from trend.cli import (
    DEFAULT_REPORT_FORMATS,
    TrendCLIError,
    _determine_seed,
    _ensure_dataframe,
)
from trend.cli import _load_configuration as _load_yaml_configuration
from trend.cli import (
    _prepare_export_config,
    _print_summary,
    _resolve_report_output_path,
    _resolve_returns_path,
    _run_pipeline,
    _write_report_files,
)
from trend.reporting import generate_unified_report
from trend_analysis.config import load_config
from trend_model.spec import ensure_run_spec

_toml_module: ModuleType | None
try:  # Python 3.11+
    _toml_module = import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    _toml_module = None

DEFAULT_OUTPUT_PATH = Path("reports") / "trend_report.html"
TOML_SUFFIXES = {".toml", ".tml"}

__all__ = ["build_parser", "run"]


@contextlib.contextmanager
def _temporary_cwd(directory: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_toml_payload(path: Path) -> Mapping[str, Any]:
    if _toml_module is not None:
        loader = _toml_module
    else:  # pragma: no cover - executed on Python < 3.11
        try:
            loader = import_module("tomli")
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
            raise TrendCLIError(
                "Reading TOML configs requires Python 3.11+ or installing the 'tomli' package."
            ) from exc
    with path.open("rb") as fh:
        data = loader.load(fh)
    if not isinstance(data, Mapping):
        raise TrendCLIError("TOML configuration must contain a top-level table")
    return data


def _load_configuration(path: str) -> tuple[Path, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if cfg_path.suffix.lower() in TOML_SUFFIXES:
        payload = _load_toml_payload(cfg_path)
        with _temporary_cwd(cfg_path.parent):
            cfg = load_config(payload)
            ensure_run_spec(cfg, base_path=cfg_path.parent)
        return cfg_path, cfg
    resolved_path, cfg_obj = cast(
        tuple[Path, Any], _load_yaml_configuration(str(cfg_path))
    )
    ensure_run_spec(cfg_obj, base_path=cfg_path.parent)
    return resolved_path, cfg_obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trend-run",
        description=(
            "Run the volatility-adjusted trend analysis pipeline using a configuration file "
            "and generate a standalone HTML report."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML or TOML configuration file",
    )
    parser.add_argument(
        "--returns",
        help="Optional override for the returns CSV defined in the configuration",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Destination for the HTML report (defaults to reports/trend_report.html; "
            "if a directory is provided the filename is generated automatically)"
        ),
    )
    parser.add_argument(
        "--artefacts",
        help="Directory where CSV/JSON/XLSX/TXT artefacts should be written",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=DEFAULT_REPORT_FORMATS,
        help="Subset of export formats to produce when --artefacts is supplied",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the random seed used during the simulation",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a PDF version of the report (requires the fpdf2 dependency)",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.formats and not args.artefacts:
        parser.error("The --formats option requires --artefacts to be specified")

    try:
        cfg_path, cfg = _load_configuration(args.config)
        returns_path = _resolve_returns_path(cfg_path, cfg, args.returns)
        returns_df = _ensure_dataframe(returns_path)
        _determine_seed(cfg, args.seed)

        export_dir = (
            Path(args.artefacts).expanduser().resolve() if args.artefacts else None
        )
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
        formats = tuple(args.formats) if args.formats else None
        _prepare_export_config(
            cfg, export_dir, formats if export_dir is not None else None
        )

        result, run_id, _ = _run_pipeline(
            cfg,
            returns_df,
            source_path=returns_path,
            log_file=None,
            structured_log=False,
            bundle=None,
        )
        _print_summary(cfg, result)

        if export_dir is not None:
            _write_report_files(export_dir, cfg, result, run_id=run_id)

        report_path = _resolve_report_output_path(args.output, export_dir, run_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        artefacts = generate_unified_report(
            result,
            cfg,
            run_id=run_id,
            include_pdf=args.pdf,
            spec=getattr(cfg, "_trend_run_spec", None),
        )
        report_path.write_text(artefacts.html, encoding="utf-8")
        print(f"Report written: {report_path}")
        if args.pdf:
            if artefacts.pdf_bytes is None:
                raise TrendCLIError(
                    "PDF generation failed – install the 'fpdf2' dependency to enable --pdf output"
                )
            pdf_path = report_path.with_suffix(".pdf")
            pdf_path.write_bytes(artefacts.pdf_bytes)
            print(f"PDF report written: {pdf_path}")
        return 0
    except TrendCLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(run())
