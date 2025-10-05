"""Simplified command-line helpers built on the legacy ``trend`` CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[no-redef]

from trend import cli as trend_cli
from trend.reporting import generate_unified_report
from trend_analysis.config import load_config

DEFAULT_REPORT_FORMATS: tuple[str, ...] = tuple(trend_cli.DEFAULT_REPORT_FORMATS)


class TrendRunError(RuntimeError):
    """Raised when the headless runner encounters a recoverable error."""


def _load_configuration(path: str | Path) -> tuple[Path, Any]:
    """Load a YAML or TOML configuration file."""

    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    suffix = cfg_path.suffix.lower()
    if suffix in {".toml", ".tml"}:
        with cfg_path.open("rb") as handle:
            payload = tomllib.load(handle)
        if not isinstance(payload, dict):
            raise TrendRunError("TOML configuration must contain a top-level table")
        return cfg_path.resolve(), load_config(payload)

    return trend_cli._load_configuration(str(cfg_path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trend-run",
        description="Run the Trend Model pipeline and emit a unified report.",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination HTML report path (or directory)",
    )
    parser.add_argument(
        "--returns",
        help="Override the returns CSV defined in the configuration",
    )
    parser.add_argument(
        "--out",
        help="Directory where CSV/JSON/XLSX/TXT artefacts will be written",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=DEFAULT_REPORT_FORMATS,
        help="Subset of export formats when --out is provided",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate a PDF alongside the HTML report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the random seed for deterministic runs",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    """Execute the analysis pipeline using ``trend.cli`` helpers."""

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)

        cfg_path, cfg = _load_configuration(args.config)
        returns_path = trend_cli._resolve_returns_path(cfg_path, cfg, args.returns)
        returns_df = trend_cli._ensure_dataframe(returns_path)
        trend_cli._determine_seed(cfg, args.seed)

        export_dir: Path | None = None
        formats: tuple[str, ...] | None = None
        if args.out:
            export_dir = Path(args.out).expanduser().resolve()
            formats = (
                tuple(args.formats)
                if args.formats is not None
                else DEFAULT_REPORT_FORMATS
            )
        elif args.formats:
            raise TrendRunError("--formats requires --out to specify an export directory")

        if export_dir is not None:
            trend_cli._prepare_export_config(cfg, export_dir, formats)

        result, run_id, _ = trend_cli._run_pipeline(
            cfg,
            returns_df,
            source_path=returns_path,
            log_file=None,
            structured_log=False,
            bundle=None,
        )

        trend_cli._print_summary(cfg, result)

        if export_dir is not None:
            trend_cli._write_report_files(export_dir, cfg, result, run_id=run_id)

        report_path = trend_cli._resolve_report_output_path(args.output, export_dir, run_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            artefacts = generate_unified_report(
                result, cfg, run_id=run_id, include_pdf=args.pdf
            )
        except RuntimeError as exc:
            raise TrendRunError(str(exc)) from exc

        report_path.write_text(artefacts.html, encoding="utf-8")
        print(f"Report written: {report_path}")

        if args.pdf:
            if artefacts.pdf_bytes is None:
                raise TrendRunError(
                    "PDF generation failed – install the 'fpdf2' extra to enable --pdf output"
                )
            pdf_path = report_path.with_suffix(".pdf")
            pdf_path.write_bytes(artefacts.pdf_bytes)
            print(f"PDF report written: {pdf_path}")

        return 0
    except TrendRunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


__all__ = ["run"]

