"""Focused command handlers used by :mod:`trend.cli`.

The public CLI remains responsible for parsing and shared configuration setup.
This module owns the command-specific side effects so those contracts can be
tested through ``trend.cli:main`` without retaining command implementations in
the parser front door.
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

ErrorFactory = Callable[[str], Exception]


@dataclass(frozen=True)
class PreparedCommandInputs:
    """Shared config and data inputs prepared for run/report command handlers."""

    cfg_path: Path
    cfg: Any
    returns_path: Path
    returns_df: Any
    seed: int


def prepare_command_inputs(
    args: argparse.Namespace,
    *,
    load_configuration: Callable[[str], tuple[Path, Any]],
    prepare_config: Callable[[Any], None],
    ensure_run_spec: Callable[..., Any],
    resolve_returns_path: Callable[..., Path],
    ensure_dataframe: Callable[..., Any],
    determine_seed: Callable[[Any, int | None], int],
) -> PreparedCommandInputs:
    """Load the shared run/report inputs outside the parser front door."""

    cfg_path, cfg = load_configuration(args.config)
    prepare_config(cfg)
    ensure_run_spec(cfg, base_path=cfg_path.parent)
    returns_path = resolve_returns_path(cfg_path, cfg, getattr(args, "returns", None))
    try:
        inspect.signature(ensure_dataframe).bind(returns_path, config=cfg)
    except (TypeError, ValueError):
        # Keep injected path-only test/extension loaders compatible. The
        # canonical CLI loader accepts ``config`` and applies its ingestion
        # contract before the pipeline sees the frame.
        returns_df = ensure_dataframe(returns_path)
    else:
        returns_df = ensure_dataframe(returns_path, config=cfg)
    seed = determine_seed(cfg, getattr(args, "seed", None))
    return PreparedCommandInputs(
        cfg_path=cfg_path,
        cfg=cfg,
        returns_path=returns_path,
        returns_df=returns_df,
        seed=seed,
    )


def run_check_command(*, environment_check: Callable[[], int]) -> int:
    """Run the lightweight environment check command."""

    return environment_check()


def run_app_command(
    args: argparse.Namespace,
    extra_args: list[str],
    *,
    app_path: Path,
    run_process: Callable[..., Any] = subprocess.run,
) -> int:
    """Start Streamlit while preserving the CLI's exit-code contract."""

    del args  # The parser owns app-specific argument validation.
    try:
        proc = run_process(["streamlit", "run", str(app_path), *extra_args])
    except FileNotFoundError:
        print(
            "Error: the 'streamlit' executable was not found. " "Install the optional 'app' extra.",
            file=sys.stderr,
        )
        return 127
    return int(proc.returncode)


def run_analysis_command(
    args: argparse.Namespace,
    cfg_path: Path,
    cfg: Any,
    returns_path: Path,
    returns_df: Any,
    *,
    error: ErrorFactory,
    set_cache: Callable[[bool], None],
    get_spec_preset: Callable[[str], Any],
    list_spec_presets: Callable[[], Iterable[str]],
    apply_spec_preset: Callable[[Any, Any], None],
    get_portfolio_preset: Callable[[str], Any],
    list_portfolio_presets: Callable[[], Iterable[str]],
    apply_portfolio_preset: Callable[[Any, Any], None],
    load_universe: Callable[..., tuple[Any, Any]],
    apply_universe_mask: Callable[..., Any],
    attach_universe_paths: Callable[..., None],
    find_existing_run: Callable[..., Path | None],
    calculate_run_id: Callable[..., str],
    run_pipeline: Callable[..., tuple[Any, str, Path | None]],
    write_artifacts: Callable[..., Path | None],
    print_summary: Callable[[Any, Any], None],
    finish_structured_log: Callable[[bool, str, Path | None], None],
    finalize_coverage: Callable[[], None],
) -> int:
    """Execute ``trend run`` with all side-effecting dependencies explicit."""

    set_cache(not getattr(args, "no_cache", False))
    if getattr(args, "preset", None):
        try:
            spec_preset = get_spec_preset(args.preset)
        except KeyError:
            available = ", ".join(list_spec_presets())
            raise error(f"Unknown preset '{args.preset}'. Available presets: {available}")
        apply_spec_preset(cfg, spec_preset)
        try:
            portfolio_preset = get_portfolio_preset(args.preset)
        except KeyError:
            available = ", ".join(list_portfolio_presets())
            raise error(f"Unknown preset '{args.preset}'. Available: {available}")
        apply_portfolio_preset(cfg, portfolio_preset)
    if getattr(args, "universe", None):
        mask, universe_spec = load_universe(args.universe, prices=returns_df)
        returns_df = apply_universe_mask(returns_df, mask, date_column=universe_spec.date_column)
        attach_universe_paths(cfg, universe_spec, csv_path=str(returns_path))
    if getattr(args, "skip_if_exists", False):
        candidate_run_id = calculate_run_id(cfg, returns_path)
        existing_manifest = find_existing_run(cfg, candidate_run_id)
        if existing_manifest is not None:
            print(f"already-done: run_id={candidate_run_id}")
            print(f"Existing manifest: {existing_manifest}")
            finalize_coverage()
            return 0
    result, run_id, log_path = run_pipeline(
        cfg,
        returns_df,
        source_path=returns_path,
        log_file=Path(args.log_file) if args.log_file else None,
        structured_log=not args.no_structured_log,
        bundle=Path(args.bundle) if args.bundle else None,
    )
    write_artifacts(
        cfg=cfg,
        result=result,
        config_path=cfg_path,
        input_path=returns_path,
        data_frame=returns_df,
        run_id=run_id,
        structured_log=not args.no_structured_log,
    )
    print_summary(cfg, result)
    finish_structured_log(not args.no_structured_log, run_id, log_path)
    if log_path:
        print(f"Structured log: {log_path}")
    finalize_coverage()
    return 0


def run_report_command(
    args: argparse.Namespace,
    cfg: Any,
    returns_path: Path,
    returns_df: Any,
    *,
    error: ErrorFactory,
    default_formats: Iterable[str],
    prepare_export_config: Callable[..., None],
    run_pipeline: Callable[..., tuple[Any, str, Path | None]],
    print_summary: Callable[[Any, Any], None],
    write_report_files: Callable[..., None],
    resolve_report_output_path: Callable[..., Path],
    generate_report: Callable[..., Any],
    finalize_coverage: Callable[[], None],
) -> int:
    """Execute ``trend report`` while keeping pipeline and writers injectable."""

    export_dir = Path(args.out).resolve() if args.out else None
    if export_dir is None and not args.output:
        raise error(
            "The 'report' command requires --out for artefacts or --output for the HTML report"
        )
    formats = args.formats or default_formats
    prepare_export_config(cfg, export_dir, formats if export_dir is not None else None)
    result, run_id, _ = run_pipeline(
        cfg,
        returns_df,
        source_path=returns_path,
        log_file=None,
        structured_log=False,
        bundle=None,
    )
    print_summary(cfg, result)
    if export_dir is not None:
        write_report_files(export_dir, cfg, result, run_id=run_id)
    report_path = resolve_report_output_path(args.output, export_dir, run_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        artifacts = generate_report(
            result,
            cfg,
            run_id=run_id,
            include_pdf=args.pdf,
            spec=getattr(cfg, "_trend_run_spec", None),
        )
    except RuntimeError as exc:
        raise error(str(exc)) from exc
    report_path.write_text(artifacts.html, encoding="utf-8")
    print(f"Report written: {report_path}")
    if args.pdf:
        if artifacts.pdf_bytes is None:
            raise error(
                "PDF generation failed – install the 'fpdf2' dependency to enable --pdf output"
            )
        pdf_path = report_path.with_suffix(".pdf")
        pdf_path.write_bytes(artifacts.pdf_bytes)
        print(f"PDF report written: {pdf_path}")
    finalize_coverage()
    return 0
