"""Structural ownership contracts for the canonical CLI front door."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src/trend/cli.py"
MC_VIZ_PATH = ROOT / "src/trend/mc/viz.py"
REPORT_PATH = ROOT / "src/trend/commands/report_export.py"
EXPLAIN_PATH = ROOT / "src/trend/commands/explain.py"
NL_PATH = ROOT / "src/trend/commands/nl.py"
RETIRED_OWNER_PATH = ROOT / "src/trend/cli_owned_commands.py"

MIGRATED_MC_VIZ_HANDLERS = {
    "_validate_mc_viz_output_flags",
    "_read_mc_frame",
    "_load_mc_bundle_frames",
    "_parse_mc_chart_selection",
    "_mc_chart_builders",
    "_export_mc_chart_artifacts",
    "_inject_mc_html_chart_markers",
    "_run_mc_viz_command",
}

REPORT_HANDLERS = {
    "_prepare_export_config",
    "_run_pipeline",
    "_handle_exports",
    "_write_trend_run_artifacts",
    "_write_report_files",
}

EXPLAIN_HANDLERS = {
    "_resolve_explain_details_path",
    "_build_explain_artifact_payload",
    "_build_result_chain",
    "_write_explain_artifacts",
}

NL_HANDLERS = {
    "_build_nl_chain",
    "_apply_nl_instruction",
    "_validate_nl_run_config",
}


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_cli_does_not_redefine_migrated_command_handlers() -> None:
    """The parser front door delegates MC visualization to its owning module."""

    cli_functions = _defined_functions(CLI_PATH)
    viz_functions = _defined_functions(MC_VIZ_PATH)

    assert "build_parser" in cli_functions
    assert "main" in cli_functions
    assert not (MIGRATED_MC_VIZ_HANDLERS & cli_functions)
    assert {
        "_read_mc_frame",
        "_load_mc_bundle_frames",
        "_parse_mc_chart_selection",
        "_mc_chart_builders",
        "_export_mc_chart_artifacts",
        "_inject_mc_html_chart_markers",
        "execute_mc_viz_cli",
    } <= viz_functions


def test_command_families_have_distinct_owners() -> None:
    """Report/export, explain, and NL implementations have focused owners."""

    cli_functions = _defined_functions(CLI_PATH)
    report_functions = _defined_functions(REPORT_PATH)
    explain_functions = _defined_functions(EXPLAIN_PATH)
    nl_functions = _defined_functions(NL_PATH)

    migrated = REPORT_HANDLERS | EXPLAIN_HANDLERS | NL_HANDLERS
    assert not (migrated & cli_functions)
    assert REPORT_HANDLERS <= report_functions
    assert EXPLAIN_HANDLERS <= explain_functions
    assert NL_HANDLERS <= nl_functions
    assert not (NL_HANDLERS & report_functions)
    assert not (REPORT_HANDLERS & explain_functions)
    assert not (REPORT_HANDLERS & nl_functions)


def test_catch_all_command_owner_is_retired() -> None:
    assert not RETIRED_OWNER_PATH.exists()
    cli_source = CLI_PATH.read_text(encoding="utf-8")
    assert "cli_owned_commands" not in cli_source
