"""Structural ownership contracts for the canonical CLI front door."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src/trend/cli.py"
MC_VIZ_PATH = ROOT / "src/trend/mc/viz.py"
OWNED_COMMANDS_PATH = ROOT / "src/trend/cli_owned_commands.py"

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

MIGRATED_REPORT_AND_NL_HANDLERS = {
    "_prepare_export_config",
    "_run_pipeline",
    "_handle_exports",
    "_write_trend_run_artifacts",
    "_write_report_files",
    "_resolve_explain_details_path",
    "_build_explain_artifact_payload",
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


def test_cli_delegates_report_and_nl_implementations_to_owned_module() -> None:
    """Report/export and explain/NL implementations have one supported owner."""

    cli_functions = _defined_functions(CLI_PATH)
    owned_functions = _defined_functions(OWNED_COMMANDS_PATH)

    assert not (MIGRATED_REPORT_AND_NL_HANDLERS & cli_functions)
    assert MIGRATED_REPORT_AND_NL_HANDLERS <= owned_functions
