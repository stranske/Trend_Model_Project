"""Final regression gates for retired runtime surfaces and supported commands."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from IPython.core.inputtransformer2 import TransformerManager

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOTS = (
    REPO_ROOT / "archives",
    REPO_ROOT / "docs" / "archive",
    REPO_ROOT / "docs" / "keepalive",
    REPO_ROOT / "notebooks" / "old",
)
TEXT_SUFFIXES = {
    ".cfg",
    ".cjs",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
ROOT_EXTENSIONLESS_RUNTIME_FILES = {"Dockerfile", "Makefile"}
# This is preserved review evidence, not an operator, build, or runtime surface.
ROOT_HISTORY_TEXT_FILES = {REPO_ROOT / "review-suggested-issues.md"}
ROOT_RUNTIME_TEXT_FILES = tuple(
    path
    for path in REPO_ROOT.iterdir()
    if path.is_file()
    and path not in ROOT_HISTORY_TEXT_FILES
    and (path.suffix.lower() in TEXT_SUFFIXES or path.name in ROOT_EXTENSIONLESS_RUNTIME_FILES)
)
RUNTIME_TEXT_ROOTS = (
    REPO_ROOT / "analysis",
    REPO_ROOT / "design-system",
    REPO_ROOT / "src",
    REPO_ROOT / "streamlit_app",
    REPO_ROOT / "scripts",
    REPO_ROOT / "examples",
    REPO_ROOT / "notebooks",
    REPO_ROOT / "tests",
    REPO_ROOT / "docs",
    REPO_ROOT / ".github" / "workflows",
    REPO_ROOT / ".github" / "actions",
    REPO_ROOT / ".github" / "scripts",
    REPO_ROOT / "tools",
    *ROOT_RUNTIME_TEXT_FILES,
)
FORBIDDEN_RUNTIME_IMPORTS = (
    "trend." + "compat_entrypoints",
    "trend_analysis." + "cli",
    "trend_analysis." + "run_analysis",
    "trend_analysis." + "run_multi_analysis",
)
FORBIDDEN_RUNTIME_REFERENCES = FORBIDDEN_RUNTIME_IMPORTS + (
    "trend_analysis/" + "cli.py",
    "trend_analysis/" + "run_analysis.py",
    "trend_analysis/" + "run_multi_analysis.py",
)
FORBIDDEN_RUNTIME_SYMBOLS = (
    "load_market_data_" + "csv",
    "load_market_data_" + "parquet",
)
REMOVED_TEST_ONLY_SYMBOLS = {
    "src/trend/cli.py": {
        "_write_" + "mc_manifest",
        "_is_valid_" + "tqdm_instance",
    },
    "src/trend_analysis/metrics/__init__.py": {
        "annualize_" + "return",
        "annualize_" + "volatility",
        "annualize_" + "sharpe_ratio",
        "annualize_" + "sortino_ratio",
        "info_" + "ratio",
    },
    "src/trend_analysis/core/rank_selection.py": {
        "as_" + "frame",
        "reset_" + "selector_cache",
        "selector_cache_" + "hits",
        "selector_cache_" + "misses",
        "_call_metric_" + "series",
        "_metric_fn_" + "accepts_risk_free_override",
    },
    "src/trend_analysis/monte_carlo/runner.py": {"_inject_" + "cash_returns"},
}
REMOVED_GUI_SYMBOLS = {
    "src/trend_analysis/gui/app.py": {"_normalize_gui_" + "store_cfg"},
    "streamlit_app/components/analysis_runner.py": {"Model" + "Settings"},
}
REMOVED_SCHEMA_READ_KEYS = {
    "src/trend_analysis/monte_carlo/costs.py": {
        "regimes",
        "default",
        "distribution",
        "dist",
        "bps",
        "mu",
    },
    "src/trend_analysis/monte_carlo/aggregator.py": {"fold"},
    "src/trend_analysis/monte_carlo/export.py": {"fold"},
    "src/trend_analysis/walk_forward.py": {"fold"},
}
REMOVED_PATHS = (
    "src/trend/compat_entrypoints.py",
    "src/trend_analysis/" + "cli.py",
    "src/trend_analysis/" + "run_analysis.py",
    "src/trend_analysis/" + "run_multi_analysis.py",
    "src/trend_model",
    "src/trend_portfolio_app",
    "retired/trend_portfolio_app",
    "retired/tests",
    "examples/legacy_streamlit_app",
    "scripts/trend-model",
)
NOTEBOOK_TRANSFORMER = TransformerManager()


def _textual_import_modules(text: str) -> list[str]:
    """Recover import targets from executable non-Python wrapper syntax."""

    modules = []
    for match in re.finditer(r"\bimport\s+([^\n;]+)", text):
        tail = match.group(1)
        for part in tail.split(","):
            name = part.strip().split(maxsplit=1)[0] if part.strip() else ""
            if name and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name):
                modules.append(name)
    from_pattern = re.compile(
        r"\bfrom\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s+(\([^)]*\)|[^\n;]+)",
        re.DOTALL,
    )
    for match in from_pattern.finditer(text):
        parent = match.group(1)
        imported = match.group(2).strip().strip("()")
        for item in imported.split(","):
            name = item.strip().split(maxsplit=1)[0] if item.strip() else ""
            if name and re.fullmatch(r"[A-Za-z_]\w*", name):
                modules.append(f"{parent}.{name}")
    return modules


def _is_archived(path: Path) -> bool:
    return any(path.is_relative_to(root) for root in ARCHIVE_ROOTS)


def _static_string(node: ast.AST) -> str | None:
    """Return a statically composed string literal, if one is recoverable."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _mapping_key_write_offenders(path: Path, retired_keys: set[str]) -> set[str]:
    """Find direct subscript assignments that restore retired mapping keys."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                key = _static_string(target.slice)
                if key in retired_keys:
                    offenders.add(key)
    return offenders


def _include_text_file(path: Path) -> bool:
    if (
        not path.is_file()
        or "__pycache__" in path.parts
        or "node_modules" in path.parts
        or _is_archived(path)
    ):
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return (path.suffix == "" and os.access(path, os.X_OK)) or (
        path.parent == REPO_ROOT and path.name in ROOT_EXTENSIONLESS_RUNTIME_FILES
    )


def _text_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if _include_text_file(root) else []
    return [path for path in root.rglob("*") if _include_text_file(path)]


def _defined_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _removed_symbol_offenders(path: Path, removed_names: set[str]) -> set[str]:
    """Find direct definitions and names served through module ``__getattr__``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = _defined_symbols(path) & removed_names
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "__getattr__":
            continue
        dynamic_names = {
            value for child in ast.walk(node) if (value := _static_string(child)) is not None
        }
        offenders.update(dynamic_names & removed_names)
    return offenders


def _mapping_key_reads(path: Path) -> set[str]:
    """Return statically named mapping keys read with subscripts or ``get``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys: set[str] = set()
    for node in ast.walk(tree):
        key: str | None = None
        if isinstance(node, ast.Subscript):
            key = _static_string(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            key = _static_string(node.args[0])
        if key is not None:
            keys.add(key)
    return keys


def _forbidden_import_offenders(path: Path, text: str) -> list[str]:
    is_extensionless_launcher = path.suffix == "" and os.access(path, os.X_OK)
    if path.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(text)
            code_units = []
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                code_units.append(NOTEBOOK_TRANSFORMER.transform_cell(source))
                # IPython wraps a ``%%`` cell-magic body in a string passed to
                # ``run_cell_magic``. Parse that body too, otherwise executable
                # imports inside it are invisible to the AST scan.
                lines = source.splitlines(keepends=True)
                first_content = next(
                    (index for index, line in enumerate(lines) if line.strip()),
                    None,
                )
                if first_content is not None and lines[first_content].lstrip().startswith("%%"):
                    raw_body = "".join(lines[first_content + 1 :])
                    code_units.append(NOTEBOOK_TRANSFORMER.transform_cell(raw_body))
        except (AttributeError, TypeError, ValueError):
            return []
    elif path.suffix.lower() != ".py" and not is_extensionless_launcher:
        return []
    else:
        code_units = [text]
    modules: list[str] = []
    for code_unit in code_units:
        try:
            tree = ast.parse(code_unit, filename=str(path))
        except SyntaxError:
            # Non-Python cell magics can wrap executable Python commands or
            # heredocs. Recover import forms even when their wrapper is not a
            # valid Python AST, then continue with the remaining cells.
            modules.extend(_textual_import_modules(code_unit))
            continue
        modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        modules.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        modules.extend(
            f"{node.module}.{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
            for alias in node.names
        )
    try:
        display_path = path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    return [
        f"{display_path}: {forbidden}"
        for forbidden in FORBIDDEN_RUNTIME_IMPORTS
        if any(name == forbidden or name.startswith(f"{forbidden}.") for name in modules)
    ]


def test_removed_runtime_surfaces_do_not_return() -> None:
    """Retired packages, apps, and scripts must stay absent from the checkout."""

    returned = [path for path in REMOVED_PATHS if (REPO_ROOT / path).exists()]
    assert not returned, "Retired runtime surfaces returned:\n" + "\n".join(returned)


def test_active_runtime_and_docs_do_not_reference_removed_modules() -> None:
    """Only archived history may name the retired module entry points."""

    offenders: list[str] = []
    for root in RUNTIME_TEXT_ROOTS:
        for path in _text_files(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for reference in FORBIDDEN_RUNTIME_REFERENCES:
                if reference in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {reference}")
            offenders.extend(_forbidden_import_offenders(path, text))

    assert not offenders, "Active surfaces reference retired modules:\n" + "\n".join(offenders)


def test_active_runtime_does_not_restore_removed_data_loaders() -> None:
    """Canonical data loading must not be shadowed by compatibility shims."""

    offenders: list[str] = []
    for root in RUNTIME_TEXT_ROOTS:
        for path in _text_files(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for symbol in FORBIDDEN_RUNTIME_SYMBOLS:
                if symbol in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {symbol}")

    assert not offenders, "Active surfaces restore removed data loaders:\n" + "\n".join(offenders)


def test_test_only_runtime_seams_remain_absent() -> None:
    """Tests must use canonical APIs without restoring production compatibility hooks."""

    offenders: list[str] = []
    for relative_path, removed_names in REMOVED_TEST_ONLY_SYMBOLS.items():
        returned = sorted(_removed_symbol_offenders(REPO_ROOT / relative_path, removed_names))
        offenders.extend(f"{relative_path}: {name}" for name in returned)

    assert not offenders, "Test-only runtime seams returned:\n" + "\n".join(offenders)


def test_test_only_runtime_seam_gate_detects_deliberate_restoration(tmp_path: Path) -> None:
    """Deliberate-break proof exercises the same offender logic as the real gate."""

    candidate = tmp_path / "runner.py"
    candidate.write_text(
        "class Runner:\n"
        "    def _inject_cash_returns(self, returns):\n"
        "        return returns\n"
        "\n"
        "def __getattr__(name):\n"
        "    if name == 'selector_cache_' + 'hits':\n"
        "        return 0\n"
        "    raise AttributeError(name)\n",
        encoding="utf-8",
    )

    removed = {"_inject_" + "cash_returns", "selector_cache_" + "hits"}
    assert _removed_symbol_offenders(candidate, removed) == removed


def test_gui_compatibility_adapters_remain_absent() -> None:
    """GUI state must use the canonical Config and current model-state contracts."""

    offenders: list[str] = []
    for relative_path, removed_names in REMOVED_GUI_SYMBOLS.items():
        returned = sorted(_removed_symbol_offenders(REPO_ROOT / relative_path, removed_names))
        offenders.extend(f"{relative_path}: {name}" for name in returned)

    demo_runner = REPO_ROOT / "streamlit_app/components/demo_runner.py"
    dead_state_key = "model_" + "settings"
    if dead_state_key in demo_runner.read_text(encoding="utf-8"):
        offenders.append(f"streamlit_app/components/demo_runner.py: {dead_state_key}")

    retired_keys = {"mode", "rank", "use_" + "ranking", "use_vol_" + "adjust"}
    for key in sorted(
        _mapping_key_write_offenders(REPO_ROOT / "src/trend_analysis/gui/app.py", retired_keys)
    ):
        offenders.append(f"src/trend_analysis/gui/app.py: writes retired key {key}")

    assert not offenders, "GUI compatibility adapters returned:\n" + "\n".join(offenders)


def test_gui_compatibility_gate_detects_deliberate_restoration(tmp_path: Path) -> None:
    """Deliberate-break proof: a restored translator is detected by the AST scan."""

    candidate = tmp_path / "gui.py"
    candidate.write_text(
        "def _normalize_gui_store_cfg(cfg):\n    return cfg\n",
        encoding="utf-8",
    )

    removed = {"_normalize_gui_" + "store_cfg"}
    assert _removed_symbol_offenders(candidate, removed) == removed

    candidate.write_text("store.cfg['mo' + 'de'] = 'rank'\n", encoding="utf-8")
    assert _mapping_key_write_offenders(candidate, {"mode"}) == {"mode"}


def test_removed_cost_and_fold_schema_reads_remain_absent() -> None:
    """Legacy input keys may be rejected, but never read or translated."""

    offenders: list[str] = []
    for relative_path, removed_keys in REMOVED_SCHEMA_READ_KEYS.items():
        returned = sorted(_mapping_key_reads(REPO_ROOT / relative_path) & removed_keys)
        offenders.extend(f"{relative_path}: {key}" for key in returned)

    assert not offenders, "Legacy cost/fold schema reads returned:\n" + "\n".join(offenders)


def test_removed_schema_read_gate_detects_deliberate_translation(tmp_path: Path) -> None:
    """Deliberate-break proof exercises the production mapping-read detector."""

    candidate = tmp_path / "compat.py"
    candidate.write_text(
        "def translate(cfg, record):\n" "    return cfg.get('di' + 'st'), record['fo' + 'ld']\n",
        encoding="utf-8",
    )

    assert _mapping_key_reads(candidate) & {"dist", "fold"} == {"dist", "fold"}


def test_import_from_detection_keeps_retired_modules_absent(tmp_path: Path) -> None:
    """Multiline ``from package import name`` forms must remain forbidden."""

    candidate = tmp_path / "retired_import.py"
    candidate.write_text(
        "from trend_analysis import (\n    cli,\n)\n",
        encoding="utf-8",
    )

    offenders = _forbidden_import_offenders(candidate, candidate.read_text(encoding="utf-8"))

    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_legacy_runtime_shims_remain_absent() -> None:
    """Retired test-only shims must not return to production modules."""

    pipeline_source = (REPO_ROOT / "src/trend_analysis/pipeline.py").read_text(encoding="utf-8")
    config_models_source = (REPO_ROOT / "src/trend_analysis/config/models.py").read_text(
        encoding="utf-8"
    )
    io_validators_source = (REPO_ROOT / "src/trend_analysis/io/validators.py").read_text(
        encoding="utf-8"
    )

    assert "_DEFAULT_RUN_ANALYSIS" not in pipeline_source
    assert "Backward-compatible wrapper returning raw payloads for tests" not in pipeline_source
    assert "_TREND_CONFIG_CLASS" not in config_models_source
    assert "class ValidationResult" not in io_validators_source


def test_legacy_runtime_shims_absence_gate_rejects_patch_hook_restoration() -> None:
    """Deliberate-break gate: restoring the patch hook must fail this scan."""

    forbidden_hook = "_DEFAULT_RUN_ANALYSIS"
    assert forbidden_hook not in (REPO_ROOT / "src/trend_analysis/pipeline.py").read_text(
        encoding="utf-8"
    )


def test_pipeline_private_run_facade_is_absent() -> None:
    """The public pipeline module must not restore its test-only raw facade."""

    pipeline_path = REPO_ROOT / "src/trend_analysis/pipeline.py"
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    definitions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_analysis" not in definitions

    offenders: list[str] = []
    for root_name in ("src", "tests"):
        for source_path in (REPO_ROOT / root_name).rglob("*.py"):
            source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(source_tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "trend_analysis.pipeline"
                    and any(alias.name == "_run_analysis" for alias in node.names)
                ):
                    offenders.append(source_path.relative_to(REPO_ROOT).as_posix())
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "_run_analysis"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "pipeline"
                ):
                    offenders.append(source_path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []


@pytest.mark.parametrize("directory", ["scripts", "tools"])
def test_extensionless_launchers_remain_in_text_scan(tmp_path: Path, directory: str) -> None:
    launchers = tmp_path / directory
    launchers.mkdir()
    launcher = launchers / "trend"
    launcher.write_text(
        "#!/usr/bin/env python\nfrom trend_analysis import cli\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    assert launcher in _text_files(tmp_path)
    offenders = _forbidden_import_offenders(launcher, launcher.read_text(encoding="utf-8"))
    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_workflow_and_tooling_entry_points_remain_in_text_scan(tmp_path: Path) -> None:
    """Automation and repository tools are active runtime entry points too."""

    workflow = tmp_path / ".github" / "workflows" / "check.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("run: python -m trend.cli check\n", encoding="utf-8")
    tool = tmp_path / "tools" / "coverage_guard.py"
    tool.parent.mkdir()
    tool.write_text("from trend import cli\n", encoding="utf-8")
    action = tmp_path / ".github" / "actions" / "path-classifier" / "action.yml"
    action.parent.mkdir(parents=True)
    action.write_text("runs:\n  using: composite\n", encoding="utf-8")
    helper = tmp_path / ".github" / "scripts" / "issue_format.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("from trend import cli\n", encoding="utf-8")
    action_helper = tmp_path / ".github" / "actions" / "path-classifier" / "classify.js"
    action_helper.write_text("const { execFile } = require('child_process');\n", encoding="utf-8")

    assert workflow in _text_files(tmp_path / ".github" / "workflows")
    assert tool in _text_files(tmp_path / "tools")
    assert action in _text_files(tmp_path / ".github" / "actions")
    assert helper in _text_files(tmp_path / ".github" / "scripts")
    assert action_helper in _text_files(tmp_path / ".github" / "actions")


def test_root_launchers_and_active_docs_remain_in_text_scan() -> None:
    """Root launch surfaces and active Markdown must not fall outside the gate."""

    root_runtime_files = set(ROOT_RUNTIME_TEXT_FILES)
    expected_launchers = {
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / "docker-compose.yml",
        REPO_ROOT / "Makefile",
    }
    expected_active_docs = set(REPO_ROOT.glob("*.md")) - ROOT_HISTORY_TEXT_FILES

    assert expected_launchers <= root_runtime_files
    assert expected_active_docs <= root_runtime_files


def test_active_notebooks_remain_in_text_scan() -> None:
    """Executable notebooks are active surfaces unless they live below ``old/``."""

    active_notebooks = {
        path for path in (REPO_ROOT / "notebooks").rglob("*.ipynb") if not _is_archived(path)
    }
    scanned_notebooks = set(_text_files(REPO_ROOT / "notebooks"))

    assert active_notebooks
    assert active_notebooks <= scanned_notebooks
    assert not any(_is_archived(path) for path in scanned_notebooks)


def test_notebook_code_cells_detect_retired_imports(tmp_path: Path) -> None:
    """Executable notebook cells must receive the same import gate as Python."""
    notebook = tmp_path / "active.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["from trend_analysis import cli\n"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    offenders = _forbidden_import_offenders(notebook, notebook.read_text(encoding="utf-8"))

    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_notebook_magic_line_cannot_hide_later_import_in_same_cell(
    tmp_path: Path,
) -> None:
    """IPython-only lines must not suppress later Python in the same cell."""
    notebook = tmp_path / "active.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "%matplotlib inline\n",
                            "from trend_analysis import cli\n",
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    offenders = _forbidden_import_offenders(notebook, notebook.read_text(encoding="utf-8"))

    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_notebook_assignment_and_help_escapes_cannot_hide_later_import(
    tmp_path: Path,
) -> None:
    """IPython assignment/help escapes must be transformed before AST parsing."""
    notebook = tmp_path / "active.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "files = !ls\n",
                            "value?\n",
                            "from trend_analysis import cli\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    offenders = _forbidden_import_offenders(notebook, notebook.read_text(encoding="utf-8"))

    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_notebook_cell_magic_body_cannot_hide_retired_import(tmp_path: Path) -> None:
    """Cell-magic bodies remain executable Python after IPython wrapping."""
    notebook = tmp_path / "active.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "%%capture\n",
                            "files = !ls\n",
                            "from trend_analysis import cli\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    offenders = _forbidden_import_offenders(notebook, notebook.read_text(encoding="utf-8"))

    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_non_python_cell_magic_cannot_hide_retired_import(tmp_path: Path) -> None:
    """Shell cell magics may execute Python imports through heredocs."""
    notebook = tmp_path / "active.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "%%bash\n",
                            "python - <<'PY'\n",
                            "from trend_analysis import cli\n",
                            "PY\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    offenders = _forbidden_import_offenders(notebook, notebook.read_text(encoding="utf-8"))

    assert any("trend_analysis." + "cli" in offender for offender in offenders)


def test_legacy_surface_ci_runs_for_every_classified_change() -> None:
    """Every scanned surface must trigger the CI job that enforces this contract."""

    gate = (REPO_ROOT / ".github" / "workflows" / "pr-00-gate.yml").read_text(encoding="utf-8")
    marker = "  legacy-surface:\n"
    assert marker in gate, "legacy-surface job missing from gate workflow"
    legacy_job_header = gate.split(marker, 1)[1].split("    runs-on:", 1)[0]

    assert "!cancelled()" in legacy_job_header
    assert "is_python_code" not in legacy_job_header
    assert "is_docs_only" not in legacy_job_header


@pytest.mark.parametrize(
    ("arguments", "expected_output"),
    [
        (("--help",), "usage: trend"),
        (("run", "--help"), "usage: trend run"),
        (("report", "--help"), "usage: trend report"),
        (("quick-report", "--help"), "usage: trend quick-report"),
        (("app", "--help"), "usage: trend app"),
        (("check", "--help"), "usage: trend check"),
        (("mc", "list"), "hf_macro_20y"),
        (("mc", "validate", "--help"), "usage: trend mc validate"),
        (("mc", "run", "--help"), "usage: trend mc run"),
        (("mc", "viz", "--help"), "usage: trend mc viz"),
    ],
)
def test_supported_cli_surface_smoke(arguments: tuple[str, ...], expected_output: str) -> None:
    """The final command-tree smoke covers every supported public CLI surface."""

    result = subprocess.run(
        [sys.executable, "-m", "trend.cli", *arguments],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert expected_output in result.stdout.lower()
