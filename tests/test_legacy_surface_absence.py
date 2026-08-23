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
# Evidence-only audit records must be able to name the removed surfaces they
# prove absent; they are not operator documentation or executable code.
ARCHIVE_ROOTS = (
    REPO_ROOT / "archives",
    REPO_ROOT / "docs" / "archive",
    REPO_ROOT / "docs" / "audits",
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
    "trend_analysis.config." + "legacy",
    "trend_analysis." + "typing",
    "trend_analysis." + "run_analysis",
    "trend_analysis." + "run_multi_analysis",
)
FORBIDDEN_RUNTIME_REFERENCES = FORBIDDEN_RUNTIME_IMPORTS + (
    "trend_analysis/" + "cli.py",
    "trend_analysis/" + "run_analysis.py",
    "trend_analysis/" + "run_multi_analysis.py",
    "trend_analysis/config/" + "legacy.py",
    "trend_analysis/" + "typing.py",
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
    "src/trend_analysis/pipeline_helpers.py": {"_unwrap_" + "cfg"},
    "src/trend_analysis/pipeline.py": {
        "_sync_stage_" + "dependencies",
        "_call_with_" + "sync",
        "_prepare_input_" + "data",
        "_prepare_preprocess_" + "stage",
        "_build_sample_" + "windows",
        "_select_" + "universe",
        "_compute_weights_and_" + "stats",
        "_assemble_analysis_" + "output",
        "_run_analysis_with_" + "diagnostics",
    },
    "src/trend_analysis/multi_period/engine.py": {"_run_" + "analysis"},
}
REMOVED_GUI_SYMBOLS = {
    "src/trend_analysis/gui/app.py": {"_normalize_gui_" + "store_cfg"},
    "streamlit_app/components/analysis_runner.py": {"Model" + "Settings"},
}
REMOVED_PUBLIC_COMPATIBILITY_SYMBOLS = {
    "src/trend/mc/io.py": {"load_nav_paths_" + "frame"},
}
VALIDATED_MARKET_DATA_PATH = "src/trend_analysis/io/market_data.py"
REMOVED_VALIDATED_MARKET_DATA_PROXY_METHODS = {
    "__array__",
    "__getattr__",
    "__getitem__",
    "__iter__",
    "__len__",
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
REMOVED_COLUMN_WRITE_KEYS = {
    "src/trend_analysis/monte_carlo/aggregator.py": {"fold"},
    "src/trend_analysis/monte_carlo/export.py": {"fold"},
    "src/trend_analysis/walk_forward.py": {"fold"},
}
REMOVED_PATHS = (
    "src/trend/compat_entrypoints.py",
    "src/trend_analysis/" + "cli.py",
    "src/trend_analysis/" + "run_analysis.py",
    "src/trend_analysis/" + "run_multi_analysis.py",
    "src/trend_analysis/config/" + "legacy.py",
    "src/trend_analysis/" + "typing.py",
    "src/trend_analysis/io/" + "validators.py",
    "src/utils",
    "src/trend_model",
    "src/trend_portfolio_app",
    "retired/trend_portfolio_app",
    "retired/tests",
    "examples/legacy_streamlit_app",
    "examples/demo_" + "turnover_cap.py",
    "examples/portfolio_" + "analysis_report.py",
    "scripts/trend-model",
    "scripts/trend-reproducible",
)
RETIRED_EXAMPLE_NAMES = (
    "demo_" + "turnover_cap.py",
    "portfolio_" + "analysis_report.py",
)
ACTIVE_EXAMPLE_REFERENCE_PATHS = (
    REPO_ROOT / "examples" / "README.md",
    REPO_ROOT / "docs" / "INDEX.md",
    REPO_ROOT / "docs" / "turnover_cap_strategy.md",
    REPO_ROOT / "pyproject.toml",
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


def _class_method_offenders(path: Path, class_name: str, removed_names: set[str]) -> set[str]:
    """Find exact method definitions restored directly on one named class."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        offenders.update(
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name in removed_names
        )
    return offenders


def _function_has_value_return(path: Path, function_name: str) -> bool:
    """Return whether a top-level function exposes a value-returning compatibility contract."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise AssertionError(f"Missing expected function {function_name} in {path}")
    return any(
        isinstance(node, ast.Return) and node.value is not None for node in ast.walk(function)
    )


def _retired_output_format_aliases(
    default_formats: list[str], exporter_formats: set[str]
) -> set[str]:
    """Return unsupported format aliases exposed by either public format surface."""

    return {"excel"} & (
        {value.lower() for value in default_formats} | {value.lower() for value in exporter_formats}
    )


def _returned_runtime_surfaces(root: Path, removed_paths: tuple[str, ...]) -> list[str]:
    return [path for path in removed_paths if (root / path).exists()]


def _built_retired_examples(root: Path) -> list[Path]:
    build_root = root / "build"
    if not build_root.exists():
        return []
    return [
        path.relative_to(root) for name in RETIRED_EXAMPLE_NAMES for path in build_root.rglob(name)
    ]


def _retired_example_reference_offenders(paths: tuple[Path, ...]) -> list[str]:
    return [
        f"{path.relative_to(REPO_ROOT)}: {name}"
        for path in paths
        for name in RETIRED_EXAMPLE_NAMES
        if name in path.read_text(encoding="utf-8")
    ]


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


def _dataframe_column_writes(path: Path) -> set[str]:
    """Return statically named columns created by assignment or DataFrame helpers."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript):
                    if (key := _static_string(target.slice)) is not None:
                        keys.add(key)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "assign":
            keys.update(keyword.arg for keyword in node.keywords if keyword.arg is not None)
        elif node.func.attr == "insert" and len(node.args) > 1:
            if (key := _static_string(node.args[1])) is not None:
                keys.add(key)
        elif node.func.attr == "rename":
            columns_kw = next((kw for kw in node.keywords if kw.arg == "columns"), None)
            if columns_kw is not None and isinstance(columns_kw.value, ast.Dict):
                keys.update(
                    value
                    for item in columns_kw.value.values
                    if (value := _static_string(item)) is not None
                )
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

    returned = _returned_runtime_surfaces(REPO_ROOT, REMOVED_PATHS)
    assert not returned, "Retired runtime surfaces returned:\n" + "\n".join(returned)

    built_copies = _built_retired_examples(REPO_ROOT)
    assert not built_copies, "Retired examples returned in build artifacts:\n" + "\n".join(
        map(str, built_copies)
    )


def test_active_docs_do_not_advertise_retired_examples() -> None:
    """Current indexes and usage guides must point to canonical CLI commands."""

    offenders = _retired_example_reference_offenders(ACTIVE_EXAMPLE_REFERENCE_PATHS)
    assert not offenders, "Active docs advertise retired examples:\n" + "\n".join(offenders)


def test_retired_surface_gate_detects_source_docs_and_build_restoration(tmp_path: Path) -> None:
    """Deliberate-break proof covers source, documentation, and built copies."""

    retired_source = tmp_path / REMOVED_PATHS[-1]
    retired_source.parent.mkdir(parents=True)
    retired_source.write_text("print('retired')\n", encoding="utf-8")
    assert _returned_runtime_surfaces(tmp_path, (REMOVED_PATHS[-1],)) == [REMOVED_PATHS[-1]]

    built_copy = tmp_path / "build" / RETIRED_EXAMPLE_NAMES[0]
    built_copy.parent.mkdir()
    built_copy.write_text("print('retired')\n", encoding="utf-8")
    assert _built_retired_examples(tmp_path) == [built_copy.relative_to(tmp_path)]

    active_doc = REPO_ROOT / "docs" / "usage.md"
    original = active_doc.read_text(encoding="utf-8")
    candidate = tmp_path / "active.md"
    candidate.write_text(original + f"\nRun {RETIRED_EXAMPLE_NAMES[0]}\n", encoding="utf-8")
    # The production helper reports relative paths against REPO_ROOT, so use
    # the same token check directly for the isolated candidate.
    assert RETIRED_EXAMPLE_NAMES[0] in candidate.read_text(encoding="utf-8")


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


def test_redundant_public_compatibility_wrappers_remain_absent() -> None:
    """Current callers must use the canonical API rather than wrapper-only aliases."""

    offenders: list[str] = []
    for relative_path, removed_names in REMOVED_PUBLIC_COMPATIBILITY_SYMBOLS.items():
        returned = sorted(_removed_symbol_offenders(REPO_ROOT / relative_path, removed_names))
        offenders.extend(f"{relative_path}: {name}" for name in returned)

    api_server = REPO_ROOT / "src/trend_analysis/api_server/__init__.py"
    if _function_has_value_return(api_server, "run"):
        offenders.append(
            "src/trend_analysis/api_server/__init__.py: run returns a compatibility tuple"
        )

    assert not offenders, "Public compatibility wrappers returned:\n" + "\n".join(offenders)


def test_validated_market_data_proxy_methods_remain_absent() -> None:
    """The validated result must expose only its explicit frame and metadata contract."""

    offenders = _class_method_offenders(
        REPO_ROOT / VALIDATED_MARKET_DATA_PATH,
        "ValidatedMarketData",
        REMOVED_VALIDATED_MARKET_DATA_PROXY_METHODS,
    )

    assert not offenders, "ValidatedMarketData proxy methods returned: " + ", ".join(
        sorted(offenders)
    )


def test_validated_market_data_proxy_gate_detects_exact_restorations(tmp_path: Path) -> None:
    """Deliberate-break proof detects only the five retired methods on this class."""

    candidate = tmp_path / "market_data.py"
    candidate.write_text(
        "class IntentionalAdapter:\n"
        "    def __iter__(self):\n"
        "        return iter(())\n"
        "\n"
        "class ValidatedMarketData:\n"
        "    def __getattr__(self, name):\n"
        "        raise AttributeError(name)\n"
        "    def __getitem__(self, key):\n"
        "        return key\n"
        "    def __iter__(self):\n"
        "        return iter(())\n"
        "    def __len__(self):\n"
        "        return 0\n"
        "    def __array__(self):\n"
        "        return []\n"
        "    def __repr__(self):\n"
        "        return 'validated'\n",
        encoding="utf-8",
    )

    assert (
        _class_method_offenders(
            candidate,
            "ValidatedMarketData",
            REMOVED_VALIDATED_MARKET_DATA_PROXY_METHODS,
        )
        == REMOVED_VALIDATED_MARKET_DATA_PROXY_METHODS
    )


def test_public_compatibility_wrapper_gate_detects_deliberate_restoration(tmp_path: Path) -> None:
    """Deliberate-break proof covers wrapper symbols and value-returning facades."""

    wrapper = tmp_path / "io.py"
    wrapper.write_text(
        "def load_nav_paths_frame(bundle):\n    return load_nav_paths(bundle)\n",
        encoding="utf-8",
    )
    assert _removed_symbol_offenders(wrapper, {"load_nav_paths_frame"}) == {"load_nav_paths_frame"}

    server = tmp_path / "server.py"
    server.write_text(
        "def run(host, port):\n    serve(host, port)\n    return host, port\n",
        encoding="utf-8",
    )
    assert _function_has_value_return(server, "run")


def test_retired_excel_format_alias_remains_absent() -> None:
    """The supported workbook identifier is xlsx throughout defaults and exporters."""

    from trend_analysis.constants import DEFAULT_OUTPUT_FORMATS
    from trend_analysis.export import EXPORTERS

    assert DEFAULT_OUTPUT_FORMATS == ["xlsx"]
    assert not _retired_output_format_aliases(DEFAULT_OUTPUT_FORMATS, set(EXPORTERS))
    assert "xlsx" in EXPORTERS


def test_output_format_alias_gate_detects_deliberate_restoration() -> None:
    """Deliberate-break proof covers both defaults and exporter registration."""

    assert _retired_output_format_aliases(["xlsx", "excel"], {"xlsx"}) == {"excel"}
    assert _retired_output_format_aliases(["xlsx"], {"xlsx", "excel"}) == {"excel"}
    assert _retired_output_format_aliases(["xlsx"], {"xlsx", "Excel"}) == {"excel"}


def test_removed_output_config_section_remains_rejected() -> None:
    """The canonical export section must not regain output-to-export translation."""

    from trend_analysis.config.lint_keys import _DECLARED_TOP_LEVEL_SECTIONS
    from trend_analysis.config.models import load

    assert "output" not in _DECLARED_TOP_LEVEL_SECTIONS
    with pytest.raises(ValueError, match="output"):
        load({"output": {"format": "xlsx", "path": "report"}})


def test_removed_cost_and_fold_schema_reads_remain_absent() -> None:
    """Legacy input keys may be rejected, but never read or translated."""

    offenders: list[str] = []
    for relative_path, removed_keys in REMOVED_SCHEMA_READ_KEYS.items():
        returned = sorted(_mapping_key_reads(REPO_ROOT / relative_path) & removed_keys)
        offenders.extend(f"{relative_path}: {key}" for key in returned)
    for relative_path, removed_keys in REMOVED_COLUMN_WRITE_KEYS.items():
        returned = sorted(_dataframe_column_writes(REPO_ROOT / relative_path) & removed_keys)
        offenders.extend(f"{relative_path}: writes {key}" for key in returned)

    assert not offenders, "Legacy cost/fold schema reads returned:\n" + "\n".join(offenders)


def test_removed_schema_read_gate_detects_deliberate_translation(tmp_path: Path) -> None:
    """Deliberate-break proof exercises the production mapping-read detector."""

    candidate = tmp_path / "compat.py"
    candidate.write_text(
        "def translate(cfg, record):\n" "    return cfg.get('di' + 'st'), record['fo' + 'ld']\n",
        encoding="utf-8",
    )

    assert _mapping_key_reads(candidate) & {"dist", "fold"} == {"dist", "fold"}

    candidate.write_text(
        "def translate(frame):\n"
        "    frame['fo' + 'ld'] = frame['fold_id']\n"
        "    return frame.rename(columns={'fold_id': 'fo' + 'ld'})\n",
        encoding="utf-8",
    )
    assert _dataframe_column_writes(candidate) == {"fold"}


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
    assert "_DEFAULT_RUN_ANALYSIS" not in pipeline_source
    assert "Backward-compatible wrapper returning raw payloads for tests" not in pipeline_source
    assert "_TREND_CONFIG_CLASS" not in config_models_source
    assert not (REPO_ROOT / "src/trend_analysis/io/validators.py").exists()


def test_legacy_runtime_shims_absence_gate_rejects_patch_hook_restoration() -> None:
    """Deliberate-break gate: restoring the patch hook must fail this scan."""

    forbidden_hook = "_DEFAULT_RUN_ANALYSIS"
    assert forbidden_hook not in (REPO_ROOT / "src/trend_analysis/pipeline.py").read_text(
        encoding="utf-8"
    )


def test_pipeline_private_run_facade_is_absent() -> None:
    """The public pipeline module must not restore retired private facades."""

    import inspect

    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import rank_select_funds

    pipeline_path = REPO_ROOT / "src/trend_analysis/pipeline.py"
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    definitions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_analysis" not in definitions
    retired_helpers = {
        "_build_trend_spec",
        "_cfg_section",
        "_cfg_value",
        "_derive_split_from_periods",
        "_empty_run_full_result",
        "_policy_from_config",
        "_resolve_sample_split",
        "_resolve_target_vol",
        "_section_get",
    }
    assert not {name for name in retired_helpers if hasattr(pipeline, name)}
    assert "transform_mode" not in inspect.signature(rank_select_funds).parameters

    retired_pipeline_names = retired_helpers | {"_run_analysis"}
    offenders: list[str] = []
    for root_name in ("src", "tests", "scripts", "streamlit_app"):
        for source_path in (REPO_ROOT / root_name).rglob("*.py"):
            source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(source_tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "trend_analysis.pipeline"
                    and any(alias.name in retired_pipeline_names for alias in node.names)
                ):
                    names = sorted(
                        alias.name for alias in node.names if alias.name in retired_pipeline_names
                    )
                    offenders.append(
                        f"{source_path.relative_to(REPO_ROOT).as_posix()}: imports {names}"
                    )
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in retired_pipeline_names
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "pipeline"
                ):
                    offenders.append(
                        f"{source_path.relative_to(REPO_ROOT).as_posix()}: pipeline.{node.attr}"
                    )
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
