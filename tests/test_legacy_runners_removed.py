from __future__ import annotations

import ast
import importlib.machinery
from pathlib import Path

import pytest

SRC_ROOT = Path("src").resolve()
PACKAGE_ROOT = SRC_ROOT / "trend_analysis"


@pytest.mark.parametrize(
    "module_name",
    [
        "trend_analysis.run_analysis",
        "trend_analysis.run_multi_analysis",
    ],
)
def test_legacy_runner_modules_gone(module_name: str) -> None:
    module_path = Path("src") / Path(*module_name.split(".")).with_suffix(".py")
    assert not module_path.exists()
    leaf_name = module_name.rpartition(".")[2]
    assert importlib.machinery.PathFinder.find_spec(leaf_name, [str(PACKAGE_ROOT)]) is None


def test_legacy_runner_names_not_lazy_exported() -> None:
    init_source = Path("src/trend_analysis/__init__.py").read_text()
    init_tree = ast.parse(init_source)
    exported_names: set[str] = set()

    for node in ast.walk(init_tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if target_names & {"_LAZY_SUBMODULES", "__all__"}:
            value = ast.literal_eval(node.value)
            exported_names.update(str(item) for item in value)

    assert "run_analysis" not in exported_names
    assert "run_multi_analysis" not in exported_names
