from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import (
    auto_type_hygiene,
    fix_cosmetic_aggregate,
    fix_numpy_asserts,
    mypy_return_autofix,
    update_autofix_expectations,
)


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return repo_root


def test_fix_numpy_asserts_rewrites_array_equality(
    tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = tmp_repo / "tests"
    tests_dir.mkdir()
    target = tests_dir / "test_numpy_case.py"
    target.write_text(
        """import numpy as np\n\n\n"""
        "def test_numpy_autofix():\n"
        "    fancy_array = np.array([1, 2, 3])\n"
        "    assert fancy_array == [1, 2, 3]  # inline comment\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(fix_numpy_asserts, "ROOT", tmp_repo)
    monkeypatch.setattr(fix_numpy_asserts, "TEST_ROOT", tests_dir)
    monkeypatch.setattr(
        fix_numpy_asserts,
        "TARGET_FILES",
        {Path("tests/test_numpy_case.py")},
    )

    changed = fix_numpy_asserts.process_file(target)
    assert changed is True
    updated = target.read_text(encoding="utf-8")
    assert "assert fancy_array.tolist() == [1, 2, 3]" in updated


def test_update_autofix_expectations_overwrites_constant(
    tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = tmp_repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    module_path = tests_dir / "test_expectation_target.py"
    module_path.write_text(
        "EXPECTED_DYNAMIC_VALUE = 0\n\n\n"
        "def compute_expected_dynamic_value() -> int:\n"
        "    return 3\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_repo))
    tests_pkg = sys.modules.get("tests")
    if tests_pkg is None:
        import importlib

        tests_pkg = importlib.import_module("tests")
    orig_path = list(getattr(tests_pkg, "__path__", []))
    monkeypatch.setattr(
        tests_pkg,
        "__path__",
        [str(tests_dir), *orig_path],
        raising=False,
    )
    monkeypatch.setattr(update_autofix_expectations, "ROOT", tmp_repo)
    target = update_autofix_expectations.AutofixTarget(
        module="tests.test_expectation_target",
        callable_name="compute_expected_dynamic_value",
        constant_name="EXPECTED_DYNAMIC_VALUE",
    )
    monkeypatch.setattr(update_autofix_expectations, "TARGETS", (target,))

    try:
        result = update_autofix_expectations.main()
        assert result == 0
    finally:
        sys.modules.pop("tests.test_expectation_target", None)

    updated_text = module_path.read_text(encoding="utf-8")
    assert "EXPECTED_DYNAMIC_VALUE = 3" in updated_text


def test_auto_type_hygiene_inserts_type_ignore(
    tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_repo / "src"
    src_dir.mkdir()
    source_path = src_dir / "demo.py"
    source_path.write_text(
        "import yaml\n\nvalue = yaml.safe_load('{}')\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(auto_type_hygiene, "ROOT", tmp_repo)
    monkeypatch.setattr(auto_type_hygiene, "SRC_DIRS", [src_dir])
    monkeypatch.setattr(auto_type_hygiene, "DRY_RUN", False)

    changed, new_lines = auto_type_hygiene.process_file(source_path)
    assert changed is True
    assert new_lines[0].strip().endswith("# type: ignore[import-untyped]")


def test_mypy_return_autofix_updates_annotation(
    tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_repo / "src"
    src_dir.mkdir()
    module_rel = Path("src/strings.py")
    module_path = tmp_repo / module_rel
    module_path.write_text(
        "from __future__ import annotations\n\n\n"
        "def format_user(name: str) -> int:\n"
        "    return f'hello {name}'\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mypy_return_autofix, "ROOT", tmp_repo)
    monkeypatch.setattr(mypy_return_autofix, "PROJECT_DIRS", [src_dir])
    monkeypatch.setattr(
        mypy_return_autofix,
        "MYPY_CMD",
        [
            sys.executable,
            "-m",
            "mypy",
            "--hide-error-context",
            "--no-error-summary",
            str(module_rel),
        ],
    )

    result = mypy_return_autofix.main()
    assert result == 0
    updated_source = module_path.read_text(encoding="utf-8")
    assert "def format_user(name: str) -> str:" in updated_source


def test_fix_cosmetic_aggregate_switches_separator(
    tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_repo / "automation_multifailure.py"
    target.write_text(
        "from typing import Iterable\n\n\n"
        "def aggregate_numbers(values: Iterable[int]) -> int:\n"
        '    return ",".join(str(v) for v in values)\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(fix_cosmetic_aggregate, "ROOT", tmp_repo)
    monkeypatch.setattr(fix_cosmetic_aggregate, "TARGET", target)

    result = fix_cosmetic_aggregate.main()
    assert result == 0
    updated = target.read_text(encoding="utf-8")
    assert '" | ".join' in updated
