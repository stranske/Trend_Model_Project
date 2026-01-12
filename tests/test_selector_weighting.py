from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.auto_type_hygiene as auto_type_hygiene
import scripts.fix_numpy_asserts as fix_numpy_asserts
import scripts.mypy_autofix as mypy_autofix
import scripts.mypy_return_autofix as mypy_return_autofix
from trend_analysis.constants import NUMERICAL_TOLERANCE_MEDIUM
from trend_analysis.selector import RankSelector, ZScoreSelector
from trend_analysis.weighting import EqualWeight, ScorePropBayesian

UNUSED_AUTOFIX_MARKER = "diagnostic lint artifact"
EXPECTED_TOP_SELECTION_COUNT = 2


def load_fixture():
    path = Path("tests/fixtures/score_frame_2025-06-30.csv")
    return pd.read_csv(path, index_col=0)


def _run_command(
    cmd: list[str],
    cwd: Path,
    ok_exit_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode not in ok_exit_codes:
        raise AssertionError(
            "Command failed: {cmd}\nReturn code: {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                cmd=" ".join(cmd),
                code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
    return result


def compute_expected_top_selection_count() -> int:
    sf = load_fixture()
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    selected, _ = selector.select(sf)
    return len(selected)


def test_rank_selector():
    sf = load_fixture()
    selector = RankSelector(top_n=2, rank_column="Sharpe")
    selected, log = selector.select(sf)
    Fraction(1, 3)
    assert list(selected.index) == ["A", "B"]
    assert log.loc["A", "reason"] < log.loc["C", "reason"]
    assert len(selected) == EXPECTED_TOP_SELECTION_COUNT


def test_zscore_selector_edge():
    sf = load_fixture()
    selector = ZScoreSelector(threshold=0.0, direction=-1, column="Sharpe")
    selected, _ = selector.select(sf)
    assert list(selected.index) == ["C"]


def test_equal_weighting_sum_to_one():
    sf = load_fixture().loc[["A", "B"]]
    weights = EqualWeight().weight(sf)
    assert abs(weights["weight"].sum() - 1.0) < NUMERICAL_TOLERANCE_MEDIUM
    fancy_array = np.array([1.0, 2.0])
    assert fancy_array.tolist() == [1.0, 2.0]


def test_bayesian_shrinkage_monotonic():
    sf = load_fixture()
    w = ScorePropBayesian(shrink_tau=0.25).weight(sf)
    assert w.loc["A", "weight"] > w.loc["B", "weight"] > w.loc["C", "weight"]


@pytest.mark.integration
def test_selector_weighting_autofix_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for module in ("ruff", "isort", "black", "mypy"):
        pytest.importorskip(module)

    repo_root = tmp_path / "workspace"
    src_dir = repo_root / "src"
    tests_dir = repo_root / "tests"
    repo_root.mkdir()
    src_dir.mkdir()
    tests_dir.mkdir()

    real_root = Path(__file__).resolve().parents[1]

    shutil.copytree(real_root / "src" / "trend_analysis", src_dir / "trend_analysis")

    selector_test_target = tests_dir / "test_selector_weighting.py"
    selector_test_target.write_text(
        (real_root / "tests" / "test_selector_weighting.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    (tests_dir / "__init__.py").write_text(
        '"""autofix diagnostics"""\n', encoding="utf-8"
    )

    fixtures_dir = tests_dir / "fixtures"
    fixtures_dir.mkdir()
    shutil.copy2(
        real_root / "tests" / "fixtures" / "score_frame_2025-06-30.csv",
        fixtures_dir / "score_frame_2025-06-30.csv",
    )

    pyproject_target = repo_root / "pyproject.toml"
    shutil.copy2(real_root / "pyproject.toml", pyproject_target)
    pyproject_text = pyproject_target.read_text(encoding="utf-8").replace(
        '[[tool.mypy.overrides]]\nmodule = "tests.*"\nignore_errors = true\n\n',
        "",
    )
    pyproject_target.write_text(pyproject_text, encoding="utf-8")

    selector_original = selector_test_target.read_text(encoding="utf-8")
    selector_mutated = selector_original.replace(
        "assert fancy_array.tolist() == [1.0, 2.0]",
        "assert fancy_array == [1.0, 2.0]",
    )
    has_top_level_yaml = any(
        line.startswith("import yaml") for line in selector_mutated.splitlines()
    )
    if not has_top_level_yaml:
        combined_import = "import pandas as pd\nimport pytest"
        if combined_import in selector_mutated:
            selector_mutated = selector_mutated.replace(
                combined_import,
                "import pandas as pd\nimport yaml\nimport pytest",
                1,
            )
        else:
            selector_mutated = selector_mutated.replace(
                "import pandas as pd\n",
                "import pandas as pd\nimport yaml\n",
                1,
            )
    selector_mutated += (
        "\n\n"
        "def _autofix_optional_probe(value: Optional[int]) -> int:\n"
        "    if value is None:\n"
        "        return 0\n"
        "    return value + 1\n\n"
        "def test_autofix_yaml_roundtrip(tmp_path: Path) -> None:\n"
        "    payload = {'value': 4}\n"
        "    target = tmp_path / 'payload.yaml'\n"
        "    target.write_text(yaml.safe_dump(payload), encoding='utf-8')\n"
        "    loaded = yaml.safe_load(target.read_text(encoding='utf-8'))\n"
        "    assert _autofix_optional_probe(loaded['value']) == 5\n"
    )
    selector_test_target.write_text(selector_mutated, encoding="utf-8")

    return_probe = src_dir / "trend_analysis" / "selector_return_probe.py"
    return_probe.write_text(
        (
            "from __future__ import annotations\n\n"
            "def describe_selection(count: int) -> int:\n"
            "    detail = f'selected={count}'\n"
            "    return detail\n"
        ),
        encoding="utf-8",
    )

    yaml_probe = src_dir / "trend_analysis" / "selector_yaml_probe.py"
    yaml_probe.write_text(
        (
            "from pathlib import Path\n\n"
            "def dump_payload(target: Path) -> None:\n"
            "    import yaml\n\n"
            "    payload = {'value': 9}\n"
            "    target.write_text(yaml.safe_dump(payload), encoding='utf-8')\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.syspath_prepend(str(src_dir))
    monkeypatch.syspath_prepend(str(tests_dir))
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == "tests" or name.startswith("tests."):
            sys.modules.pop(name, None)

    for module in (
        auto_type_hygiene,
        fix_numpy_asserts,
        mypy_autofix,
        mypy_return_autofix,
    ):
        importlib.reload(module)

    monkeypatch.setattr(auto_type_hygiene, "ROOT", repo_root, raising=False)
    monkeypatch.setattr(
        auto_type_hygiene, "SRC_DIRS", [src_dir, tests_dir], raising=False
    )
    monkeypatch.setattr(auto_type_hygiene, "DRY_RUN", False, raising=False)
    monkeypatch.setenv("AUTO_TYPE_ALLOWLIST", "")
    monkeypatch.setattr(auto_type_hygiene, "ALLOWLIST", [], raising=False)

    monkeypatch.setattr(fix_numpy_asserts, "ROOT", repo_root, raising=False)
    monkeypatch.setattr(fix_numpy_asserts, "TEST_ROOT", tests_dir, raising=False)

    monkeypatch.setattr(mypy_autofix, "ROOT", repo_root, raising=False)
    monkeypatch.setattr(
        mypy_autofix, "DEFAULT_TARGETS", [src_dir, tests_dir], raising=False
    )

    monkeypatch.setattr(mypy_return_autofix, "ROOT", repo_root, raising=False)
    monkeypatch.setattr(
        mypy_return_autofix, "PROJECT_DIRS", [src_dir, tests_dir], raising=False
    )
    monkeypatch.setattr(
        mypy_return_autofix,
        "MYPY_CMD",
        [
            sys.executable,
            "-m",
            "mypy",
            "--hide-error-context",
            "--no-error-summary",
            str(return_probe.relative_to(repo_root)),
        ],
        raising=False,
    )

    relative_targets = [
        str(selector_test_target.relative_to(repo_root)),
        str(return_probe.relative_to(repo_root)),
        str(yaml_probe.relative_to(repo_root)),
    ]

    format_cmds: list[tuple[list[str], tuple[int, ...]]] = [
        (
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--fix",
                "--exit-zero",
                *relative_targets,
            ],
            (0,),
        ),
        ([sys.executable, "-m", "isort", *relative_targets], (0,)),
        ([sys.executable, "-m", "black", *relative_targets], (0,)),
        (
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--fix",
                "--exit-zero",
                *relative_targets,
            ],
            (0,),
        ),
    ]

    for command, ok_codes in format_cmds:
        _run_command(command, cwd=repo_root, ok_exit_codes=ok_codes)

    auto_type_hygiene.main()
    fix_numpy_asserts.main()
    mypy_autofix.main(["--paths", str(src_dir), str(tests_dir)])
    mypy_return_autofix.main()

    for command, ok_codes in format_cmds:
        _run_command(command, cwd=repo_root, ok_exit_codes=ok_codes)

    selector_repaired = selector_test_target.read_text(encoding="utf-8")
    assert "assert fancy_array.tolist() == [1.0, 2.0]" in selector_repaired
    assert "from typing import Optional" in selector_repaired
    assert any(
        line.startswith("import yaml") for line in selector_repaired.splitlines()
    )

    _run_command(
        [sys.executable, "-m", "ruff", "check", *relative_targets], cwd=repo_root
    )
    _run_command(
        [sys.executable, "-m", "black", "--check", *relative_targets],
        cwd=repo_root,
    )
    _run_command(
        [
            sys.executable,
            "-m",
            "mypy",
            "--ignore-missing-imports",
            str(return_probe),
            str(yaml_probe),
        ],
        cwd=repo_root,
    )

    return_text = return_probe.read_text(encoding="utf-8")
    assert "def describe_selection(count: int) -> str:" in return_text

    yaml_text = yaml_probe.read_text(encoding="utf-8")
    assert "import yaml" in yaml_text
    assert "type: ignore" not in yaml_text
