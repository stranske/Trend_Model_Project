"""Installed-console-script contract tests for the supported CLI surface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path

import pytest

# Keep the shared install on one worker so venv creation is not
# duplicated across xdist workers.
pytestmark = pytest.mark.xdist_group(name="installed_cli")

REMOVED_COMMANDS = (
    "trend-analysis",
    "trend-multi-analysis",
    "trend-model",
    "trend-app",
    "trend-run",
    "trend-quick-report",
)

# Exclude heavy / generated trees when copying sources for an isolated install.
_INSTALL_COPY_IGNORE = shutil.ignore_patterns(
    ".env",
    ".env.*",
    "secrets.toml",
    ".git",
    ".github",
    ".venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".workflows-lib",
    "__pycache__",
    "*.pyc",
    "node_modules",
    "archives",
    "notebooks",
    "build",
    "dist",
    "*.egg-info",
)


def _copy_install_sources(project_root: Path, source_copy: Path) -> None:
    """Copy build inputs without retaining local credentials or generated trees."""
    shutil.copytree(
        project_root,
        source_copy,
        dirs_exist_ok=True,
        ignore=_INSTALL_COPY_IGNORE,
    )


@pytest.fixture(scope="module")
def installed_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Share one isolated installation between this module's CLI contracts."""
    environment = tmp_path_factory.mktemp("installed-cli", numbered=False) / "installed-package"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin" / "python"
    project_root = Path(__file__).resolve().parents[1]
    # Non-editable install must not build in the shared checkout: setuptools
    # writes build/ and egg-info there and races other xdist workers reading
    # checkout CSVs (EmptyDataError). Copy sources, build a wheel from the copy,
    # then install that wheel so packaged preset_data comes from the package.
    # Match pyproject.toml build-system pins; CI venvs may not expose setuptools.
    subprocess.run(
        [python, "-m", "pip", "install", "setuptools==83.0.0", "wheel==0.47.0"],
        check=True,
        text=True,
    )
    wheel_dir = tmp_path_factory.mktemp("installed-cli-wheels", numbered=False)
    # Remove the source copy even if the build fails; pytest retains tmp paths.
    with tempfile.TemporaryDirectory(prefix="installed-cli-src-") as source_dir:
        source_copy = Path(source_dir)
        _copy_install_sources(project_root, source_copy)
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                str(source_copy),
                "-w",
                str(wheel_dir),
            ],
            check=True,
            text=True,
        )
    wheel_path = next(wheel_dir.glob("*.whl"))
    # system_site_packages=True sees the CI env's same-version install and
    # would otherwise skip writing console scripts into this venv's bin/.
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel_path),
        ],
        check=True,
        text=True,
    )
    return environment / "bin"


def test_install_source_copy_excludes_local_secrets(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / ".streamlit").mkdir(parents=True)
    for name in (".env", ".env.production", ".streamlit/secrets.toml"):
        (project_root / name).write_text("synthetic-secret", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (project_root / ".streamlit/config.toml").write_text("[theme]", encoding="utf-8")
    source_copy = tmp_path / "source-copy"

    _copy_install_sources(project_root, source_copy)

    for name in (".env", ".env.production", ".streamlit/secrets.toml"):
        assert not (source_copy / name).exists()
    assert (source_copy / "pyproject.toml").is_file()
    assert (source_copy / ".streamlit/config.toml").is_file()


def test_removed_compat_entrypoints_are_not_installed(installed_bin: Path) -> None:
    """Only supported console scripts are present after an isolated installation."""
    assert (installed_bin / "trend").is_file()
    assert (installed_bin / "trend-llm-proxy").is_file()
    for command in REMOVED_COMMANDS:
        assert not (installed_bin / command).exists()

    result = subprocess.run(
        [installed_bin / "trend", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{installed_bin}:{os.environ.get('PATH', '')}"},
    )
    assert result.returncode == 0, result.stderr
    assert "usage: trend" in result.stdout.lower()


def test_installed_monte_carlo_list_uses_packaged_registry(
    installed_bin: Path, tmp_path: Path
) -> None:
    """The installed CLI must list its shipped scenarios outside the checkout."""
    env = {**os.environ, "PATH": f"{installed_bin}:{os.environ.get('PATH', '')}"}
    env.pop("PYTHONPATH", None)
    env.pop("TREND_REPO_ROOT", None)
    result = subprocess.run(
        [installed_bin / "trend", "mc", "list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    expected_names = {
        "cost_regime_example",
        "credit_stress_15y",
        "hf_credit_liquidity_7y",
        "hf_diversified_5y",
        "hf_equity_ls_10y",
        "hf_macro_20y",
        "example_scenario",
    }
    assert expected_names.issubset(result.stdout.split())

    registry = subprocess.run(
        [
            installed_bin / "python",
            "-c",
            (
                "import json; import yaml; "
                "from trend_analysis.monte_carlo.registry import list_scenarios; "
                "print(json.dumps([(item.name, item.path.is_file() and "
                "isinstance(yaml.safe_load(item.path.read_text(encoding='utf-8')), dict)) "
                "for item in list_scenarios()]))"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert registry.returncode == 0, registry.stderr
    scenarios = dict(json.loads(registry.stdout))
    assert set(scenarios) == expected_names
    assert all(scenarios.values())
