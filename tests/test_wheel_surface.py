"""Wheel-content contract for the post-legacy supported package surface."""

from __future__ import annotations

import configparser
import shutil
import subprocess
import venv
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Split ``trend_analysis/`` literals below so this file does not self-match the
# legacy-surface scanner while still asserting retired wheel paths.
REMOVED_WHEEL_PREFIXES = (
    "trend/compat_entrypoints.py",
    "trend_analysis/" + "cli.py",
    "trend_analysis/" + "run_analysis.py",
    "trend_analysis/" + "run_multi_analysis.py",
    "trend_model/",
    "trend_portfolio_app/",
    "retired/",
    "examples/legacy_streamlit_app/",
    "examples/demo_" + "turnover_cap.py",
    "examples/portfolio_" + "analysis_report.py",
    "utils/",
)
EXPECTED_CONSOLE_SCRIPTS = {
    "trend": "trend.cli:main",
    "trend-llm-proxy": "trend_analysis.llm_proxy.cli:main",
}


def _build_wheel(destination: Path) -> Path:
    # ``pip wheel`` materializes ``*.egg-info`` while preparing metadata. Build
    # from tracked files plus non-ignored candidate additions so this contract
    # remains safe when the full suite creates ignored build artifacts. The
    # Gate interpreter need not itself expose the PEP 517 backend, so build in a
    # small isolated environment with the project's pinned build tools.
    build_root = destination / "source"
    environment = destination / "wheel-build-env"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    subprocess.run(
        [python, "-m", "pip", "install", "setuptools==83.0.0", "wheel==0.47.0"],
        check=True,
        capture_output=True,
        text=True,
    )
    candidate_files = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            ".",
            ":(exclude)build/**",
            ":(exclude)dist/**",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in candidate_files:
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode("utf-8"))
        # ``git ls-files`` includes staged or worktree deletions until commit;
        # build the wheel from the actual candidate tree.
        if not (REPO_ROOT / relative_path).exists():
            continue
        target_path = build_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative_path, target_path)

    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(destination),
        ],
        cwd=build_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(destination.glob("*.whl"))
    assert len(wheels) == 1, f"Expected one wheel, found: {wheels}"
    return wheels[0]


def test_wheel_builder_skips_missing_tracked_paths(tmp_path: Path, monkeypatch) -> None:
    """A staged deletion must not make the isolated wheel source copy fail."""

    original_run = subprocess.run

    def run_with_missing_path(command, *args, **kwargs):
        result = original_run(command, *args, **kwargs)
        if command == ["git", "ls-files", "-z"]:
            return subprocess.CompletedProcess(
                command,
                result.returncode,
                stdout=result.stdout + b"tests/missing-tracked-file.py\0",
                stderr=result.stderr,
            )
        return result

    monkeypatch.setattr(subprocess, "run", run_with_missing_path)

    assert _build_wheel(tmp_path).exists()


def test_wheel_contains_only_supported_entry_points_and_no_retired_surfaces(tmp_path: Path) -> None:
    """Build a wheel and inspect its files/metadata instead of trusting setup config alone."""

    wheel = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        returned = [
            name
            for name in names
            if any(name.startswith(prefix) for prefix in REMOVED_WHEEL_PREFIXES)
        ]
        assert not returned, "Wheel ships retired surfaces:\n" + "\n".join(returned)

        metadata_paths = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        assert len(metadata_paths) == 1
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(metadata_paths[0]).decode("utf-8"))

    assert dict(parser["console_scripts"]) == EXPECTED_CONSOLE_SCRIPTS

    python = tmp_path / "wheel-build-env" / "bin" / "python"
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    isolated = subprocess.run(
        [
            python,
            "-c",
            "import importlib.util; from trend_analysis.util.paths import proj_path; "
            "assert importlib.util.find_spec('utils') is None; assert proj_path()",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert isolated.returncode == 0, isolated.stderr
