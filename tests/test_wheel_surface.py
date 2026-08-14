"""Wheel-content contract for the post-legacy supported package surface."""

from __future__ import annotations

import configparser
import shutil
import subprocess
import venv
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOVED_WHEEL_PREFIXES = (
    "trend/compat_entrypoints.py",
    "trend_analysis/run_analysis.py",
    "trend_analysis/run_multi_analysis.py",
    "trend_model/",
    "trend_portfolio_app/",
    "retired/",
    "examples/legacy_streamlit_app/",
)
EXPECTED_CONSOLE_SCRIPTS = {
    "trend": "trend.cli:main",
    "trend-llm-proxy": "trend_analysis.llm_proxy.cli:main",
}


def _build_wheel(destination: Path) -> Path:
    # ``pip wheel`` materializes ``*.egg-info`` while preparing metadata. Build
    # from tracked files so this contract remains safe when the full suite runs
    # under pytest-xdist and other tests create untracked build artifacts. The
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
    tracked_files = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in tracked_files:
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode("utf-8"))
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
