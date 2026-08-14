"""Wheel-content contract for the post-legacy supported package surface."""

from __future__ import annotations

import configparser
import subprocess
import sys
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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(destination),
        ],
        cwd=REPO_ROOT,
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
