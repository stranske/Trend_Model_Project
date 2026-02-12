from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CHARTS = ("fan", "path_dist", "risk_return")


def _fixture_bundle_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "mc_bundle"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _kaleido_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("kaleido") is not None
    except Exception:
        return False


def _run_mc_viz(
    bundle_dir: Path,
    out_dir: Path,
    *,
    charts: str = "fan,path_dist,risk_return",
) -> subprocess.CompletedProcess[str]:
    project_root = _project_root()
    env = os.environ.copy()
    src_dir = project_root / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else str(src_dir)

    cmd = [
        sys.executable,
        "-m",
        "trend.cli",
        "mc",
        "viz",
        "--bundle",
        str(bundle_dir),
        "--out",
        str(out_dir),
        "--charts",
        charts,
        "--html",
        "--json",
        "--png",
    ]
    return subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _expected_png_paths(plots_dir: Path, charts: tuple[str, ...]) -> list[Path]:
    return [plots_dir / f"{chart}.png" for chart in charts]


@pytest.mark.integration
def test_mc_viz_cli_end_to_end_generates_expected_outputs(tmp_path: Path) -> None:
    bundle_dir = _fixture_bundle_dir()
    assert bundle_dir.is_dir()

    out_dir = tmp_path / "out"
    result = _run_mc_viz(bundle_dir, out_dir)

    assert result.returncode == 0, result.stderr

    plots_dir = out_dir / "plots"
    assert plots_dir.is_dir()

    for chart in CHARTS:
        html_path = plots_dir / f"{chart}.html"
        json_path = plots_dir / f"{chart}.json"
        assert html_path.is_file()
        assert json_path.is_file()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        chart_data = payload.get("data") if "data" in payload else payload.get("series")
        assert isinstance(chart_data, list)
        assert len(chart_data) > 0

        html_text = html_path.read_text(encoding="utf-8").lower()
        expected_markers = {
            "fan": ("fan chart", "fan"),
            "path_dist": ("path distribution", "path_dist"),
            "risk_return": ("risk vs. return", "risk_return"),
        }[chart]
        assert any(marker in html_text for marker in expected_markers)

    png_files = sorted(plots_dir.glob("*.png"))
    expected_png_paths = _expected_png_paths(plots_dir, CHARTS)
    if _kaleido_available():
        assert len(png_files) == len(CHARTS)
        for png_path in expected_png_paths:
            assert png_path.is_file()
            assert png_path.stat().st_size > 0, (
                f"Expected non-empty PNG chart artifact at {png_path}, "
                f"but file size was {png_path.stat().st_size} bytes."
            )
    else:
        assert len(png_files) == 0
        assert "PNG export skipped" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_skips_existing_plot_file_without_overwrite(tmp_path: Path) -> None:
    bundle_dir = _fixture_bundle_dir()
    assert bundle_dir.is_dir()

    out_dir = tmp_path / "out"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    existing_path = plots_dir / "risk_return.json"
    original_bytes = b'{"preexisting": true}'
    existing_path.write_bytes(original_bytes)

    result = _run_mc_viz(bundle_dir, out_dir)

    assert result.returncode == 0, result.stderr
    assert existing_path.read_bytes() == original_bytes
    assert "risk_return.json" in result.stderr
    assert "destination file already exists" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_errors_when_nav_paths_missing_for_required_chart(tmp_path: Path) -> None:
    bundle_dir = _fixture_bundle_dir()
    assert bundle_dir.is_dir()
    missing_nav_bundle_dir = tmp_path / "bundle_no_nav_paths"
    shutil.copytree(bundle_dir, missing_nav_bundle_dir)
    (missing_nav_bundle_dir / "nav_paths.parquet").unlink()

    out_dir = tmp_path / "out_required"
    result = _run_mc_viz(missing_nav_bundle_dir, out_dir, charts="fan,path_dist")

    assert result.returncode != 0
    assert "nav_paths.parquet" in result.stderr
    assert "path_dist" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_warns_and_continues_when_nav_paths_missing_for_non_required_charts(
    tmp_path: Path,
) -> None:
    bundle_dir = _fixture_bundle_dir()
    assert bundle_dir.is_dir()
    missing_nav_bundle_dir = tmp_path / "bundle_no_nav_paths"
    shutil.copytree(bundle_dir, missing_nav_bundle_dir)
    (missing_nav_bundle_dir / "nav_paths.parquet").unlink()

    out_dir = tmp_path / "out_non_required"
    result = _run_mc_viz(missing_nav_bundle_dir, out_dir, charts="fan,risk_return")

    assert result.returncode == 0, result.stderr
    assert "nav_paths.parquet" in result.stderr
    assert "NAV-path-dependent visuals" in result.stderr
    assert (out_dir / "plots" / "fan.html").is_file()
    assert (out_dir / "plots" / "risk_return.html").is_file()
