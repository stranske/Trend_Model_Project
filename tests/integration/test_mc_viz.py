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
    """Return True when kaleido is importable *and* can actually produce PNGs."""
    try:
        import importlib.util

        if importlib.util.find_spec("kaleido") is None:
            return False
        # Verify actual rendering works (kaleido 1.x may import but fail to
        # render with newer Plotly versions).
        import plotly.graph_objects as go
        import plotly.io as pio

        fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
        pio.to_image(fig, format="png", validate=True)
        return True
    except Exception:
        return False


def _run_mc_viz(
    bundle_dir: Path,
    out_dir: Path,
    *,
    charts: str = "fan,path_dist,risk_return",
    html: bool = True,
    json_flag: bool = True,
    png: bool = True,
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
    ]
    if html:
        cmd.append("--html")
    if json_flag:
        cmd.append("--json")
    if png:
        cmd.append("--png")
    return subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_mc_viz_without_kaleido(
    bundle_dir: Path,
    out_dir: Path,
    *,
    charts: str = "fan,path_dist,risk_return",
) -> subprocess.CompletedProcess[str]:
    """Run mc viz with kaleido hidden from the import system."""
    project_root = _project_root()
    env = os.environ.copy()
    src_dir = project_root / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else str(src_dir)

    # Use a wrapper that blocks kaleido imports before invoking the CLI.
    wrapper = (
        "import sys;"
        "import importlib.util;"
        "_orig = importlib.util.find_spec;"
        "importlib.util.find_spec = lambda n, *a, **kw: None if n == 'kaleido' else _orig(n, *a, **kw);"
        "sys.argv = ['trend', 'mc', 'viz',"
        f"'--bundle', {str(bundle_dir)!r},"
        f"'--out', {str(out_dir)!r},"
        f"'--charts', {charts!r},"
        "'--html', '--json', '--png'];"
        "from trend.cli import main; sys.exit(main())"
    )
    return subprocess.run(
        [sys.executable, "-c", wrapper],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Integration Tests — Missing Inputs
# ---------------------------------------------------------------------------


def _marker_for(chart: str) -> str:
    return f"<!-- mc-viz-chart:{chart} -->"


def _expected_png_paths(plots_dir: Path, charts: tuple[str, ...]) -> list[Path]:
    return [plots_dir / f"{chart}.png" for chart in charts]


@pytest.mark.integration
def test_mc_viz_cli_errors_when_nav_paths_missing_for_path_dist(tmp_path: Path) -> None:
    """Missing nav_paths.parquet for path_dist -> non-zero exit code."""
    bundle_dir = _fixture_bundle_dir()
    assert bundle_dir.is_dir()
    missing_nav_bundle = tmp_path / "bundle_no_nav"
    shutil.copytree(bundle_dir, missing_nav_bundle)
    (missing_nav_bundle / "nav_paths.parquet").unlink()

    out_dir = tmp_path / "out"
    result = _run_mc_viz(missing_nav_bundle, out_dir, charts="fan,path_dist", png=False)

    assert result.returncode != 0
    assert "Chart(s) path_dist require nav_paths.parquet in the MC bundle." in result.stderr
    assert "Add nav_paths.parquet or remove these chart(s) from --charts." in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_full_run_with_html_markers_and_chart_consistency(tmp_path: Path) -> None:
    """Full run asserts HTML markers, JSON payloads, and chart-set consistency."""
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

        html_text = html_path.read_text(encoding="utf-8")
        assert _marker_for(chart) in html_text
        assert f'id="chart-{chart}"' in html_text

    png_files = sorted(plots_dir.glob("*.png"))
    expected_png = _expected_png_paths(plots_dir, CHARTS)
    html_chart_ids = {path.stem for path in plots_dir.glob("*.html")}
    assert html_chart_ids == set(CHARTS)
    if _kaleido_available():
        assert len(png_files) == len(CHARTS)
        for png_path in expected_png:
            assert png_path.is_file()
            assert png_path.stat().st_size > 0, (
                f"Expected non-empty PNG chart artifact at {png_path}, "
                f"but file size was {png_path.stat().st_size} bytes."
            )
        png_chart_ids = {path.stem for path in png_files}
        assert png_chart_ids == html_chart_ids
    else:
        assert len(png_files) == 0
        assert "PNG export skipped" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_outputs_only_selected_chart_set_across_formats(tmp_path: Path) -> None:
    bundle_dir = _fixture_bundle_dir()
    assert bundle_dir.is_dir()

    selected = ("fan", "risk_return")
    out_dir = tmp_path / "out_selected"
    result = _run_mc_viz(bundle_dir, out_dir, charts="fan,risk_return")

    assert result.returncode == 0, result.stderr
    plots_dir = out_dir / "plots"
    assert plots_dir.is_dir()

    html_chart_ids = {path.stem for path in plots_dir.glob("*.html")}
    assert html_chart_ids == set(selected)
    for chart in selected:
        html_text = (plots_dir / f"{chart}.html").read_text(encoding="utf-8")
        assert _marker_for(chart) in html_text
        assert f'id="chart-{chart}"' in html_text

    png_chart_ids = {path.stem for path in plots_dir.glob("*.png")}
    if _kaleido_available():
        assert png_chart_ids == set(selected)
    else:
        assert png_chart_ids == set()
        assert "PNG export skipped" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_renames_existing_plot_file_collision_without_overwrite(tmp_path: Path) -> None:
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
    assert (plots_dir / "risk_return-1.json").is_file()
    assert "risk_return.json" in result.stderr
    assert "Renamed extracted 'risk_return.json' to 'risk_return-1.json'" in result.stderr


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
    assert "Chart(s) path_dist require nav_paths.parquet in the MC bundle." in result.stderr
    assert "Add nav_paths.parquet or remove these chart(s) from --charts." in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_error_message_lists_exact_missing_filename(tmp_path: Path) -> None:
    """Error message contains the literal string 'nav_paths.parquet'."""
    bundle_dir = _fixture_bundle_dir()
    missing_nav_bundle = tmp_path / "bundle_no_nav"
    shutil.copytree(bundle_dir, missing_nav_bundle)
    (missing_nav_bundle / "nav_paths.parquet").unlink()

    out_dir = tmp_path / "out"
    result = _run_mc_viz(missing_nav_bundle, out_dir, charts="path_dist", png=False)

    assert result.returncode != 0
    assert "nav_paths.parquet" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_error_lists_multiple_missing_files(tmp_path: Path) -> None:
    """When multiple required inputs are absent, all are listed."""
    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()

    out_dir = tmp_path / "out"
    result = _run_mc_viz(empty_bundle, out_dir, charts="fan", png=False)

    assert result.returncode != 0
    assert "summary" in result.stderr
    assert "results" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_fan_missing_required_inputs(tmp_path: Path) -> None:
    """fan chart with missing required inputs -> non-zero exit and error."""
    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()

    out_dir = tmp_path / "out"
    result = _run_mc_viz(empty_bundle, out_dir, charts="fan", png=False)

    assert result.returncode != 0
    assert "summary" in result.stderr or "results" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_risk_return_missing_required_inputs(tmp_path: Path) -> None:
    """risk_return chart with missing required inputs -> non-zero exit."""
    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()

    out_dir = tmp_path / "out"
    result = _run_mc_viz(empty_bundle, out_dir, charts="risk_return", png=False)

    assert result.returncode != 0
    assert "summary" in result.stderr or "results" in result.stderr


# ---------------------------------------------------------------------------
# Integration Tests — Successful Runs
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mc_viz_cli_creates_plots_directory(tmp_path: Path) -> None:
    """On success, <out_dir>/plots/ is created."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz(bundle_dir, out_dir, png=False)

    assert result.returncode == 0, result.stderr
    assert (out_dir / "plots").is_dir()


@pytest.mark.integration
def test_mc_viz_cli_generates_html_artifacts(tmp_path: Path) -> None:
    """HTML files matching *fan*.html etc. are generated when --html is set."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz(bundle_dir, out_dir, png=False)
    assert result.returncode == 0, result.stderr

    plots_dir = out_dir / "plots"
    for chart in CHARTS:
        matches = list(plots_dir.glob(f"*{chart}*.html"))
        assert len(matches) >= 1, f"No HTML file found for chart '{chart}'"


@pytest.mark.integration
def test_mc_viz_cli_generates_json_artifacts(tmp_path: Path) -> None:
    """JSON files matching *fan*.json etc. are generated when --json is set."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz(bundle_dir, out_dir, png=False)
    assert result.returncode == 0, result.stderr

    plots_dir = out_dir / "plots"
    for chart in CHARTS:
        matches = list(plots_dir.glob(f"*{chart}*.json"))
        assert len(matches) >= 1, f"No JSON file found for chart '{chart}'"


@pytest.mark.integration
def test_mc_viz_cli_correct_artifact_count(tmp_path: Path) -> None:
    """3 HTML + 3 JSON for --charts fan,path_dist,risk_return --html --json."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz(bundle_dir, out_dir, png=False)
    assert result.returncode == 0, result.stderr

    plots_dir = out_dir / "plots"
    html_files = list(plots_dir.glob("*.html"))
    json_files = list(plots_dir.glob("*.json"))
    assert len(html_files) == 3
    assert len(json_files) == 3


@pytest.mark.integration
def test_mc_viz_cli_artifact_filenames(tmp_path: Path) -> None:
    """Artifact filenames follow the expected naming convention."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz(bundle_dir, out_dir, png=False)
    assert result.returncode == 0, result.stderr

    plots_dir = out_dir / "plots"
    for chart in CHARTS:
        assert (plots_dir / f"{chart}.html").is_file()
        assert (plots_dir / f"{chart}.json").is_file()


# ---------------------------------------------------------------------------
# Integration Tests — PNG Behavior
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _kaleido_available(), reason="kaleido not functional")
def test_mc_viz_cli_generates_png_when_kaleido_available(tmp_path: Path) -> None:
    """PNGs generated (>=3) when kaleido works and --png is set."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz(bundle_dir, out_dir)
    assert result.returncode == 0, result.stderr

    plots_dir = out_dir / "plots"
    png_files = sorted(plots_dir.glob("*.png"))
    assert len(png_files) >= 3


@pytest.mark.integration
def test_mc_viz_cli_fails_when_kaleido_missing_and_png_requested(tmp_path: Path) -> None:
    """Graceful degradation when --png is requested but kaleido unavailable."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz_without_kaleido(bundle_dir, out_dir)

    assert result.returncode == 0, result.stderr
    assert "kaleido" in result.stderr.lower()
    plots_dir = out_dir / "plots"
    assert len(list(plots_dir.glob("*.png"))) == 0


@pytest.mark.integration
def test_mc_viz_cli_error_contains_install_hint_when_kaleido_missing(
    tmp_path: Path,
) -> None:
    """Warning message contains both 'kaleido' and 'pip install kaleido'."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz_without_kaleido(bundle_dir, out_dir)

    assert result.returncode == 0, result.stderr
    assert "kaleido" in result.stderr
    assert "pip install kaleido" in result.stderr


@pytest.mark.integration
def test_mc_viz_cli_no_png_when_flag_not_set(tmp_path: Path) -> None:
    """No PNG files when --png is not set, regardless of kaleido."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"

    result = _run_mc_viz(bundle_dir, out_dir, png=False)
    assert result.returncode == 0, result.stderr

    plots_dir = out_dir / "plots"
    png_files = list(plots_dir.glob("*.png"))
    assert len(png_files) == 0


# ---------------------------------------------------------------------------
# Integration Tests — Nav-paths warning and nested output
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mc_viz_cli_warns_and_continues_when_nav_paths_missing_for_non_required_charts(
    tmp_path: Path,
) -> None:
    """fan/risk_return succeed without nav_paths.parquet but emit a warning."""
    bundle_dir = _fixture_bundle_dir()
    missing_nav_bundle = tmp_path / "bundle_no_nav"
    shutil.copytree(bundle_dir, missing_nav_bundle)
    (missing_nav_bundle / "nav_paths.parquet").unlink()

    out_dir = tmp_path / "out"
    result = _run_mc_viz(missing_nav_bundle, out_dir, charts="fan,risk_return", png=False)

    assert result.returncode == 0, result.stderr
    assert "Missing optional nav_paths.parquet file in MC bundle:" in result.stderr
    assert "Continuing without NAV-path data." in result.stderr
    assert "NAV-path-dependent visuals" in result.stderr
    assert (out_dir / "plots" / "fan.html").is_file()
    assert (out_dir / "plots" / "risk_return.html").is_file()


@pytest.mark.integration
def test_mc_viz_cli_creates_nested_output_directory_tree(tmp_path: Path) -> None:
    """Deeply nested output directories are created automatically."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "nested" / "a" / "b" / "c" / "plots_output"
    assert not out_dir.exists()

    result = _run_mc_viz(bundle_dir, out_dir, charts="fan", png=False)

    assert result.returncode == 0, result.stderr
    plots_dir = out_dir / "plots"
    assert plots_dir.is_dir()
    assert (plots_dir / "fan.html").is_file()
    assert (plots_dir / "fan.json").is_file()


@pytest.mark.integration
def test_mc_viz_cli_skips_existing_plot_file_without_overwrite(tmp_path: Path) -> None:
    """Pre-existing files in plots/ are renamed to avoid collision."""
    bundle_dir = _fixture_bundle_dir()
    out_dir = tmp_path / "out"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    existing_path = plots_dir / "risk_return.json"
    original_bytes = b'{"preexisting": true}'
    existing_path.write_bytes(original_bytes)

    result = _run_mc_viz(bundle_dir, out_dir, png=False)

    assert result.returncode == 0, result.stderr
    assert existing_path.read_bytes() == original_bytes
    assert "risk_return.json" in result.stderr
    assert "Renamed extracted" in result.stderr or "collision" in result.stderr
