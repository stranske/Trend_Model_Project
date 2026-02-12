from __future__ import annotations

import inspect
from pathlib import Path
from unittest import mock

import pytest

from trend.mc import execute_mc_viz
from trend.mc.viz import (
    CHART_REQUIREMENTS,
    TrendCLIError,
    check_png_dependency,
    validate_mc_viz_bundle_requirements,
)


# ---------------------------------------------------------------------------
# Signature / constant tests
# ---------------------------------------------------------------------------


def test_execute_mc_viz_public_signature() -> None:
    signature = inspect.signature(execute_mc_viz)
    assert list(signature.parameters) == [
        "bundle_path",
        "out_dir",
        "charts",
        "html",
        "json",
        "png",
    ]
    assert execute_mc_viz.__doc__


def test_chart_requirements_define_supported_mc_viz_inputs() -> None:
    assert set(CHART_REQUIREMENTS) == {"fan", "path_dist", "risk_return"}
    assert CHART_REQUIREMENTS["fan"] == ("summary", "results")
    assert CHART_REQUIREMENTS["risk_return"] == ("summary", "results")
    assert CHART_REQUIREMENTS["path_dist"] == ("summary", "results", "nav_paths.parquet")


# ---------------------------------------------------------------------------
# Bundle validation tests
# ---------------------------------------------------------------------------


def _write_bundle_file(bundle_dir: Path, filename: str) -> None:
    (bundle_dir / filename).write_text("x", encoding="utf-8")


def test_bundle_validation_accepts_supported_formats_for_summary_and_results(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle_file(bundle_dir, "summary.csv")
    _write_bundle_file(bundle_dir, "results.json")

    missing = validate_mc_viz_bundle_requirements(bundle_dir, ["fan"])

    assert missing == []


def test_bundle_validation_reports_missing_results_requirement(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle_file(bundle_dir, "summary.parquet")

    missing = validate_mc_viz_bundle_requirements(bundle_dir, ["risk_return"])

    assert missing == ["results.parquet/results.csv/results.json (one required)"]


def test_bundle_validation_requires_nav_paths_parquet_for_path_dist(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle_file(bundle_dir, "summary.csv")
    _write_bundle_file(bundle_dir, "results.csv")
    _write_bundle_file(bundle_dir, "nav_paths.parquet")

    missing = validate_mc_viz_bundle_requirements(bundle_dir, "path_dist")

    assert missing == []


@pytest.mark.parametrize("filename", ["nav_paths.csv", "nav_paths.json"])
def test_bundle_validation_rejects_non_parquet_nav_paths_for_path_dist(
    tmp_path: Path, filename: str
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle_file(bundle_dir, "summary.csv")
    _write_bundle_file(bundle_dir, "results.csv")
    _write_bundle_file(bundle_dir, filename)

    missing = validate_mc_viz_bundle_requirements(bundle_dir, "path_dist")

    assert missing == ["nav_paths.parquet"]


def test_bundle_validation_deduplicates_requirements_across_charts(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle_file(bundle_dir, "summary.csv")

    missing = validate_mc_viz_bundle_requirements(bundle_dir, ["fan", "risk_return"])

    assert missing == ["results.parquet/results.csv/results.json (one required)"]


def test_bundle_validation_rejects_unknown_chart_identifier(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    with pytest.raises(ValueError, match="Unsupported chart identifier"):
        validate_mc_viz_bundle_requirements(bundle_dir, "fan,unknown")


# ---------------------------------------------------------------------------
# check_png_dependency tests
# ---------------------------------------------------------------------------


def test_check_png_dependency_returns_true_when_kaleido_importable() -> None:
    fake_spec = mock.MagicMock()
    with mock.patch("importlib.util.find_spec", return_value=fake_spec):
        assert check_png_dependency() is True


def test_check_png_dependency_returns_false_when_kaleido_missing() -> None:
    with mock.patch("importlib.util.find_spec", return_value=None):
        assert check_png_dependency() is False


def test_check_png_dependency_returns_false_on_import_error() -> None:
    with mock.patch("importlib.util.find_spec", side_effect=ModuleNotFoundError):
        assert check_png_dependency() is False


# ---------------------------------------------------------------------------
# execute_mc_viz error-path tests
# ---------------------------------------------------------------------------


def test_execute_mc_viz_raises_on_no_output_flags(tmp_path: Path) -> None:
    with pytest.raises(TrendCLIError, match="at least one output flag"):
        execute_mc_viz(tmp_path, tmp_path / "out", "fan", html=False, json=False, png=False)


def test_execute_mc_viz_raises_on_png_without_kaleido(tmp_path: Path) -> None:
    with mock.patch("trend.mc.viz.check_png_dependency", return_value=False):
        with pytest.raises(TrendCLIError, match="pip install kaleido"):
            execute_mc_viz(
                tmp_path, tmp_path / "out", "fan", html=False, json=False, png=True
            )


def test_execute_mc_viz_raises_on_missing_bundle(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(TrendCLIError, match="does not exist"):
        execute_mc_viz(missing, tmp_path / "out", "fan", html=True, json=False, png=False)


def test_trend_cli_error_is_runtime_error() -> None:
    assert issubclass(TrendCLIError, RuntimeError)
