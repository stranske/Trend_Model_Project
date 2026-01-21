from __future__ import annotations

from pathlib import Path

from scripts.inventory_pipeline_patch_tests import _collect_patch_hits


def _write_case(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "case.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_collect_patch_hits_detects_monkeypatch_alias(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "\n".join(
            [
                "import trend_analysis.pipeline as pipeline",
                "",
                "def test_case(monkeypatch):",
                "    monkeypatch.setattr(pipeline, \"load_csv\", lambda *a, **k: None)",
            ]
        ),
    )

    hits = _collect_patch_hits(path)

    assert len(hits) == 1
    assert hits[0].kind == "monkeypatch.setattr"
    assert "monkeypatch.setattr(pipeline" in hits[0].snippet


def test_collect_patch_hits_detects_trend_package_attr(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "\n".join(
            [
                "import trend_analysis",
                "",
                "def test_case(monkeypatch):",
                "    monkeypatch.setattr(trend_analysis.pipeline, \"load_csv\", lambda: None)",
            ]
        ),
    )

    hits = _collect_patch_hits(path)

    assert len(hits) == 1
    assert hits[0].kind == "monkeypatch.setattr"
    assert "trend_analysis.pipeline" in hits[0].snippet


def test_collect_patch_hits_detects_dynamic_import_alias(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "\n".join(
            [
                "import importlib",
                "",
                "pipeline = importlib.import_module(\"trend_analysis.pipeline\")",
                "",
                "def test_case(monkeypatch):",
                "    monkeypatch.setattr(pipeline, \"load_csv\", lambda: None)",
            ]
        ),
    )

    hits = _collect_patch_hits(path)

    assert len(hits) == 1
    assert hits[0].kind == "monkeypatch.setattr"
    assert "monkeypatch.setattr(pipeline" in hits[0].snippet


def test_collect_patch_hits_detects_patch_string(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "\n".join(
            [
                "from unittest.mock import patch",
                "",
                "def test_case():",
                "    with patch(\"trend_analysis.pipeline.load_csv\"):",
                "        pass",
            ]
        ),
    )

    hits = _collect_patch_hits(path)

    assert len(hits) == 1
    assert hits[0].kind == "patch"
    assert "trend_analysis.pipeline.load_csv" in hits[0].snippet
