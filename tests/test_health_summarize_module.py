from __future__ import annotations

from collections.abc import Iterator
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "health_summarize" / "__init__.py"


def _load_health_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("health_summarize_pkg", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load health_summarize module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def summarize() -> Iterator[ModuleType]:
    sys.modules.pop("health_summarize_pkg", None)
    module = _load_health_module()
    try:
        yield module
    finally:
        sys.modules.pop("health_summarize_pkg", None)


def test_read_bool_variants(summarize: ModuleType) -> None:
    assert summarize._read_bool(True) is True
    assert summarize._read_bool(None) is False
    assert summarize._read_bool("true") is True
    assert summarize._read_bool("no") is False
    assert summarize._read_bool("surprise") is True


def test_load_json_handles_missing_and_invalid(summarize: ModuleType, tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert summarize._load_json(missing) is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    assert summarize._load_json(invalid) is None

    valid = tmp_path / "valid.json"
    payload = {"hello": "world"}
    valid.write_text(json.dumps(payload), encoding="utf-8")
    assert summarize._load_json(valid) == payload


def test_doc_url_uses_pr_base_branch(summarize: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://example.com")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

    doc_url = summarize._doc_url()
    assert doc_url == (
        "https://example.com/owner/repo/blob/main/"
        "docs/ci/WORKFLOWS.md#ci-signature-guard-fixtures"
    )


def _write_signature_fixture(
    summarize: ModuleType, tmp_path: Path, jobs: list[dict[str, str]]
) -> tuple[Path, Path]:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    expected_path = tmp_path / "expected.txt"
    expected_path.write_text(
        summarize.build_signature_hash(jobs),
        encoding="utf-8",
    )
    return jobs_path, expected_path


def test_signature_row_success(summarize: ModuleType, tmp_path: Path) -> None:
    jobs = [{"name": "Tests", "step": "pytest", "stack": "runner"}]
    jobs_path, expected_path = _write_signature_fixture(summarize, tmp_path, jobs)

    row = summarize._signature_row(jobs_path, expected_path)
    assert row["check"] == "Health 43 CI Signature Guard"
    assert row["conclusion"] == "success"
    assert "✅" in row["status"]


def test_signature_row_reports_mismatch(summarize: ModuleType, tmp_path: Path) -> None:
    jobs = [{"name": "Lint", "step": "ruff", "stack": "ci"}]
    jobs_path, expected_path = _write_signature_fixture(summarize, tmp_path, jobs)
    expected_path.write_text("different", encoding="utf-8")

    row = summarize._signature_row(jobs_path, expected_path)
    assert row["conclusion"] == "failure"
    assert "Hash drift" in row["status"]


def test_signature_row_handles_invalid_jobs(summarize: ModuleType, tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps({"invalid": True}), encoding="utf-8")
    expected_path = tmp_path / "expected.txt"

    row = summarize._signature_row(jobs_path, expected_path)
    assert row["conclusion"] == "failure"
    assert "Fixture unreadable" in row["status"]


def test_signature_row_without_expected_fixture(
    summarize: ModuleType, tmp_path: Path
) -> None:
    jobs = [{"name": "Lint", "step": "ruff", "stack": "ci"}]
    jobs_path, expected_path = _write_signature_fixture(summarize, tmp_path, jobs)
    expected_path.unlink()

    row = summarize._signature_row(jobs_path, expected_path)
    assert row["status"].startswith("✅ Computed")
    assert row["details"] == "Signature generated from workflow jobs."


@pytest.mark.parametrize(
    "section, expected",
    [
        ("foo", ["foo"]),
        (["a", "", "b"], ["a", "b"]),
        ({"contexts": ["x", "y"]}, ["x", "y"]),
        ({}, []),
        (None, []),
    ],
)
def test_extract_contexts_variants(
    summarize: ModuleType, section: object, expected: list[str]
) -> None:
    assert summarize._extract_contexts(section) == expected


def test_format_bool_handles_unknown(summarize: ModuleType) -> None:
    assert summarize._format_bool(None) == "❔ Unknown"
    assert summarize._format_bool(True) == "✅ True"
    assert summarize._format_bool(False) == "❌ False"


def test_format_delta_reports_changes(summarize: ModuleType) -> None:
    current = {"current": {"contexts": ["lint", "tests"], "strict": True}}
    previous = {"after": {"contexts": ["lint"], "strict": False}}

    delta = summarize._format_delta(current, previous)
    assert "+tests" in delta
    assert "Require up to date" in delta


def test_format_delta_handles_non_dict_current(summarize: ModuleType) -> None:
    delta = summarize._format_delta({"current": None}, {"current": {}})
    assert delta == "No changes"


def test_select_previous_section_handles_non_dict(summarize: ModuleType) -> None:
    assert summarize._select_previous_section(None) == {}
    assert summarize._select_previous_section({}) == {}


def test_snapshot_detail_handles_missing(summarize: ModuleType) -> None:
    detail, severity = summarize._snapshot_detail(
        "Verification",
        None,
        None,
        has_token=False,
    )
    assert "Observer mode" in detail
    assert severity == "warning"


def test_snapshot_detail_warns_on_changes(
    summarize: ModuleType, tmp_path: Path
) -> None:
    current = {
        "changes_required": True,
        "current": {"contexts": ["lint"], "strict": True},
        "changes_applied": True,
        "strict_unknown": True,
    }
    previous = {"after": {"contexts": ["lint", "tests"], "strict": False}}

    detail, severity = summarize._snapshot_detail(
        "Enforcement", current, previous, has_token=True
    )
    assert "Changes required" in detail
    assert "Δ" in detail
    assert severity == "warning"


def test_snapshot_detail_handles_error(summarize: ModuleType) -> None:
    detail, severity = summarize._snapshot_detail(
        "Enforcement", {"error": "boom"}, None, has_token=True
    )
    assert detail.startswith("❌ Enforcement")
    assert severity == "failure"


def test_snapshot_detail_includes_require_strict(summarize: ModuleType) -> None:
    snapshot = {
        "require_strict": True,
        "current": {"contexts": ["lint"], "strict": True},
        "after": {"contexts": ["lint"], "strict": False},
    }
    detail, severity = summarize._snapshot_detail(
        "Verification", snapshot, None, has_token=True
    )
    assert "Require up to date" in detail
    assert severity == "success"


def test_branch_row_with_and_without_snapshots(
    summarize: ModuleType, tmp_path: Path
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "previous").mkdir()
    (snapshot_dir / "enforcement.json").write_text(
        json.dumps({"changes_required": False, "current": {"strict": True}}),
        encoding="utf-8",
    )
    (snapshot_dir / "verification.json").write_text(
        json.dumps({"changes_required": True, "current": {"strict": True}}),
        encoding="utf-8",
    )

    row = summarize._branch_row(snapshot_dir, has_token=True)
    assert row["check"] == "Health 44 Gate Branch Protection"
    assert "Branch protection" in row["status"]

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_row = summarize._branch_row(empty_dir, has_token=False)
    assert "Observer mode" in empty_row["details"]
    assert empty_row["conclusion"] == "warning"


def test_combine_severity_prefers_highest(summarize: ModuleType) -> None:
    assert summarize._combine_severity(["success", "warning"]) == "warning"
    assert summarize._combine_severity(["info", "failure"]) == "failure"


def test_write_json_and_summary(summarize: ModuleType, tmp_path: Path) -> None:
    rows = [{"check": "C", "status": "S", "details": "D"}]
    json_target = tmp_path / "summary" / "rows.json"
    md_target = tmp_path / "summary.md"

    summarize._write_json(json_target, rows)
    summarize._write_summary(md_target, rows)

    assert json.loads(json_target.read_text(encoding="utf-8")) == rows
    text = md_target.read_text(encoding="utf-8")
    assert "Health guardrail" in text
    assert "| C |" in text


def test_write_summary_ignores_empty_rows(summarize: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "summary.md"
    summarize._write_summary(target, [])
    assert not target.exists()


def test_main_executes_end_to_end(
    summarize: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [{"name": "Tests", "step": "pytest", "stack": "trace"}]
    jobs_path, expected_path = _write_signature_fixture(summarize, tmp_path, jobs)

    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "enforcement.json").write_text(
        json.dumps({"changes_required": False, "current": {"strict": True}}),
        encoding="utf-8",
    )

    json_output = tmp_path / "rows.json"
    md_output = tmp_path / "summary.md"

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")

    code = summarize.main(
        [
            "--signature-jobs",
            str(jobs_path),
            "--signature-expected",
            str(expected_path),
            "--snapshot-dir",
            str(snapshot_dir),
            "--has-enforce-token",
            "true",
            "--write-json",
            str(json_output),
            "--write-summary",
            str(md_output),
        ]
    )

    assert code == 0
    assert json_output.exists()
    assert md_output.exists()


def test_main_handles_empty_arguments(
    summarize: ModuleType, tmp_path: Path
) -> None:
    json_output = tmp_path / "rows.json"
    md_output = tmp_path / "summary.md"

    code = summarize.main(
        [
            "--write-json",
            str(json_output),
            "--write-summary",
            str(md_output),
        ]
    )

    assert code == 0
    assert json.loads(json_output.read_text(encoding="utf-8")) == []
    assert not md_output.exists()
