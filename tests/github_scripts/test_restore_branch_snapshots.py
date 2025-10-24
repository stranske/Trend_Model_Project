from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import restore_branch_snapshots as rbs


def test_select_latest_filters_by_prefix_and_run() -> None:
    artifacts = [
        rbs.Artifact(id=1, name="gate-branch-protection-1", created_at="2024-01-01T00:00:00Z", expired=False, workflow_run_id=10),
        rbs.Artifact(id=2, name="gate-branch-protection-2", created_at="2024-01-02T00:00:00Z", expired=False, workflow_run_id=20),
        rbs.Artifact(id=3, name="other-artifact", created_at="2024-01-03T00:00:00Z", expired=False, workflow_run_id=30),
    ]
    result = rbs._select_latest(artifacts, "gate-branch-protection-", current_run_id="20")
    assert result is not None
    assert result.id == 1


def test_copy_json_requires_content(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "dest"
    with pytest.raises(rbs.RestoreError):
        rbs._copy_json(source, destination)


def test_restore_previous_snapshots_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "gate-branch-protection"
    snapshot_dir.mkdir()

    artifacts = [
        rbs.Artifact(id=5, name="gate-branch-protection-123", created_at="2024-01-05T00:00:00Z", expired=False, workflow_run_id=99)
    ]

    monkeypatch.setattr(rbs, "_collect_artifacts", lambda session, repo, token: artifacts)

    def fake_download(session, repo, token, artifact_id, destination):
        destination.write_bytes(b"dummy")

    def fake_extract(archive, target_dir):
        target = target_dir / snapshot_dir.name
        target.mkdir(parents=True, exist_ok=True)
        (target / "snapshot.json").write_text("{}", encoding="utf-8")

    def fake_copy(source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "snapshot.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(rbs, "_download_artifact", fake_download)
    monkeypatch.setattr(rbs, "_extract_zip", fake_extract)
    monkeypatch.setattr(rbs, "_copy_json", fake_copy)

    restored = rbs.restore_previous_snapshots(
        repo="owner/repo",
        token="token",
        snapshot_dir=snapshot_dir,
        run_id="101",
    )

    assert restored is True
    previous_dir = snapshot_dir / "previous"
    assert (previous_dir / "snapshot.json").exists()
