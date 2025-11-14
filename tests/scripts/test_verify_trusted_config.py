from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from scripts import verify_trusted_config as vtc


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRUSTED_CONFIG_PATHS", raising=False)
    monkeypatch.delenv("TARGET_RUN_ID", raising=False)


def test_main_requires_allowed_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    result = vtc.main()

    captured = capsys.readouterr()
    assert result == 1
    assert "No trusted config paths" in captured.err


def test_main_reports_missing_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("TRUSTED_CONFIG_PATHS", "config.yml")
    monkeypatch.chdir(tmp_path)

    result = vtc.main()

    captured = capsys.readouterr()
    assert result == 1
    assert "checkout missing" in captured.err


def test_main_reports_missing_and_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "trusted-config"
    root.mkdir()
    allowed = {"primary.yml", "nested/allowed.yml"}
    for rel in allowed:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
    extra_file = root / "extra.yml"
    extra_file.write_text("extra", encoding="utf-8")
    monkeypatch.setenv("TRUSTED_CONFIG_PATHS", "\n".join(sorted(allowed)))

    result = vtc.main()

    captured = capsys.readouterr()
    assert result == 1
    assert "Missing required" not in captured.err
    assert "Unexpected file" in captured.err


def test_main_handles_missing_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "trusted-config"
    root.mkdir()
    (root / "primary.yml").write_text("ok", encoding="utf-8")
    monkeypatch.setenv("TRUSTED_CONFIG_PATHS", "primary.yml\nmissing.yml")

    result = vtc.main()

    captured = capsys.readouterr()
    assert result == 1
    assert "Missing required" in captured.err
    assert "missing.yml" in captured.err


def test_main_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "trusted-config"
    (root / "nested").mkdir(parents=True)
    allowed = ["primary.yml", "nested/allowed.yml"]
    for rel in allowed:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "ignored").write_text("noop", encoding="utf-8")
    monkeypatch.setenv("TRUSTED_CONFIG_PATHS", "\n".join(allowed))

    result = vtc.main()

    captured = capsys.readouterr()
    assert result == 0
    assert "OK" in captured.out
    assert captured.err == ""


def test_main_ignores_non_relative_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "trusted-config"
    root.mkdir()
    (root / "primary.yml").write_text("ok", encoding="utf-8")
    monkeypatch.setenv("TRUSTED_CONFIG_PATHS", "primary.yml")

    calls: list[str] = []

    class FakePath:
        def is_file(self) -> bool:
            return True

        def relative_to(self, other: Path) -> Path:
            calls.append("relative_to")
            raise ValueError("outside root")

    real_rglob = Path.rglob

    root_resolved = root.resolve()

    def fake_rglob(self: Path, pattern: str):
        if self.resolve() == root_resolved:
            yield FakePath()
            yield from real_rglob(self, pattern)
        else:
            yield from real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", fake_rglob)

    result = vtc.main()

    captured = capsys.readouterr()
    assert result == 0
    assert "OK" in captured.out
    assert calls == ["relative_to"]


def test_module_entrypoint_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRUSTED_CONFIG_PATHS", raising=False)

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(Path(vtc.__file__), run_name="__main__")

    assert exc.value.code == 1
