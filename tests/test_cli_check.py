import importlib
import platform
import sys
import types
from pathlib import Path

import pandas as pd

PKG_PATH = Path(__file__).resolve().parents[1] / "src" / "trend_analysis"
pkg = types.ModuleType("trend_analysis")
pkg.__path__ = [str(PKG_PATH)]
sys.modules.setdefault("trend_analysis", pkg)
cli = importlib.import_module("trend_analysis.cli")


def test_check_environment_ok(tmp_path, capsys):
    lock = tmp_path / "req.lock"
    lock.write_text(f"pandas=={pd.__version__}\n")
    ret = cli.check_environment(lock)
    out = capsys.readouterr().out
    assert ret == 0
    assert platform.python_version() in out
    assert f"pandas {pd.__version__}" in out


def test_check_environment_mismatch(tmp_path, capsys):
    lock = tmp_path / "req.lock"
    lock.write_text("some-missing-package==0.0.1\n")
    ret = cli.check_environment(lock)
    out = capsys.readouterr().out
    assert ret == 1
    assert "some-missing-package" in out
    assert "expected 0.0.1" in out


def test_check_environment_missing_file(tmp_path, capsys):
    lock = tmp_path / "missing.lock"
    ret = cli.check_environment(lock)
    out = capsys.readouterr().out
    assert ret == 1
    assert f"Lock file not found: {lock}" in out


def test_check_environment_skips_non_pinned_lines(tmp_path, capsys):
    lock = tmp_path / "req.lock"
    lock.write_text("# comment\nwheel>=0.0\n\n", encoding="utf-8")

    ret = cli.check_environment(lock)
    out = capsys.readouterr().out

    assert ret == 0
    assert "wheel" not in out
    assert "All packages match" in out


def test_main_check_flag_without_subcommand(monkeypatch):
    called: dict[str, bool] = {}

    def fake_check(lock_path=None):
        called["ran"] = True
        return 0

    monkeypatch.setattr(cli, "check_environment", fake_check)

    rc = cli.main(["--check"])

    assert rc == 0
    assert called == {"ran": True}
