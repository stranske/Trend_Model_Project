import importlib
import sys
import types
import platform
from pathlib import Path

import pandas as pd

PKG_PATH = Path(__file__).resolve().parents[1] / "src" / "trend_analysis"
sys.path.insert(0, str(PKG_PATH.parent))
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
