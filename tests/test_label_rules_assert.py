"""Tests for the label_rules_assert guard script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github/scripts/label_rules_assert.py"


def _load_script():
    """Dynamically load the guard script as a module."""
    spec = importlib.util.spec_from_file_location("label_rules_assert", SCRIPT_PATH)
    assert spec and spec.loader, "Unable to load label_rules_assert.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module():
    return _load_script()


@pytest.fixture
def trusted_config(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    trusted = root / "trusted-config"
    (trusted / ".github/scripts").mkdir(parents=True)
    (trusted / ".github/agent-label-rules.json").write_text("{}", encoding="utf-8")
    allowlist = ".github/agent-label-rules.json\n.github/scripts/label_rules_assert.py"
    monkeypatch.setenv("TRUSTED_LABEL_RULE_PATHS", allowlist)
    monkeypatch.chdir(root)
    return trusted


def test_label_rules_assert_passes_with_exact_allowlist(script_module, trusted_config):
    (trusted_config / ".github/scripts/label_rules_assert.py").write_text("pass", encoding="utf-8")

    exit_code = script_module.main()

    assert exit_code == 0


def test_label_rules_assert_fails_when_extra_file_present(script_module, trusted_config):
    (trusted_config / ".github/scripts/label_rules_assert.py").write_text("pass", encoding="utf-8")
    # Create a file that is not part of the allowlist and should trigger failure.
    (trusted_config / ".github/extra.json").write_text("{}", encoding="utf-8")

    exit_code = script_module.main()

    assert exit_code == 1
