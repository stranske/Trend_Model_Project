"""Coverage-focused tests for ``trend_analysis.__init__`` internals."""

import importlib
import os
import sys
from types import ModuleType

import pytest


def test_import_does_not_mutate_dataclasses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package never installs the retired dataclasses compatibility patch."""
    import dataclasses

    import trend_analysis

    original_is_type = getattr(dataclasses, "_is_type", None)
    for marker in ("_trend_patched", "_trend_model_patched"):
        monkeypatch.delattr(dataclasses, marker, raising=False)
    monkeypatch.delitem(trend_analysis.__dict__, "_SAFE_IS_TYPE", raising=False)

    importlib.reload(trend_analysis)

    assert getattr(dataclasses, "_is_type", None) is original_is_type
    assert not hasattr(dataclasses, "_trend_patched")
    assert not hasattr(dataclasses, "_trend_model_patched")
    assert not hasattr(trend_analysis, "_SAFE_IS_TYPE")
    assert not hasattr(trend_analysis, "_patch_dataclasses_module_guard")


def test_ensure_registered_restores_module(monkeypatch):
    import trend_analysis as ta

    placeholder = ModuleType("trend_analysis")
    monkeypatch.setitem(sys.modules, "trend_analysis", placeholder)

    ta._ensure_registered()

    assert sys.modules["trend_analysis"] is ta


def test_getattr_lazy_import(monkeypatch):
    import trend_analysis as ta

    dummy_mod = ModuleType("trend_analysis.dummy_lazy")
    monkeypatch.setitem(sys.modules, "trend_analysis.dummy_lazy", dummy_mod)
    monkeypatch.setitem(ta._LAZY_SUBMODULES, "dummy_attr", "trend_analysis.dummy_lazy")
    monkeypatch.delitem(ta.__dict__, "dummy_attr", raising=False)

    result = getattr(ta, "dummy_attr")

    assert result is dummy_mod
    assert ta.dummy_attr is dummy_mod


def test_getattr_unknown_raises_attribute_error():
    import trend_analysis as ta

    with pytest.raises(AttributeError):
        getattr(ta, "non_existent_submodule")


def test_spec_proxy_triggers_registration(monkeypatch):
    import trend_analysis as ta

    class Spec:
        name = "trend_analysis"

    proxy = ta._SpecProxy(Spec())

    monkeypatch.setitem(sys.modules, "trend_analysis", ModuleType("trend_analysis"))

    assert proxy.name == "trend_analysis"
    assert sys.modules["trend_analysis"] is ta


def test_configure_matplotlib_config_dir_sets_env(monkeypatch, tmp_path):
    import trend_analysis as ta

    target = tmp_path / "mpl_config"
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("TREND_MPLCONFIGDIR_MODE", raising=False)
    monkeypatch.setenv("TREND_MPLCONFIGDIR", str(target))

    result = ta.configure_matplotlib_config_dir()

    assert result == target
    assert os.environ["MPLCONFIGDIR"] == str(target)
    assert target.exists()


def test_configure_matplotlib_config_dir_respects_off_mode(monkeypatch, tmp_path):
    import trend_analysis as ta

    target = tmp_path / "mpl_disabled"
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setenv("TREND_MPLCONFIGDIR_MODE", "off")
    monkeypatch.setenv("TREND_MPLCONFIGDIR", str(target))

    result = ta.configure_matplotlib_config_dir()

    assert result is None
    assert "MPLCONFIGDIR" not in os.environ
    assert not target.exists()


def test_configure_matplotlib_config_dir_keeps_existing(monkeypatch, tmp_path):
    import trend_analysis as ta

    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(existing))
    monkeypatch.setenv("TREND_MPLCONFIGDIR", str(tmp_path / "ignored"))

    result = ta.configure_matplotlib_config_dir()

    assert result == existing
    assert os.environ["MPLCONFIGDIR"] == str(existing)
