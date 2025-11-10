import sys
from types import ModuleType

import pytest


@pytest.fixture()
def trend_analysis_module():
    sys.modules.pop("trend_analysis", None)
    module = __import__("trend_analysis")
    assert "_SAFE_IS_TYPE" in module.__dict__
    try:
        yield module
    finally:
        sys.modules["trend_analysis"] = module


def test_dataclasses_guard_recreates_missing_module(trend_analysis_module):
    import dataclasses
    import typing

    dataclass_module = "tests.temp_missing_dataclass_module"
    dummy_cls = type("Dummy", (), {"__module__": dataclass_module})

    sys.modules.pop(dataclass_module, None)

    result = dataclasses._is_type(
        "ClassVar",
        dummy_cls,
        typing,
        typing.ClassVar,
        lambda obj, mod: obj is typing.ClassVar,
    )

    assert result is False
    recreated = sys.modules[dataclass_module]
    assert isinstance(recreated, ModuleType)
    assert recreated.__package__ == "tests"


def test_dataclasses_guard_reimports_modules(monkeypatch, trend_analysis_module):
    import dataclasses
    import typing

    module_name = "tmptest_missing.module"
    dummy_cls = type("Dummy", (), {"__module__": module_name})

    sys.modules.pop(module_name, None)

    calls: list[str] = []

    def fake_import(name: str) -> ModuleType:
        calls.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(trend_analysis_module.importlib, "import_module", fake_import)

    result = dataclasses._is_type(
        "ClassVar",
        dummy_cls,
        typing,
        typing.ClassVar,
        lambda obj, mod: obj is typing.ClassVar,
    )

    assert result is False
    recreated = sys.modules[module_name]
    assert isinstance(recreated, ModuleType)
    assert recreated.__package__ == "tmptest_missing"
    assert calls == [module_name]


def test_safe_is_type_reimports_missing_module(monkeypatch, trend_analysis_module):
    import typing

    module_name = "tmptest_branch.module"
    dummy_cls = type("Dummy", (), {"__module__": module_name})
    sys.modules.pop(module_name, None)

    calls: list[str] = []

    def fake_import(name: str) -> ModuleType:
        calls.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(trend_analysis_module.importlib, "import_module", fake_import)

    result = trend_analysis_module._SAFE_IS_TYPE(
        "ClassVar",
        dummy_cls,
        typing,
        typing.ClassVar,
        lambda obj, mod: obj is typing.ClassVar,
    )

    assert result is False
    recreated = sys.modules[module_name]
    assert isinstance(recreated, ModuleType)
    assert recreated.__package__ == "tmptest_branch"
    assert calls == [module_name]


def test_safe_is_type_requires_module_name(trend_analysis_module):
    import typing

    nameless_cls = type("Nameless", (), {"__module__": ""})

    with pytest.raises(AttributeError):
        trend_analysis_module._SAFE_IS_TYPE(
            "ClassVar",
            nameless_cls,
            typing,
            typing.ClassVar,
            lambda obj, mod: obj is typing.ClassVar,
        )


def test_spec_proxy_re_registers_module(trend_analysis_module):
    sys.modules["trend_analysis"] = ModuleType("trend_analysis")

    name = trend_analysis_module.__spec__.name

    assert name == "trend_analysis"
    assert sys.modules["trend_analysis"] is trend_analysis_module


def test_lazy_getattr_imports_requested_module(trend_analysis_module):
    sys.modules.pop("trend_analysis.cli", None)
    trend_analysis_module.__dict__.pop("cli", None)

    cli_module = trend_analysis_module.cli

    assert cli_module is sys.modules["trend_analysis.cli"]
    assert trend_analysis_module.__dict__["cli"] is cli_module
