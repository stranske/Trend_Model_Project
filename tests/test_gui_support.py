"""Tests for light-weight GUI support utilities."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from trend_analysis.gui import plugins as plugin_module
from trend_analysis.gui import utils as gui_utils


class DummyPlugin:
    """Minimal plugin type used for registration tests."""


class AnotherPlugin:
    """Second plugin type used to verify de-duplication."""


@pytest.fixture(autouse=True)
def reset_plugin_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test starts with a clean plugin registry."""

    monkeypatch.setattr(plugin_module, "_PLUGIN_REGISTRY", [])


def test_register_plugin_deduplicates() -> None:
    """Plugins should only be registered once."""

    plugin_module.register_plugin(DummyPlugin)
    plugin_module.register_plugin(DummyPlugin)
    plugin_module.register_plugin(AnotherPlugin)

    assert list(plugin_module.iter_plugins()) == [DummyPlugin, AnotherPlugin]


def test_discover_plugins_loads_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entry-point discovery must load and register plugins."""

    def fake_entry_points(*, group: str):
        assert group == "trend_analysis.gui_plugins"
        return [
            SimpleNamespace(load=lambda: DummyPlugin),
            SimpleNamespace(load=lambda: AnotherPlugin),
        ]

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

    plugin_module.discover_plugins()

    assert list(plugin_module.iter_plugins()) == [DummyPlugin, AnotherPlugin]


def test_debounce_coalesces_calls() -> None:
    """Only the final call within the debounce window should run."""

    calls: list[tuple[tuple[int, ...], dict[str, int]]] = []

    def sync_callback(*args: int, **kwargs: int) -> None:
        calls.append((args, kwargs))

    async def runner() -> None:
        debounced = gui_utils.debounce(wait_ms=20)(sync_callback)
        await debounced(1, value=1)
        await debounced(2, value=2)
        await asyncio.sleep(0.05)

    asyncio.run(runner())

    assert calls == [((2,), {"value": 2})]


def test_debounce_awaits_coroutine() -> None:
    """If the wrapped function returns a coroutine it should be awaited."""

    observed: list[int] = []

    async def async_callback(value: int) -> None:
        await asyncio.sleep(0)
        observed.append(value)

    async def runner() -> None:
        debounced = gui_utils.debounce(wait_ms=5)(async_callback)
        await debounced(10)
        await asyncio.sleep(0.02)

    asyncio.run(runner())

    assert observed == [10]


def test_list_builtin_cfgs_returns_sorted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in configuration listing should return sorted YAML stem names."""

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "zeta.yml").write_text("z: 1")
    (cfg_dir / "alpha.yml").write_text("a: 1")
    (cfg_dir / "notes.txt").write_text("ignore")

    monkeypatch.setattr(gui_utils, "_find_config_directory", lambda: cfg_dir)

    assert gui_utils.list_builtin_cfgs() == ["alpha", "zeta"]
