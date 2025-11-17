"""Safety checks preventing legacy packages from leaking into public imports."""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

LEGACY_SEGMENTS = ("Old", "retired")


def _has_legacy_segment(module_name: str) -> bool:
    parts = module_name.split(".")
    legacy_lower = {segment.lower() for segment in LEGACY_SEGMENTS}
    return any(part.lower() in legacy_lower for part in parts)


@pytest.mark.parametrize(
    "package_name",
    [
        "trend_model",
        "trend_analysis",
        "trend_portfolio_app",
    ],
)
def test_import_does_not_register_legacy_modules(package_name: str) -> None:
    """Importing supported packages must not implicitly load legacy modules."""

    importlib.invalidate_caches()
    importlib.import_module(package_name)

    loaded_names = [
        name
        for name in sys.modules
        if name.startswith(f"{package_name}.") and _has_legacy_segment(name)
    ]

    assert not loaded_names, (
        "Legacy modules should never be importable; "
        f"found unexpected modules when importing {package_name}: {loaded_names}"
    )


@pytest.mark.parametrize(
    "package_name",
    [
        "trend_model",
        "trend_analysis",
        "trend_portfolio_app",
    ],
)
def test_pkgutil_cannot_discover_legacy_modules(package_name: str) -> None:
    """Package discovery must not reveal legacy namespaces."""

    package = importlib.import_module(package_name)

    legacy_discovered = [
        found.name
        for found in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}.")
        if _has_legacy_segment(found.name)
    ]

    assert not legacy_discovered, (
        "Legacy modules were discoverable via pkgutil; "
        f"found unexpected entries for {package_name}: {legacy_discovered}"
    )
