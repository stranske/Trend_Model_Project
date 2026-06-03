"""Guard that ``trend_analysis.rebalancing`` is the package, not a stale shim.

Historically a ``rebalancing.py`` back-compat shim coexisted with the
``rebalancing/`` package. The package shadows the module on disk, so the shim
was dead code (A24). These tests pin the canonical layout: the import must
resolve to the package ``__init__`` and the dead shim file must not exist.
"""

from pathlib import Path

import trend_analysis.rebalancing as reb


def test_rebalancing_resolves_to_package() -> None:
    """``import trend_analysis.rebalancing`` must load the package __init__."""

    assert reb.__file__.replace("\\", "/").endswith("rebalancing/__init__.py")


def test_dead_shim_module_is_absent() -> None:
    """The shadowed ``rebalancing.py`` shim must not be reintroduced.

    Re-creating the shim file is the A24 deliberate-break gate: because the
    package always wins import resolution, ``__file__`` alone would not catch a
    resurrected shim, so this test fails on the file's mere presence.
    """

    package_init = Path(reb.__file__)
    shim_module = package_init.parent.with_name("rebalancing.py")
    assert not shim_module.exists(), (
        f"dead shim shadowed by package exists: {shim_module}"
    )
