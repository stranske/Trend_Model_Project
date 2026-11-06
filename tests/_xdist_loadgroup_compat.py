"""Keep ``--dist=loadgroup`` honouring ``xdist_group`` under pytest >= 9.2.

pytest-xdist (<= 3.8.0) implements loadgroup by suffixing each grouped item's
id with ``@<group>`` via ``item._nodeid = ...``.  pytest 9.2 moved the node id
to a structured ``item._id`` and stopped reading ``_nodeid``, so the suffix is
silently dropped and grouped tests (including every ``@pytest.mark.serial``
test) are spread across workers.  Tests that share repository state, such as
``tests/golden/test_demo.py`` wiping ``demo/exports``, then race each other.

This hook runs after xdist's and copies the stranded suffix into ``_id``.  It
is a no-op wherever ``nodeid`` already reflects ``_nodeid`` (pytest <= 9.1, or
a future xdist that sets the id itself), so it can be deleted once xdist
supports the structured node id.
"""

from __future__ import annotations

import pytest

PLUGIN_NAME = "trend-xdist-loadgroup-compat"


def restore_group_suffixes(items: list[pytest.Item]) -> int:
    """Apply xdist's ``@group`` id suffix where pytest ignored it; return the count."""

    try:
        from _pytest.nodeid import NodeId
    except ImportError:  # pytest <= 9.1 reads ``_nodeid`` directly
        return 0

    restored = 0
    for item in items:
        stranded = item.__dict__.get("_nodeid")
        if isinstance(stranded, str) and stranded != item.nodeid:
            item._id = NodeId.parse(stranded)
            restored += 1
    return restored


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    restore_group_suffixes(items)
