"""Keep broad exception handlers from increasing silently.

The counter parses every Python file below ``src/trend_analysis`` and
``streamlit_app`` except files beneath directories named ``archives``,
``retired``, ``.venv``, or ``node_modules``. It counts each
``ast.ExceptHandler`` whose type is either absent (a bare ``except:``) or the
simple name ``Exception`` (``except Exception:``), regardless of whether the
handler later logs or re-raises. Comments, strings, qualified exception names,
and tuple handlers are not counted.

To reproduce a budget, run this module's ``_count_broad_exception_handlers``
helper against the corresponding directory from the repository root.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_AREAS = {
    "src/trend_analysis": REPO_ROOT / "src" / "trend_analysis",
    "streamlit_app": REPO_ROOT / "streamlit_app",
}
EXCLUDED_DIRECTORY_NAMES = frozenset({"archives", "retired", ".venv", "node_modules"})

MAX_BROAD_EXCEPT_SRC = 202
# Raised 63 -> 65 as an explicit decision after #5912 (universe membership wiring).
# Both new handlers were reviewed and neither swallows an error:
#   * streamlit_app/components/universe_membership_input.py - catches only to unlink the
#     partially-written upload, then re-raises the original exception.
#   * streamlit_app/pages/1_Data.py - UI error boundary that surfaces the failure to the
#     user via st.error() instead of a raw Streamlit traceback.
# This counter deliberately counts logging and re-raising handlers too (see module
# docstring), so a non-swallowing handler still consumes budget.
MAX_BROAD_EXCEPT_APP = 65


def _count_broad_exception_handlers(root: Path) -> int:
    count = 0
    for path in root.rglob("*.py"):
        if EXCLUDED_DIRECTORY_NAMES.intersection(path.relative_to(root).parts):
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += sum(
            isinstance(node, ast.ExceptHandler)
            and (
                node.type is None or isinstance(node.type, ast.Name) and node.type.id == "Exception"
            )
            for node in ast.walk(tree)
        )
    return count


def test_broad_exception_budget_not_exceeded() -> None:
    budgets = {
        "src/trend_analysis": MAX_BROAD_EXCEPT_SRC,
        "streamlit_app": MAX_BROAD_EXCEPT_APP,
    }
    overages = []

    for area, root in SOURCE_AREAS.items():
        current = _count_broad_exception_handlers(root)
        budget = budgets[area]
        if current > budget:
            overages.append(f"{area}: current count {current}, budget {budget}")

    assert not overages, (
        "Broad-exception budget exceeded:\n- "
        + "\n- ".join(overages)
        + "\nLower the budget when handlers are removed, but do not raise it without an "
        "explicit decision."
    )
