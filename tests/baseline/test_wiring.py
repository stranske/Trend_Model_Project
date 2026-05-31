"""Tier 2: wiring checks -- a UI-exposed flag must actually change output.

This is the direct test for the original bug class: a control exists but flipping
it does nothing because it was never wired to the logic. We test at the logic
layer (run the pipeline with the flag on vs off and require the output to differ).

The Streamlit *render* layer (driving the real pages through st.testing AppTest)
lives in ``test_streamlit_smoke.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_catalog
from .harness import run_scenario

_TOGGLES = load_catalog()["toggles"]
_TOG_IDS = [t["id"] for t in _TOGGLES]


def _outputs_differ(a, b, tol: float = 1e-8) -> bool:
    wa, wb = a.fund_weights, b.fund_weights
    if list(wa.index) != list(wb.index):
        return True
    if len(wa) and float(np.max(np.abs(wa.to_numpy() - wb.to_numpy()))) > tol:
        return True
    da, db = a.derived(), b.derived()
    return any(
        abs(da[k] - db[k]) > tol
        for k in ("ann_return", "ann_vol", "max_drawdown")
        if np.isfinite(da[k]) and np.isfinite(db[k])
    )


@pytest.mark.parametrize("tog", _TOGGLES, ids=_TOG_IDS)
def test_flag_is_wired(tog):
    flag = tog["flag"]
    out_on = run_scenario("config/demo.yml", {flag: True})
    out_off = run_scenario("config/demo.yml", {flag: False})
    differ = _outputs_differ(out_on, out_off)
    msg = f"{flag} produced identical output on/off -> appears UNWIRED"
    if tog.get("enforce", True):
        assert differ, msg
    elif not differ:
        pytest.skip("[report-only] " + msg)
