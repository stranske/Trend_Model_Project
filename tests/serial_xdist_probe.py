"""Minimal serial-marked test for xdist loadgroup subprocess probes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _write_probe(request: pytest.FixtureRequest) -> None:
    base = Path(os.environ["SERIAL_PROBE_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    safe = request.node.name.replace("/", "_")
    out = base / f"{safe}.json"
    payload = {
        "worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
        "nodeid": request.node.nodeid,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.serial
def test_serial_probe_a(request: pytest.FixtureRequest) -> None:
    _write_probe(request)


@pytest.mark.serial
def test_serial_probe_b(request: pytest.FixtureRequest) -> None:
    _write_probe(request)
