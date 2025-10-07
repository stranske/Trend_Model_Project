#!/usr/bin/env python3
"""Shared health-response parser for Docker smoke checks.

The script reads the HEALTH_RESPONSE environment variable and exits 0 when the
payload represents a healthy state.  This mirrors the logic used by the CI
smoke workflow and the local docker_smoke.sh helper.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping

SUCCESS_VALUES = {
    "ok",
    "healthy",
    "pass",
    "passed",
    "true",
    "1",
    "up",
    "ready",
    "success",
}
KEYS_TO_CHECK = ("status", "state", "health", "message", "detail")


def _normalize(value: str) -> str:
    return value.strip().lower()


def _is_success(value: object) -> bool:
    if isinstance(value, str):
        return _normalize(value) in SUCCESS_VALUES
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, Mapping):
        for key in KEYS_TO_CHECK:
            if key in value and _is_success(value[key]):
                return True
        return any(_is_success(v) for v in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return any(_is_success(item) for item in value)
    return False


def main() -> int:
    payload = os.environ.get("HEALTH_RESPONSE", "").strip()
    if not payload:
        return 1

    normalized = _normalize(payload)
    if normalized in SUCCESS_VALUES:
        return 0

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return 1

    return 0 if _is_success(data) else 1


if __name__ == "__main__":
    sys.exit(main())
