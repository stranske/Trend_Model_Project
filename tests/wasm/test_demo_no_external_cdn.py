"""Offline-runtime guards for the stlite browser demo."""

from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_HTML = REPO_ROOT / "demo" / "wasm" / "index.html"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_wasm_demo_assets.py"


def _runtime_refs(html: str) -> list[str]:
    refs: list[str] = []
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"<script\b[^>]*\bsrc=[\"'](?P<url>https?://[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE,
        )
    )
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"<link\b[^>]*\bhref=[\"'](?P<url>https?://[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE,
        )
    )
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"\bimport\b[^;]*?\bfrom\s+[\"'](?P<url>https?://[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    return refs


def test_demo_html_has_no_external_runtime_ref() -> None:
    html = DEMO_HTML.read_text(encoding="utf-8")
    assert _runtime_refs(html) == []
    assert "cdn.jsdelivr.net" not in html
    assert "./vendor/stlite@0.79.4/stlite.js" in html
    assert "./vendor/pyodide-0.27.2/pyodide.mjs" in html


def test_demo_static_asset_graph_is_locally_reachable() -> None:
    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--repo-root", str(REPO_ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wasm demo asset smoke passed" in result.stdout
