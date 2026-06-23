#!/usr/bin/env python3
"""Smoke-check the static asset graph for the stlite/Pyodide browser demo.

This probe intentionally does not require a browser. It serves the demo through
the same URL layout expected by ``demo/wasm/index.html`` and verifies that the
HTML, manifest, vendored runtime files, and application source files are all
reachable locally.
"""

from __future__ import annotations

import argparse
import html.parser
import json
import mimetypes
import posixpath
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

import build_wasm_demo

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_EXTERNAL_MARKERS = (
    "cdn.jsdelivr.net",
    "unpkg.com",
    "esm.sh",
    "skypack.dev",
)


class _AssetRefParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "script" and attr_map.get("src"):
            self.refs.append(attr_map["src"] or "")
        if tag == "link" and attr_map.get("href"):
            self.refs.append(attr_map["href"] or "")


class _DemoHandler(BaseHTTPRequestHandler):
    repo_root = REPO_ROOT

    def do_HEAD(self) -> None:
        self._send_file(send_body=False)

    def do_GET(self) -> None:
        self._send_file(send_body=True)

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_file(self, *, send_body: bool) -> None:
        target = self._target_path()
        if target is None or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        if send_body:
            with target.open("rb") as fh:
                self.wfile.write(fh.read())

    def _target_path(self) -> Path | None:
        parsed = urllib.parse.urlparse(self.path)
        path = posixpath.normpath(urllib.parse.unquote(parsed.path)).lstrip("/")
        if path in ("", "."):
            path = "index.html"

        demo_root = self.repo_root / "demo" / "wasm"
        if path in {"index.html", "manifest.json", "README.md"}:
            candidate = demo_root / path
        elif path.startswith("vendor/"):
            candidate = demo_root / path
        elif path.startswith("app/"):
            candidate = self.repo_root / path[len("app/") :]
        else:
            return None

        try:
            resolved = candidate.resolve()
            allowed = (demo_root.resolve(), self.repo_root.resolve())
            if not any(resolved == root or root in resolved.parents for root in allowed):
                return None
        except OSError:
            return None
        return resolved


def _relative_refs(html: str) -> list[str]:
    parser = _AssetRefParser()
    parser.feed(html)
    refs = list(parser.refs)
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"\bimport\b[^;]*?\bfrom\s+[\"'](?P<url>[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"\bnew\s+URL\(\s*[\"'](?P<url>[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE,
        )
    )
    return [ref for ref in refs if ref and not ref.startswith(("data:", "blob:"))]


def _external_refs(refs: Iterable[str]) -> list[str]:
    return [ref for ref in refs if urllib.parse.urlparse(ref).scheme in {"http", "https"}]


def _head(url: str) -> None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != HTTPStatus.OK:
            raise RuntimeError(f"{url} returned HTTP {response.status}")


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        if response.status != HTTPStatus.OK:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return response.read().decode("utf-8")


def _check_manifest_fresh(repo_root: Path) -> None:
    missing = build_wasm_demo.missing_runtime_files(repo_root)
    if missing:
        details = "\n".join(f"- demo/wasm/vendor/{rel}" for rel in missing)
        raise RuntimeError(f"vendored wasm runtime is incomplete:\n{details}")

    manifest_path = repo_root / "demo" / "wasm" / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest missing: {manifest_path}")
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    fresh = build_wasm_demo.build_manifest(repo_root)
    if existing != fresh:
        raise RuntimeError("manifest is stale; run `python scripts/build_wasm_demo.py`")


def smoke(repo_root: Path) -> list[str]:
    _check_manifest_fresh(repo_root)
    handler = type("DemoHandler", (_DemoHandler,), {"repo_root": repo_root})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    checked: list[str] = []
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/"
        html = _get_text(base_url)
        checked.append(base_url)

        if any(marker in html for marker in FORBIDDEN_EXTERNAL_MARKERS):
            raise RuntimeError("demo HTML still references a forbidden external CDN")

        refs = _relative_refs(html)
        external = _external_refs(refs)
        if external:
            raise RuntimeError("demo HTML has external runtime refs: " + ", ".join(external))

        for ref in refs:
            url = urllib.parse.urljoin(base_url, ref)
            _head(url)
            checked.append(url)

        manifest_url = urllib.parse.urljoin(base_url, "manifest.json")
        manifest = json.loads(_get_text(manifest_url))
        checked.append(manifest_url)

        for rel in build_wasm_demo.VENDORED_RUNTIME_FILES:
            url = urllib.parse.urljoin(base_url, f"vendor/{rel}")
            _head(url)
            checked.append(url)

        for rel in manifest["files"]:
            url = urllib.parse.urljoin(base_url, f"app/{rel}")
            _head(url)
            checked.append(url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to serve; defaults to the script's checkout.",
    )
    args = parser.parse_args(argv)

    try:
        checked = smoke(args.repo_root.resolve())
    except (OSError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"wasm demo asset smoke failed: {exc}", file=sys.stderr)
        return 1

    print(f"wasm demo asset smoke passed: checked {len(checked)} local URLs")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
