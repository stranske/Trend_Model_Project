#!/usr/bin/env python3
"""Assemble the stlite/Pyodide browser-demo manifest for issue #5343.

The browser demo (``demo/wasm/index.html``) runs the real Streamlit app in the
browser via `stlite <https://github.com/whitphx/stlite>`_ / Pyodide. Rather than
committing a duplicated copy of the application source, this build step produces
a ``manifest.json`` describing:

* the Streamlit entrypoint,
* the repo-relative source files the demo must load into the in-browser
  filesystem (the ``streamlit_app`` package, the ``trend_analysis`` engine, and
  the bundled synthetic demo data), and
* the per-profile Python requirement sets.

``index.html`` fetches the manifest and each listed file (the deploy publishes
this repository subset next to the page), reconstructs the virtual filesystem,
and mounts stlite. Keeping the manifest generated -- instead of a hand-edited
static page -- is what makes the artifact actually run the deterministic engine
(an explicit non-goal of the issue is "a static page that cannot run the actual
deterministic engine").

Usage::

    python scripts/build_wasm_demo.py            # writes demo/wasm/manifest.json
    python scripts/build_wasm_demo.py --check     # verify the manifest is fresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "demo" / "wasm" / "manifest.json"

#: Streamlit entrypoint, repo-relative.
ENTRYPOINT = "streamlit_app/app.py"

#: Source directories whose ``*.py`` files are loaded into the browser FS.
SOURCE_DIRS = (
    "streamlit_app",
    "src/trend_analysis",
)

#: Bundled synthetic data shipped with the demo (presentation-safe default).
DATA_FILES = (
    "demo/demo_returns.csv",
)

#: Python requirements installed under Pyodide, per runtime profile. The
#: presentation-safe set intentionally omits LangChain so the default load is
#: lean and has no LLM dependency footprint; public_llm_demo adds the LangChain
#: stack pinned in ``pyproject.toml``.
REQUIREMENTS = {
    "presentation_safe": [
        "streamlit==1.57.0",
        "numpy==2.4.6",
        "pandas==3.0.3",
    ],
    "public_llm_demo": [
        "streamlit==1.57.0",
        "numpy==2.4.6",
        "pandas==3.0.3",
        "langchain>=1.2,<1.3",
        "langchain-core>=1.2,<1.4",
        "langchain-community>=0.4,<0.5",
    ],
}


def _iter_source_files(repo_root: Path) -> Iterable[str]:
    for rel_dir in SOURCE_DIRS:
        base = repo_root / rel_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path.relative_to(repo_root).as_posix()


def build_manifest(repo_root: Path = REPO_ROOT) -> dict:
    """Return the demo manifest as a JSON-serialisable dict (pure)."""

    files = list(_iter_source_files(repo_root))
    for rel in DATA_FILES:
        if (repo_root / rel).is_file():
            files.append(rel)
    return {
        "entrypoint": ENTRYPOINT,
        "default_profile": "presentation_safe",
        "requirements": REQUIREMENTS,
        "files": files,
    }


def write_manifest(repo_root: Path = REPO_ROOT) -> Path:
    manifest = build_manifest(repo_root)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return MANIFEST_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed manifest matches a fresh build; do not write.",
    )
    args = parser.parse_args(argv)

    fresh = build_manifest()
    if args.check:
        if not MANIFEST_PATH.is_file():
            print(f"manifest missing: {MANIFEST_PATH}", file=sys.stderr)
            return 1
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing != fresh:
            print(
                "manifest is stale; run `python scripts/build_wasm_demo.py`",
                file=sys.stderr,
            )
            return 1
        print("manifest is fresh")
        return 0

    path = write_manifest()
    print(f"wrote {path} ({len(fresh['files'])} files)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
