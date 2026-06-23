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
    python scripts/build_wasm_demo.py --check     # verify generated manifest freshness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "demo" / "wasm" / "manifest.json"
WASM_VENDOR_DIR = REPO_ROOT / "demo" / "wasm" / "vendor"

#: Pure-PyPI wheels that are NOT in the Pyodide lock and so cannot be resolved by
#: name like the lock-backed requirements. They are committed under
#: ``demo/wasm/vendor/pypi/`` and installed by passing absolute same-origin wheel
#: URLs to micropip (``index.html`` rewrites every ``*.whl`` requirement entry).
#: ``plotly`` powers the Monte Carlo page's ``st.plotly_chart`` surfaces; it is
#: pure-PyPI for this runtime. (plotly 6 dropped the ``tenacity`` dependency that
#: plotly 5 carried; its only runtime deps are ``narwhals`` and ``packaging``,
#: both already in the vendored Pyodide lock — see ``PYODIDE_LOCK_PYPI_OVERRIDES``
#: for the narwhals version bump that plotly 6.x's express module needs.)
PYPI_WHEEL_DIR = "pypi"
PYPI_WHEELS = ("plotly-6.8.0-py3-none-any.whl",)

#: The Pyodide 0.27.2 lock ships narwhals 1.10.0, which predates the
#: ``from_native(..., pass_through=...)`` API that plotly 6.x's ``plotly.express``
#: calls (``go``/graph_objects works on 1.10.0, ``px`` does not). micropip will
#: not upgrade a package the stlite bootstrap already installed from the lock, so
#: the lock itself is bumped in place to narwhals 1.15.1 (the floor plotly 6.8.0
#: declares). A single narwhals then serves both the lock's altair and plotly.
#: The wheel is pure-PyPI (not on the Pyodide CDN); see fetch_offline_runtime.py.
PYODIDE_LOCK_PYPI_OVERRIDES = {"narwhals": "1.15.1"}

#: Manifest requirement strings for the vendored PyPI wheels: repo-relative paths
#: (resolved to absolute same-origin URLs at runtime in ``index.html``).
PYPI_WHEEL_REQUIREMENTS = tuple(
    f"vendor/{PYPI_WHEEL_DIR}/{name}" for name in PYPI_WHEELS
)

VENDORED_RUNTIME_FILES = (
    *(f"{PYPI_WHEEL_DIR}/{name}" for name in PYPI_WHEELS),
    "stlite@0.79.4/stlite.js",
    "stlite@0.79.4/style.css",
    "stlite@0.79.4/wheels/stlite_lib-0.1.0-py3-none-any.whl",
    "stlite@0.79.4/wheels/streamlit-1.41.0-cp312-none-any.whl",
    "pyodide-0.27.2/pyodide.mjs",
    "pyodide-0.27.2/pyodide.asm.js",
    "pyodide-0.27.2/pyodide.asm.wasm",
    "pyodide-0.27.2/python_stdlib.zip",
    "pyodide-0.27.2/pyodide-lock.json",
    "pyodide-0.27.2/Jinja2-3.1.3-py3-none-any.whl",
    "pyodide-0.27.2/MarkupSafe-2.1.5-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/PyYAML-6.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/altair-5.4.1-py3-none-any.whl",
    "pyodide-0.27.2/annotated_types-0.6.0-py3-none-any.whl",
    "pyodide-0.27.2/attrs-23.2.0-py3-none-any.whl",
    "pyodide-0.27.2/cachetools-5.3.3-py3-none-any.whl",
    "pyodide-0.27.2/contourpy-1.3.0-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/cramjam-2.8.3-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/cycler-0.12.1-py3-none-any.whl",
    "pyodide-0.27.2/fastparquet-2024.5.0-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/fonttools-4.51.0-py3-none-any.whl",
    "pyodide-0.27.2/fsspec-2024.3.1-py3-none-any.whl",
    "pyodide-0.27.2/jsonschema-4.21.1-py3-none-any.whl",
    "pyodide-0.27.2/jsonschema_specifications-2023.12.1-py3-none-any.whl",
    "pyodide-0.27.2/kiwisolver-1.4.5-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/matplotlib-3.8.4-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/matplotlib_pyodide-0.2.3-py3-none-any.whl",
    "pyodide-0.27.2/micropip-0.8.0-py3-none-any.whl",
    "pyodide-0.27.2/narwhals-1.15.1-py3-none-any.whl",
    "pyodide-0.27.2/numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/packaging-24.2-py3-none-any.whl",
    "pyodide-0.27.2/pandas-2.2.3-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/pillow-10.2.0-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/protobuf-5.29.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/pydantic-2.10.5-py3-none-any.whl",
    "pyodide-0.27.2/pydantic_core-2.27.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/pyodide_http-0.2.1-py3-none-any.whl",
    "pyodide-0.27.2/pyparsing-3.1.2-py3-none-any.whl",
    "pyodide-0.27.2/pyrsistent-0.20.0-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
    "pyodide-0.27.2/pytz-2024.1-py2.py3-none-any.whl",
    "pyodide-0.27.2/referencing-0.34.0-py3-none-any.whl",
    "pyodide-0.27.2/rpds_py-0.18.0-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/scipy-1.14.1-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "pyodide-0.27.2/six-1.16.0-py2.py3-none-any.whl",
    "pyodide-0.27.2/typing_extensions-4.11.0-py3-none-any.whl",
)

#: Streamlit entrypoint, repo-relative.
ENTRYPOINT = "streamlit_app/app.py"

#: Source directories whose ``*.py`` files are loaded into the browser FS.
SOURCE_DIRS = (
    "streamlit_app",
    "analysis",
    "src/trend_analysis",
    "src/trend",
    "src/utils",
    "src/data",
    "src/backtest",
)

#: Bundled synthetic data shipped with the demo (presentation-safe default).
DATA_FILES = ("demo/demo_returns.csv",)

#: Python requirements installed under Pyodide, per runtime profile. The
#: presentation-safe set intentionally omits LangChain so the default load is
#: lean and has no LLM dependency footprint; public_llm_demo adds the LangChain
#: stack pinned in ``pyproject.toml``.
REQUIREMENTS = {
    "presentation_safe": [
        "numpy",
        "pandas",
        "pyyaml",
        "pydantic",
        "scipy",
        "matplotlib",
        # plotly (vendored wheel, resolved to an absolute URL in index.html) so
        # the Monte Carlo page's st.plotly_chart surfaces render offline. Its
        # narwhals/packaging deps are satisfied by the vendored Pyodide lock.
        *PYPI_WHEEL_REQUIREMENTS,
    ],
    "public_llm_demo": [
        "numpy",
        "pandas",
        "pyyaml",
        "pydantic",
        "scipy",
        "matplotlib",
        "langchain>=1.3,<1.4",
        "langchain-core>=1.4,<1.5",
        "langchain-community>=0.4,<0.5",
        "langchain-openai>=1.0,<1.1",
        "langchain-anthropic>=1.2,<1.3",
        "langchain-ollama>=1.0,<1.1",
        *PYPI_WHEEL_REQUIREMENTS,
    ],
}


def missing_runtime_files(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return vendored runtime files that are required for offline boot."""

    vendor_dir = repo_root / "demo" / "wasm" / "vendor"
    return [rel for rel in VENDORED_RUNTIME_FILES if not (vendor_dir / rel).is_file()]


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
        help="Verify the generated manifest artifact matches a fresh build; do not write.",
    )
    args = parser.parse_args(argv)

    fresh = build_manifest()
    if args.check:
        missing = missing_runtime_files()
        if missing:
            print(
                "vendored wasm runtime is incomplete:\n"
                + "\n".join(f"- demo/wasm/vendor/{rel}" for rel in missing),
                file=sys.stderr,
            )
            return 1
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
