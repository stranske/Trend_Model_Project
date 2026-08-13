#!/usr/bin/env python3
"""Assemble the static stlite bundle consumed by GitHub Pages.

The runtime assets live under ``demo/wasm`` while the application sources named
by ``build_wasm_demo.py`` remain in their canonical repository locations. A
Pages artifact needs both in one tree, so this command writes the generated
manifest at the site root and copies each declared source beneath ``app/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

if __package__:
    from scripts.build_wasm_demo import REPO_ROOT, build_manifest
else:  # Direct invocation: python scripts/build_wasm_site.py
    from build_wasm_demo import REPO_ROOT, build_manifest


DEFAULT_OUTPUT = REPO_ROOT / "dist" / "wasm-demo"


def _validated_relative_path(value: str) -> Path:
    """Return a safe relative path from a generated manifest entry."""

    raw_parts = value.split("/")
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"unsafe manifest path: {value!r}")
    return Path(*posix.parts)


def _prepare_output(repo_root: Path, output_dir: Path) -> None:
    repo = repo_root.resolve()
    output = output_dir.resolve()
    if output == repo or output in repo.parents:
        raise ValueError(f"refusing broad output path: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)


def assemble_site(repo_root: Path = REPO_ROOT, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Build a self-contained Pages artifact and return its manifest."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    _prepare_output(repo_root, output_dir)

    demo_root = repo_root / "demo" / "wasm"
    index_path = demo_root / "index.html"
    vendor_path = demo_root / "vendor"
    if not index_path.is_file() or not vendor_path.is_dir():
        raise FileNotFoundError("demo/wasm must contain index.html and vendor/")

    shutil.copy2(index_path, output_dir / "index.html")
    shutil.copytree(vendor_path, output_dir / "vendor")

    manifest = build_manifest(repo_root)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    app_root = output_dir / "app"
    for value in manifest["files"]:
        relative = _validated_relative_path(value)
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"manifest source is missing: {relative.as_posix()}")
        destination = app_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Pages artifact directory (default: dist/wasm-demo)",
    )
    args = parser.parse_args(argv)

    manifest = assemble_site(output_dir=args.output)
    print(f"wrote {args.output.resolve()} ({len(manifest['files'])} application files)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
