#!/usr/bin/env python3
"""Strip output from Jupyter notebooks in-place."""

import sys

import nbformat


def strip_output(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
    changed = False
    for cell in nb.cells:
        if cell.get("outputs"):
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)


def main(argv: list[str]) -> None:
    for filename in argv:
        strip_output(filename)


if __name__ == "__main__":
    main(sys.argv[1:])
