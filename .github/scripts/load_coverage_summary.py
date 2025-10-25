#!/usr/bin/env python3
"""Load coverage summary markdown from artifacts and prepare for GitHub output."""

import os
from pathlib import Path


def main() -> None:
    """Find coverage summary markdown and write to output file."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    root = Path("gate_artifacts/downloads")

    summary_path: Path | None = None
    candidates = list(root.rglob("coverage-summary.md"))
    if candidates:
        summary_path = candidates[0]

    if summary_path is None:
        print("No coverage summary markdown found; continuing without it.")
    else:
        text = summary_path.read_text(encoding="utf-8")
        dest = Path("gate-coverage-summary.md")
        dest.write_text(text, encoding="utf-8")
        if output_path:
            with Path(output_path).open("a", encoding="utf-8") as handle:
                handle.write("body<<EOF\n")
                handle.write(text)
                if not text.endswith("\n"):
                    handle.write("\n")
                handle.write("EOF\n")
        print(f"Embedded coverage summary from {summary_path}")


if __name__ == "__main__":
    main()
