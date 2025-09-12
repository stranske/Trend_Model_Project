#!/usr/bin/env python
"""Demo CLI for the multi-period engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ensure src directory is on sys.path when executed with PYTHONPATH=./src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trend_analysis.config import load
from trend_analysis.multi_period import engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-multi-demo")
    parser.add_argument("-c", "--config", help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    engine.run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
