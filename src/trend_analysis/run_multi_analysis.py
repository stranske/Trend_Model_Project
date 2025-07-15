from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from trend_analysis.config import load
from trend_analysis.multi_period.engine import run as run_multi


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for multi-period trend analysis."""
    parser = argparse.ArgumentParser(prog="trend-analysis-multi")
    parser.add_argument("-c", "--config", help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    # Run multi-period analysis; returns list of dicts
    results = run_multi(cfg)

    # Prepare export
    export_cfg = getattr(cfg, 'export', {}) or {}
    out_dir = export_cfg.get('directory')
    formats = export_cfg.get('formats', [])
    filename = export_cfg.get('filename', 'analysis')

    if not out_dir or not formats:
        return 0

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write each period result to CSV if requested
    for i, res in enumerate(results):
        df = pd.DataFrame([res])
        if any(fmt.lower() == 'csv' for fmt in formats):
            df.to_csv(out_path / f"{filename}_{i}.csv", index=False)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
