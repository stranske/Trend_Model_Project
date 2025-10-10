"""Utilities for computing coverage trend information for CI workflows."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_WARN_DROP = 1.0


@dataclass
class Baseline:
    """Represents the stored coverage baseline configuration."""

    line: Optional[float]
    warn_drop: float


@dataclass
class TrendResult:
    """Computed coverage trend details."""

    current: Optional[float]
    baseline: Optional[float]
    warn_drop: float
    delta: Optional[float]
    status: str
    minimum: Optional[float]

    def summary_lines(self) -> list[str]:
        lines = ["### Coverage Trend"]
        if (
            self.current is not None
            and self.baseline is not None
            and self.delta is not None
        ):
            lines.append(
                f"- Trend: {self.baseline:.2f}% → {self.current:.2f}% ({self.delta:+.2f} pts)"
            )
        elif self.current is not None and self.baseline is None:
            lines.append(
                f"- Trend: current {self.current:.2f}% (no baseline configured)"
            )
        elif self.current is None and self.baseline is not None:
            lines.append(
                f"- Trend: baseline {self.baseline:.2f}% (no coverage data)"
            )
        else:
            lines.append("- Trend: coverage unavailable")
        if self.current is not None:
            lines.append(f"- Current: {self.current:.2f}%")
        else:
            lines.append("- Current: unavailable")
        if self.baseline is not None:
            lines.append(f"- Baseline: {self.baseline:.2f}%")
        else:
            lines.append("- Baseline: unavailable")
        if self.delta is not None:
            lines.append(f"- Change: {self.delta:+.2f} pts")
            if self.status == "warn":
                lines.append(
                    f"- Warning: drop exceeds {self.warn_drop:.2f}-pt soft limit"
                )
        elif self.status == "no-baseline":
            lines.append(f"- Soft drop limit: {self.warn_drop:.2f} pts")
        if self.minimum is not None:
            lines.append(f"- Required minimum: {self.minimum:.2f}%")
        return lines

    def comment_body(self) -> str:
        if (
            self.status != "warn"
            or self.delta is None
            or self.current is None
            or self.baseline is None
        ):
            return ""
        lines = [
            "🔶 Coverage drop alert",
            "",
            f"Baseline coverage: {self.baseline:.2f}%",
            f"Current coverage: {self.current:.2f}%",
            f"Change: {self.delta:+.2f} percentage points",
        ]
        if self.minimum is not None:
            lines.append(f"Hard minimum: {self.minimum:.2f}%")
        lines.extend(
            [
                "",
                (
                    "The drop exceeds the soft limit of "
                    f"{self.warn_drop:.2f} points. This is a warning only; CI remains green."
                ),
                "",
                "Update config/coverage-baseline.json if the new level is expected.",
            ]
        )
        return "\n".join(lines)


def read_coverage(coverage_xml: Path, coverage_json: Path) -> Optional[float]:
    """Return the coverage percentage from XML or JSON reports."""

    if coverage_xml.is_file():
        try:
            root = ET.parse(coverage_xml).getroot()
        except ET.ParseError as exc:  # pragma: no cover - defensive logging
            print(f"Failed to parse {coverage_xml}: {exc}", file=sys.stderr)
        else:
            rate = root.get("line-rate")
            if rate is not None:
                try:
                    return float(rate) * 100.0
                except ValueError:  # pragma: no cover - defensive logging
                    print(f"Invalid line-rate value: {rate}", file=sys.stderr)
    if coverage_json.is_file():
        try:
            payload = json.loads(coverage_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
            print(f"Failed to parse {coverage_json}: {exc}", file=sys.stderr)
        else:
            totals = payload.get("totals") or {}
            try:
                covered = float(totals.get("covered_lines", 0))
                total = float(totals.get("num_statements", 0))
            except (TypeError, ValueError):
                return None
            if total:
                return covered / total * 100.0
    return None


def load_baseline(path: Path) -> Baseline:
    """Load the coverage baseline configuration."""

    if not path:
        return Baseline(line=None, warn_drop=DEFAULT_WARN_DROP)
    if not path.is_file():
        print(f"Baseline file {path} not found", file=sys.stderr)
        return Baseline(line=None, warn_drop=DEFAULT_WARN_DROP)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Unable to parse baseline file {path}: {exc}", file=sys.stderr)
        return Baseline(line=None, warn_drop=DEFAULT_WARN_DROP)
    line_value = data.get("line")
    warn_value = data.get("warn_drop", DEFAULT_WARN_DROP)
    line = float(line_value) if isinstance(line_value, (int, float)) else None
    warn = (
        float(warn_value)
        if isinstance(warn_value, (int, float)) and warn_value >= 0
        else DEFAULT_WARN_DROP
    )
    if line is None:
        print(
            f"Baseline file {path} missing numeric 'line' entry",
            file=sys.stderr,
        )
    return Baseline(line=line, warn_drop=warn)


def evaluate_trend(
    current: Optional[float],
    baseline: Baseline,
    minimum: Optional[float] = None,
) -> TrendResult:
    """Compute the coverage trend and return the structured result."""

    delta: Optional[float] = None
    status = "no-data"
    if current is not None and baseline.line is not None:
        delta = current - baseline.line
        status = "warn" if delta < -baseline.warn_drop else "ok"
    elif current is not None:
        status = "no-baseline"
    return TrendResult(
        current=current,
        baseline=baseline.line,
        warn_drop=baseline.warn_drop,
        delta=delta,
        status=status,
        minimum=minimum,
    )


def write_lines(path: Optional[Path], lines: Iterable[str]) -> None:
    if not path:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def dump_artifact(path: Path, result: TrendResult) -> None:
    payload = {
        "current": result.current,
        "baseline": result.baseline,
        "delta": result.delta,
        "warn_drop": result.warn_drop,
        "minimum": result.minimum,
        "status": result.status,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_github_output(path: Optional[Path], result: TrendResult) -> None:
    if not path:
        return
    outputs = []
    if result.current is not None:
        outputs.append(f"current={result.current:.2f}")
    if result.baseline is not None:
        outputs.append(f"baseline={result.baseline:.2f}")
    if result.delta is not None:
        outputs.append(f"delta={result.delta:.2f}")
    outputs.append(f"warn_drop={result.warn_drop:.2f}")
    if result.minimum is not None:
        outputs.append(f"minimum={result.minimum:.2f}")
    outputs.append(f"status={result.status}")
    comment = result.comment_body()
    if comment:
        outputs.append("comment<<EOF")
        outputs.append(comment)
        outputs.append("EOF")
    with path.open("a", encoding="utf-8") as handle:
        for line in outputs:
            handle.write(line + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute coverage trend data.")
    parser.add_argument("--coverage-xml", type=Path, default=Path("coverage.xml"))
    parser.add_argument("--coverage-json", type=Path, default=Path("coverage.json"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("config/coverage-baseline.json")
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, default=None)
    parser.add_argument("--minimum", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    current = read_coverage(args.coverage_xml, args.coverage_json)
    baseline = load_baseline(args.baseline)
    result = evaluate_trend(current, baseline, minimum=args.minimum)
    if args.summary_path:
        write_lines(args.summary_path, result.summary_lines())
    dump_artifact(args.artifact_path, result)
    write_github_output(args.github_output, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
