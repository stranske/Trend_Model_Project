import ast
import logging
from pathlib import Path

from trend_analysis.perf.timing import log_timing, timed_stage


def test_log_timing_emits_single_line(caplog):
    caplog.set_level(logging.INFO, logger="trend_analysis.performance")

    log_timing("example", duration_s=0.001, status="hit", foo="bar")

    record = caplog.records[-1]
    assert "stage=example" in record.message
    assert "status=hit" in record.message
    assert "foo=bar" in record.message


def test_log_timing_noop_when_logger_disabled(caplog):
    caplog.set_level(logging.WARNING, logger="trend_analysis.performance")

    log_timing("example", duration_s=0.002, status="miss")

    assert not caplog.records


def test_timed_stage_allows_mutating_state(caplog):
    caplog.set_level(logging.INFO, logger="trend_analysis.performance")

    with timed_stage("slow_step", status="miss", foo="bar") as state:
        state["status"] = "hit"
        state.setdefault("extra", {})["rows"] = 12

    record = caplog.records[-1]
    assert "stage=slow_step" in record.message
    assert "status=hit" in record.message
    assert "rows=12" in record.message


def test_timed_stage_defaults_and_extra_metadata(caplog):
    caplog.set_level(logging.INFO, logger="trend_analysis.performance")

    with timed_stage("lazy_step") as state:
        state.setdefault("extra", {})["note"] = "n/a"

    record = caplog.records[-1]
    assert "stage=lazy_step" in record.message
    assert "duration_ms" in record.message
    assert "status=" not in record.message
    assert "note=n/a" in record.message


def test_timing_helpers_have_one_implementation_owner() -> None:
    root = Path(__file__).parents[1] / "src" / "trend_analysis"
    definitions = {
        symbol: [
            path.relative_to(root).as_posix()
            for path in root.rglob("*.py")
            if symbol
            in {
                node.name
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        ]
        for symbol in ("log_timing", "timed_stage")
    }

    assert definitions == {
        "log_timing": ["perf/timing.py"],
        "timed_stage": ["perf/timing.py"],
    }
