import logging

from trend_analysis.perf.timing import log_timing, timed_stage


def test_log_timing_emits_single_line(caplog):
    caplog.set_level(logging.INFO, logger="trend_analysis.performance")

    log_timing("example", duration_s=0.001, status="hit", foo="bar")

    record = caplog.records[-1]
    assert "stage=example" in record.message
    assert "status=hit" in record.message
    assert "foo=bar" in record.message


def test_timed_stage_allows_mutating_state(caplog):
    caplog.set_level(logging.INFO, logger="trend_analysis.performance")

    with timed_stage("slow_step", status="miss", foo="bar") as state:
        state["status"] = "hit"
        state.setdefault("extra", {})["rows"] = 12

    record = caplog.records[-1]
    assert "stage=slow_step" in record.message
    assert "status=hit" in record.message
    assert "rows=12" in record.message
