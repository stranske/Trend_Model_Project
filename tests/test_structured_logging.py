import json
from pathlib import Path

from trend_analysis.logging import init_run_logger, log_step


def test_structured_logging_basic(tmp_path: Path):
    run_id = "test123"
    log_path = tmp_path / "logs" / "run_test.jsonl"
    init_run_logger(run_id, log_path)
    log_step(run_id, "load_data", "Loaded data", rows=42)
    log_step(run_id, "selection", "Selected managers", count=5)
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3  # init + 2 steps
    parsed = [json.loads(line) for line in lines]
    for obj in parsed:
        assert obj["run_id"] == run_id
        assert set(obj).issuperset({"ts", "run_id", "step", "level", "msg", "extra"})
    # Check one specific extra field present
    selection = [o for o in parsed if o["step"] == "selection"][0]
    assert selection["extra"]["count"] == 5
