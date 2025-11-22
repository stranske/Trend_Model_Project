"""Test weight engine creation logging improvements."""

import logging
from unittest.mock import patch

import pandas as pd

from trend_analysis import pipeline


def make_df():
    """Create test DataFrame with returns data."""
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": [0.02, 0.01, 0.03, 0.04, 0.02, 0.01],
            "B": [0.01, 0.02, 0.02, 0.03, 0.01, 0.02],
        }
    )


def test_weight_engine_success_logging(caplog):
    """Test successful weight engine creation logs success message."""
    df = make_df()

    with caplog.at_level(logging.DEBUG):
        result = pipeline._run_analysis(
            df,
            "2020-01",
            "2020-03",  # in sample
            "2020-04",
            "2020-06",  # out of sample
            target_vol=1.0,
            monthly_cost=0.0,
            weighting_scheme="risk_parity",
        )

    # Check that analysis succeeded
    assert result is not None

    # Check that success message was logged
    debug_logs = [
        record for record in caplog.records if record.levelno == logging.DEBUG
    ]
    success_logs = [log for log in debug_logs if "Successfully created" in log.message]
    assert len(success_logs) > 0
    assert "risk_parity" in success_logs[0].message


def test_weight_engine_failure_logging(caplog):
    """Test weight engine creation failure logs fallback message."""
    df = make_df()

    with caplog.at_level(logging.DEBUG):
        result = pipeline._run_analysis(
            df,
            "2020-01",
            "2020-03",  # in sample
            "2020-04",
            "2020-06",  # out of sample
            target_vol=1.0,
            monthly_cost=0.0,
            weighting_scheme="nonexistent_engine",  # This should fail
        )

    # Check that analysis succeeded (fallback to equal weights)
    assert result is not None
    # Fallback info structure present
    assert result.get("weight_engine_fallback") is not None
    fb = result["weight_engine_fallback"]
    assert fb["engine"] == "nonexistent_engine"

    # Check that fallback message was logged
    debug_logs = [
        record for record in caplog.records if record.levelno == logging.DEBUG
    ]
    fallback_logs = [
        log for log in debug_logs if "falling back to equal weights" in log.message
    ]
    assert len(fallback_logs) > 0
    assert "Weight engine creation failed" in fallback_logs[0].message
    # A single WARNING should have been emitted for visibility
    warn_logs = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    warning_fallback = [
        w for w in warn_logs if "falling back to equal weights" in w.message
    ]
    assert len(warning_fallback) == 1


def test_weight_engine_import_failure_logging(caplog):
    """Test weight engine creation with import failure logs fallback
    message."""
    df = make_df()

    # Mock the import to fail
    with patch("trend_analysis.plugins.create_weight_engine") as mock_import:
        mock_import.side_effect = ImportError("Mock import error")

        with caplog.at_level(logging.DEBUG):
            result = pipeline._run_analysis(
                df,
                "2020-01",
                "2020-03",  # in sample
                "2020-04",
                "2020-06",  # out of sample
                target_vol=1.0,
                monthly_cost=0.0,
                weighting_scheme="risk_parity",
            )

    # Check that analysis succeeded (fallback to equal weights)
    assert result is not None

    # Check that fallback message was logged with the specific error
    debug_logs = [
        record for record in caplog.records if record.levelno == logging.DEBUG
    ]
    fallback_logs = [
        log for log in debug_logs if "falling back to equal weights" in log.message
    ]
    assert len(fallback_logs) > 0
    assert "Mock import error" in fallback_logs[0].message


def test_weight_engine_failure_preserves_logger_level_and_metadata(caplog):
    """Fallback path should not mutate logger levels and carries diagnostics."""

    df = make_df()
    original_level = pipeline.logger.level
    pipeline.logger.setLevel(logging.INFO)

    try:
        with caplog.at_level(logging.DEBUG, logger=pipeline.logger.name):
            result = pipeline._run_analysis(
                df,
                "2020-01",
                "2020-03",  # in sample
                "2020-04",
                "2020-06",  # out of sample
                target_vol=1.0,
                monthly_cost=0.0,
                weighting_scheme="nonexistent_engine",
            )

        assert pipeline.logger.level == logging.INFO
    finally:
        pipeline.logger.setLevel(original_level)
    fallback = result["weight_engine_fallback"]
    assert fallback["engine"] == "nonexistent_engine"
    assert fallback["logger_level"]

    debug_logs = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG and getattr(record, "weight_engine", None)
        == "nonexistent_engine"
    ]
    assert any("falling back to equal weights" in record.message for record in debug_logs)


def test_weight_engine_success_preserves_logger_level(caplog):
    """Successful engine creation should leave logger level untouched."""

    df = make_df()
    original_level = pipeline.logger.level
    pipeline.logger.setLevel(logging.INFO)

    try:
        with caplog.at_level(logging.DEBUG, logger=pipeline.logger.name):
            result = pipeline._run_analysis(
                df,
                "2020-01",
                "2020-03",  # in sample
                "2020-04",
                "2020-06",  # out of sample
                target_vol=1.0,
                monthly_cost=0.0,
                weighting_scheme="risk_parity",
            )

        assert result is not None
        assert pipeline.logger.level == logging.INFO
    finally:
        pipeline.logger.setLevel(original_level)

    debug_logs = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG and getattr(record, "weight_engine", None)
        == "risk_parity"
    ]
    assert any("Successfully created risk_parity weight engine" in record.message for record in debug_logs)


def test_weight_engine_no_scheme_no_logging(caplog):
    """Test that no weight engine logging occurs when no weighting scheme is
    provided."""
    df = make_df()

    with caplog.at_level(logging.DEBUG):
        result = pipeline._run_analysis(
            df,
            "2020-01",
            "2020-03",  # in sample
            "2020-04",
            "2020-06",  # out of sample
            target_vol=1.0,
            monthly_cost=0.0,
            weighting_scheme=None,  # No scheme provided
        )

    # Check that analysis succeeded
    assert result is not None

    # Check that no weight engine related messages were logged
    debug_logs = [
        record for record in caplog.records if record.levelno == logging.DEBUG
    ]
    weight_logs = [log for log in debug_logs if "weight engine" in log.message.lower()]
    assert len(weight_logs) == 0


def test_weight_engine_equal_scheme_no_logging(caplog):
    """Test that no weight engine logging occurs for equal weighting scheme."""
    df = make_df()

    with caplog.at_level(logging.DEBUG):
        result = pipeline._run_analysis(
            df,
            "2020-01",
            "2020-03",  # in sample
            "2020-04",
            "2020-06",  # out of sample
            target_vol=1.0,
            monthly_cost=0.0,
            weighting_scheme="equal",  # Equal weighting - should skip engine creation
        )

    # Check that analysis succeeded
    assert result is not None

    # Check that no weight engine related messages were logged
    debug_logs = [
        record for record in caplog.records if record.levelno == logging.DEBUG
    ]
    weight_logs = [log for log in debug_logs if "weight engine" in log.message.lower()]
    assert len(weight_logs) == 0
