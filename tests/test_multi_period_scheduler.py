from trend_analysis.multi_period.scheduler import generate_periods


def test_generate_periods_includes_truncated_final_window():
    cfg = {
        "multi_period": {
            "frequency": "A",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2000-07",
            # End is earlier than a full 12-month out window for the final period.
            # OOS would begin at 2002-07, so end must be after that to allow a
            # truncated final window.
            "end": "2003-02",
        }
    }

    periods = generate_periods(cfg)
    assert periods, "Expected at least one period"

    # Final period should be truncated to the configured end date (month-end).
    assert periods[-1].out_end.startswith("2003-02")


def test_generate_periods_start_mode_oos_anchors_first_oos_start():
    cfg = {
        "multi_period": {
            "frequency": "A",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2000-07",
            "end": "2001-12",
            "start_mode": "oos",
        }
    }

    periods = generate_periods(cfg)
    assert periods
    first = periods[0]

    # start anchors the first out-of-sample start (month-end)
    assert first.out_start.startswith("2000-07")

    # in-sample ends immediately before OOS starts
    assert first.in_end.startswith("2000-06")

    # with 2 annual windows lookback (24 months), in-sample begins 1998-07
    assert first.in_start.startswith("1998-07")


def test_generate_periods_stops_when_no_oos_available():
    cfg = {
        "multi_period": {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 6,
            "start": "2000-01",
            # End before any OOS could begin.
            "end": "2000-01",
        }
    }

    periods = generate_periods(cfg)
    assert periods == []
