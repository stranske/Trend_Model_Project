# Issue 6147: whole-percent configuration round trip

The Model page now rounds both conversions from the stored `rank_pct` fraction
to the integer percentage widget. A saved `0.29` displays as 29 and remains
`0.29` after rendering and saving the form. The widget range and percent-to-fraction
save conversion are unchanged.

Validation on 2026-10-27, using the canonical repository `.venv/bin/python`:

```text
python -m pytest tests/streamlit/test_coverage_wp_t3.py tests/streamlit/test_coverage_wp_t8.py -q
223 passed in 2.26s
```

The existing strict-xfail marker was removed from
`test_sync_model_widgets_round_trips_whole_percentages`. The render-and-save
regression also covers 25% and 29%, asserting the caption and persisted fraction.

Deliberate-break proof: temporarily restore both original `int(float(...) * 100)`
expressions, then run:

```text
python -m pytest tests/streamlit/test_coverage_wp_t3.py::test_sync_model_widgets_round_trips_whole_percentages tests/streamlit/test_coverage_wp_t8.py::test_inclusion_top_pct_and_buy_hold_captions -q
2 failed, 1 passed in 0.44s
```

The failures expose the 28-versus-29 widget value and the missing 29% caption.
After restoring the implementation byte-for-byte, the same command passes:

```text
3 passed in 0.33s
```

These tests use the existing Streamlit test doubles; they do not claim a live
browser exercise. Raw logs, the exact interpreter command, and the restored-file
SHA-256 are retained in the opener run `run-20261027T0400`.

## Review recovery, 2026-10-27

Review threads `discussion_r4385020411` and `discussion_r4385020426` exposed an
out-of-range imported fraction and a render-only assertion. Both conversion
points now clamp the rounded percentage to the existing widget range, 1–50.
The 25% and 29% tests submit Save Configuration, require its success message
and cache invalidation, and check the fraction produced by that save path.

Before the clamp, the focused boundary/save command reported **4 failed,
4 passed**: imports of 0.506 and 0.004 produced 51 and 0 in each conversion
path. With the fix, the two affected modules report **228 passed**:

```bash
python -m pytest tests/streamlit/test_coverage_wp_t3.py tests/streamlit/test_coverage_wp_t8.py -q
```

To verify that the save assertions observe a real update, the conversion
`rank_pct_input / 100.0` was temporarily changed to `/ 10.0`. The 25% and 29%
save cases both failed, with saved values 2.5 and 2.9. Restoring the file
byte-for-byte made both pass. The mutation was never committed.

Validation used the canonical repo's `.venv/bin/python`; Ruff and diff checks
pass. Black used `/opt/anaconda3/bin/black --target-version py312` because
Black is absent from the selected test interpreter. Full local transcripts
and the restore checksum are retained in automation run
`run-20261027T0457`. Hosted checks and the new post-push review floor must be
verified separately for the published commit.
