# Issue #1683 – Unified Report Builder Checklist

All acceptance criteria for [Issue #1683](https://github.com/stranske/Trend_Model_Project/issues/1683) have been implemented.

- [x] Implement a shared report generator that outputs both HTML and optional PDF with consistent content across CLI and Streamlit app (`src/trend/reporting/unified.py`).
- [x] Build report sections covering the executive summary, metrics table, turnover/exposure charts, parameter summary, and caveats (HTML/PDF rendering helpers).
- [x] Produce a deterministic narrative paragraph sourced from `BacktestResult` data and static specifications (`_narrative`/`_build_backtest`).
- [x] Add CLI support for specifying an `--output` path for generated reports, including shared path resolution and optional PDF export (`trend report --output/--pdf`).
- [x] Add a Streamlit "Download report" button that uses the shared generator output (`streamlit_app/pages/4_Results.py`).
- [x] Append a small "Past performance does not guarantee future results" disclaimer/footer to generated reports.
- [x] Verify that the CLI-generated HTML report matches the downloadable app report exactly, satisfying acceptance criteria (regression test in `tests/test_trend_cli.py::test_cli_report_matches_shared_generator`).

Automated verification:

- `pytest tests/test_unified_report.py tests/test_trend_cli.py::test_cli_report_matches_shared_generator`
- `pytest tests/test_trend_cli.py::test_main_report_supports_output_file_only tests/test_trend_cli.py::test_main_report_uses_requested_directory tests/test_trend_cli.py::test_main_report_writes_pdf_when_requested tests/test_trend_cli.py::test_main_report_pdf_dependency_error tests/test_trend_cli_entrypoints.py::test_main_report_command`
