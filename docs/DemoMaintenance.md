# Demo Maintenance

This document summarises the steps required to keep the demo pipeline in sync with the exporter and pipeline.

1. **Bootstrap the environment**
   ```bash
   ./scripts/setup_env.sh
   ```
2. **Generate the demo dataset**
   ```bash
   python scripts/generate_demo.py
   ```
3. **Run the full demo pipeline and export checks**
   ```bash
python scripts/run_multi_demo.py
```
The script calls `export.export_data()` so CSV, Excel, JSON and TXT outputs are
produced in one go. It now writes the workbook frames from the multi‑period run
via a single call to `export.export_data` to validate all exporters. Extend the
script and `config/demo.yml` whenever new exporter options are introduced. It
also exercises the multi-period export helpers, verifies the CLI wrappers and
checks the Jupyter notebook utilities.
It now validates configuration round-tripping via ``model_dump`` and
``model_dump_json``.
It further exercises core utilities like ``_zscore`` and the base weighting
stub to ensure low-level helpers behave correctly. The demo now also selects
funds using ``information_ratio`` so the new metric is tested end-to-end.
It now verifies the Excel formatter registry via ``reset_formatters_excel`` and
``register_formatter_excel`` so custom formatting hooks remain functional.
It additionally calls the multi-period exporters with ``include_metrics`` both
enabled and disabled to ensure all code paths are exercised.
It now verifies metric functions with DataFrame input, exercises
``ParamStore.to_dict`` and runs the multi-period engine using a pre-loaded
DataFrame to cover the optional argument.
It also tests ``zscore_window`` and ``ddof`` behaviour via ``_apply_transform``
so advanced ranking options remain covered.
4. **Run the test suite**
   ```bash
   ./scripts/run_tests.sh
   ```
   Ensure all unit tests pass after modifying the demo pipeline.
5. **Keep demo config current**
   - Update `config/demo.yml` and demo scripts whenever export or pipeline behaviour changes so the demo exercises every code path.
