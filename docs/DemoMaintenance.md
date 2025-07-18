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
   The script calls `export.export_data()` so CSV, Excel, JSON and TXT outputs are produced in one go. Extend the script and `config/demo.yml` whenever new exporter options are introduced. It now also exercises the multi-period export helpers, verifies the CLI wrappers and checks the Jupyter notebook utilities.
4. **Run the test suite**
   ```bash
   ./scripts/run_tests.sh
   ```
   Ensure all unit tests pass after modifying the demo pipeline.
5. **Keep demo config current**
   - Update `config/demo.yml` and demo scripts whenever export or pipeline behaviour changes so the demo exercises every code path.

