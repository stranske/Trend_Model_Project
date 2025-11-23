archives/generated/2025/2025-11-22_gate-summary.md
**Keepalive checklist snapshot (issue #3689)**

#### Scope
- [x] `analysis/tearsheet.py::render(results, out="reports/tearsheet.md")` with headline stats and run metadata.
- [x] Plots for equity curve, rolling Sharpe/vol, drawdown, and turnover.
- [x] CLI wiring: `python -m src.cli report --last-run`.
- [x] Reference the new report from `Portfolio_Test_Results_Summary.md`.

#### Tasks
- [x] Implement renderer and basic plots.
- [x] Minimal CLI and example run command in README.
- [x] Test that the file is written and includes expected sections.

#### Acceptance criteria
- [x] Running the CLI produces a tearsheet file with stats and plots using the latest `Results`.
- [x] CI artifact (optional) or repo file present after a demo run.