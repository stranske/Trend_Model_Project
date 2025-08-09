# Codex Work Instructions (App + Sim Layer)
- Prefer calling `trend_analysis.pipeline.single_period_run` when available.
- Expand tests first; each PR solves one issue.
- How to run: `./scripts/run_streamlit.sh`, `pytest -q`.
- Acceptance criteria: schema validator, policy engine behavior, simulator smoke test, pipeline integration parity within tolerance.
- Backlog: preview score frame, weight heatmap, integrate native rank_selection after upstream merge, add expected shortfall & diversification value, export commit hash.
