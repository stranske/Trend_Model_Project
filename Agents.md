# Agents.md

> **Canonical implementation roadmap**: See [docs/phase-2/Agents.md](docs/phase-2/Agents.md) for the complete Phase-2 implementation spec and step-by-step guidance.

## Mission

Converge the scattered modules into one fully test-covered, vectorised pipeline that can be invoked from a single CLI entry-point. Never touch notebooks living under any directory whose name ends in `old/`.

---

## Quick Reference

### Demo Pipeline

```bash
# 1. Bootstrap environment
./scripts/setup_env.sh

# 2. Generate demo dataset
python scripts/generate_demo.py

# 3. Run full demo pipeline
python scripts/run_multi_demo.py

# 4. Run test suite
./scripts/run_tests.sh
```

See [docs/DemoMaintenance.md](docs/DemoMaintenance.md) for the full checklist.

### Key Entry Points

| Purpose | Command |
|---------|---------|
| CLI analysis | `PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml` |
| Streamlit app | `./scripts/run_streamlit.sh` |
| Tests | `./scripts/run_tests.sh` |
| Fast validation | `./scripts/dev_check.sh --fix` |

---

## Automation Entry Points

### Agents 70 Orchestrator
- **File**: `.github/workflows/agents-70-orchestrator.yml`
- **Role**: Single automation front door for all agent operations
- **Triggers**: 20-minute schedule sweep plus manual `workflow_dispatch`

### Agents 63 Issue Intake
- **File**: `.github/workflows/agents-63-issue-intake.yml`
- **Role**: Bootstrap Codex PRs on `agent:codex` labeled issues
- **Triggers**: Issue events (`opened`, `labeled`, `reopened`) plus manual dispatch

### Agents 64 Verify Agent Assignment
- **File**: `.github/workflows/agents-64-verify-agent-assignment.yml`
- **Role**: Validates agent assignment before orchestrator proceeds

Legacy consumer wrappers were retired. See [docs/archive/ARCHIVE_WORKFLOWS.md](docs/archive/ARCHIVE_WORKFLOWS.md).

---

## Canonical Module Locations

| Layer / Concern | Canonical Location | Deprecated |
|-----------------|-------------------|------------|
| Data ingest & cleaning | `trend_analysis/data.py` | `data_utils.py`, notebook helpers |
| Portfolio logic & metrics | `trend_analysis/metrics.py` | loops in `run_analysis.py` |
| Export / I/O | `trend_analysis/export/` | root-level `exports.py` |
| Domain kernels | `trend_analysis/core/` | standalone `core/` directory |
| Pipeline orchestration | `trend_analysis/pipeline.py` | duplicated control flow |
| CLI entry-point | `trend_analysis/cli.py` | bespoke `scripts/*.py` |
| Config | `config/defaults.yml` via `trend_analysis.config.load()` | hard-coded constants |
| Tests | `tests/` (pytest with coverage gate) | — |

**Rule**: One concern → one module. Replacements must delete or comment-out whatever they obsolete in the same PR.

---

## Guard-Rails

### Communication
- When a request contains multiple tasks, explicitly recap which items are complete and which remain before handing control back.
- Call out any suggestion that is only a partial fix, and list the follow-up steps required for it to succeed.
- Highlight assumptions about workflow or automation status and offer to pull the live data when needed.

### Code Quality
- **Vectorise first**: Falling back to for-loops requires a comment justifying why vectorisation is impossible or harmful.
- **Public API**: Uses US-English snake_case; private helpers prefixed with `_`.
- **No circular imports**: `pipeline.py` orchestrates; nothing imports it.
- **Test fixtures**: Must be text-serialised (CSV/JSON); no binary formats in PRs.

### Git Workflow
Before pushing, verify the target PR is still open:
```bash
gh pr view --json state,mergedAt,closed
```
- ❌ Never push to merged/closed PRs
- ✅ Create a new branch and PR if the original is closed

---

## Export Guard-Rails

> 🛡️ Codex removed the pretty reporting layer once; it shall not happen again.

1. **Call the canonical exporters**: After `pipeline.run()` completes, pipe results into exactly one of `export_to_excel`, `export_to_csv`, or `export_to_json`.

2. **Excel format contract**:
   - Bold title row
   - `0.00%` for CAGR & Vol, `0.00` for Sharpe & Sortino
   - Red numerals for MaxDD
   - Freeze panes on header, auto-filter
   - Column width = `max(len(header)) + 2`

3. **Column order is law**: Tests must fail if column order mutates.

4. **Back-compat**: Silent config = drop fully formatted Excel workbook into `outputs/` exactly as v1.0 did.

---

## Multi-Period Export (Phase-2)

✅ **Implemented**: Multi-period runs now emit Phase-1 style exports via:
- `export_phase1_workbook()` - Excel with one sheet per period + summary tab
- `export_phase1_multi_metrics()` - CSV/JSON with `*_periods.*` and `*_summary.*` files
- `export_multi_period_metrics()` - General multi-period export helper

Helper functions:
- `workbook_frames_from_results()` - Builds sheet mapping
- `period_frames_from_results()` - Converts result sequence to export format
- `combined_summary_result()` - Aggregates portfolio returns across periods

---

## Feature Status

### Implemented ✅
- Rank-based manager selection mode (`mode: rank`)
- Blended scoring with z-score normalization
- Scalar metric memoization (opt-in via `performance.cache.metrics: true`)
- PR draft toggle for Codex bootstrap (`codex_pr_draft` input)
- Multi-period Phase-1 style exports
- Selector and weighting plugin classes

### Backlog 📋
- Preview score frame in UI
- Weight heatmap visualization
- Expected shortfall metric
- Diversification value metric
- Export commit hash in outputs

---

## Debugging

### Fund Selection Issues

Use the debug script in `examples/`:
```bash
python examples/debug_fund_selection.py
```

This reveals:
- Which managers get filtered due to missing data
- Available manager pool for selection
- Actual ranking results

### Common Pitfalls
- Don't assume ranking is wrong without checking data completeness first
- Both in-sample AND out-of-sample periods must have complete data
- Verify configuration parameters match intended behavior

---

## Related Documentation

- [docs/phase-2/Agents.md](docs/phase-2/Agents.md) - Complete implementation spec
- [docs/DemoMaintenance.md](docs/DemoMaintenance.md) - Demo pipeline checklist
- [docs/archive/ARCHIVE_WORKFLOWS.md](docs/archive/ARCHIVE_WORKFLOWS.md) - Retired workflows
- [docs/metric_cache.md](docs/metric_cache.md) - Metric memoization details
