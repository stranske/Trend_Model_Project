# Agent Issue Template

Use this template when creating issues for Codex or ChatGPT agents.

---

## Template

```markdown
## Why

[Explain the motivation or problem being solved]

## Scope

[Define what is included in the work]

## Non-Goals

[What is explicitly out of scope - optional section]

## Tasks

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Implementation notes

[Technical details, file paths, branch names - optional section]
```

---

## Required Labels

Apply **exactly one** agent assignment label:
- `agent:codex` - For autonomous coding tasks
- `agent:chatgpt` - For research, documentation, or planning tasks

Optional labels:
- `priority: high|medium|low`
- Category labels (`enhancement`, `bug`, `documentation`, etc.)
- `agents:keepalive` - For workflow health checks

---

## Example: Bug Fix Issue

```markdown
## Why

The `vol_adj_enabled` setting in the UI has no observable effect on portfolio weights.
Users expect enabling volatility adjustment to change the portfolio allocation.

## Scope

Wire `vol_adj_enabled` and related settings (`vol_window_length`, `vol_window_decay`, 
`vol_ewma_lambda`) through the pipeline so they affect weight calculations.

## Non-Goals

- Changing the volatility calculation algorithm
- Adding new volatility models

## Tasks

- [ ] Trace `vol_adj_enabled` from UI state to Config
- [ ] Verify Config.volatility is passed to pipeline
- [ ] Add volatility adjustment to weight calculation in pipeline.py
- [ ] Add wiring test to verify setting has effect

## Acceptance criteria

- [ ] Changing `vol_adj_enabled` produces different portfolio weights
- [ ] Settings wiring test passes for volatility settings
- [ ] No regression in existing tests

## Implementation notes

Files:
- `streamlit_app/components/analysis_runner.py` - Config building
- `src/trend_analysis/pipeline.py` - Weight calculation
- `scripts/test_settings_wiring.py` - Validation tests

Branch: `fix/volatility-settings-wiring`
```

---

## Example: Feature Request Issue

```markdown
## Why

The hard entry/exit thresholds (`z_entry_hard`, `z_exit_hard`) are defined in the UI
but have no implementation in the selection logic.

## Scope

Implement hard z-score thresholds that provide absolute barriers for fund entry/exit,
complementing the existing soft threshold + strikes system.

## Non-Goals

- Changing the soft threshold behavior
- Adding new threshold types

## Tasks

- [ ] Add hard threshold logic to fund selection in multi_period/engine.py
- [ ] Wire z_entry_hard and z_exit_hard from Config to selection logic
- [ ] Add unit tests for hard threshold behavior
- [ ] Add wiring test to verify settings have effect

## Acceptance criteria

- [ ] Funds below z_entry_hard are never selected regardless of score
- [ ] Funds above z_exit_hard are never removed regardless of score
- [ ] Settings wiring tests pass for hard threshold settings
- [ ] Existing soft threshold behavior unchanged

## Implementation notes

Related to: #4025 (signals not wired)

Files:
- `src/trend_analysis/multi_period/engine.py` - Selection logic
- `src/trend_analysis/config.py` - Config definitions
```

---

## Section Reference

| Section | Required | Description |
|---------|----------|-------------|
| Why | ✅ | Motivation or problem statement |
| Scope | ✅ | What's included in the work |
| Non-Goals | ❌ | What's explicitly excluded |
| Tasks | ✅ | Checklist of work items |
| Acceptance criteria | ✅ | Definition of done |
| Implementation notes | ❌ | Technical details, file paths |

---

## See Also

- [Issue Format Guide](../ci/ISSUE_FORMAT_GUIDE.md) - Detailed format specifications
- [AGENTS_POLICY.md](../AGENTS_POLICY.md) - Workflow protection policy
- [agent-automation.md](../agent-automation.md) - Automation overview
