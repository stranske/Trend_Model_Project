# Keepalive Status - PR #5168

## Scope
Closes #5168.

## Checklist Reconciliation
Reviewed recent commit `35fcf506` before continuing. It updated `config/scenarios/monte_carlo/cost_regime_example.yml` and `tests/monte_carlo/test_registry.py`, which satisfies both task items about canonical direct regime blocks and the registry assertion.

## Tasks
- [x] Clarify that canonical Monte Carlo cost regimes are direct top-level blocks under costs, not a nested regimes structure.
- [x] Add a registry test assertion that the known-good cost_regime_example scenario uses the canonical direct-regime shape.

## Acceptance Criteria
- [ ] `UV_CACHE_DIR=/private/tmp/uv-cache uv run --with pytest-xdist --with pytest-cov pytest tests/monte_carlo/test_costs.py --no-cov`

## Verification
- Added stricter canonical-shape assertions in `tests/monte_carlo/test_costs.py`:
  - validates `default_regime == "calm"`
  - validates `costs.calm` and `costs.stress` are direct mapping blocks
- Ran: `pytest tests/monte_carlo/test_costs.py -m "not slow" -q` (pass: 10 tests)

## Blockers
- The required acceptance command cannot be executed in this environment because `uv` is not installed (`/bin/bash: uv: command not found`).

## Progress
2/3 checklist items complete.
