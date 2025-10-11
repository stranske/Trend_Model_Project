<!-- bootstrap for codex on issue #2196 -->

## Task Checklist

- [x] Create `.github/workflows/autofix-versions.env` with pinned formatter/type versions.
- [x] Source the env file across CI workflows (`pr-gate.yml`, `reusable-ci.yml`, `maint-32-autofix.yml`) and the composite autofix action.
- [x] Mirror the pins in local developer tooling (`scripts/style_gate_local.sh`, `scripts/dev_check.sh`, `scripts/validate_fast.sh`).
- [x] Document the env-file flow in `docs/ci/WORKFLOWS.md` and note the failure mode when pins are missing.
- [x] Validate parity by running `./scripts/style_gate_local.sh` and ensuring pytest workflow naming guard stays green.
