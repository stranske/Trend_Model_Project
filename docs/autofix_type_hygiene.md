# Automated Autofix & Type Hygiene Pipeline

This repository includes an extended **autofix** workflow that standardises style and performs *lightweight* type hygiene automatically on pull requests. The entrypoint is `.github/workflows/autofix.yml` (PR trigger), which shares the same reusable composite used by the post-CI follower `maint-post-ci.yml`.

## What It Does (Scope)
1. Code formatting & style
   - `ruff` (lint + --fix for safe rules)
   - `black` (code formatting)
   - `isort` (import sorting where unambiguous)
   - `docformatter` (docstring wrapping)
2. Type stub installation
   - Runs `mypy --install-types --non-interactive` to fetch missing third‑party stubs where available.
3. Targeted type hygiene
   - Executes `scripts/auto_type_hygiene.py` which injects `# type: ignore[import-untyped]` ONLY for a small allowlist of third‑party imports (default: `yaml`).
   - Idempotent: re-running does not duplicate ignores.
4. Warm mypy cache
   - Invokes a non-blocking mypy run so subsequent CI/type checks are faster.

If any step produces changes, the workflow auto‑commits them back to the PR branch with a conventional message (e.g. `chore(autofix): style + type hygiene`).

## What It Intentionally Does NOT Do
- It does **not** attempt deep structural refactors or resolve complex type inference issues.
- It does **not** add stubs for internal modules or apply `# type: ignore` broadly.
- It does **not** silence genuine semantic/type errors (e.g. wrong argument counts, incompatible assignments).
- It does **not** enforce exhaustive strict mypy modes across legacy areas not yet migrated.

This narrow scope keeps the automation safe, deterministic, and low‑noise.

## Extending the Allowlist
The allowlist for untyped third‑party imports lives in `scripts/auto_type_hygiene.py`:
```python
ALLOWLIST = {"yaml"}
```
To add another untyped module (e.g. `fastapi` if desired):
1. Edit the set to include the module root name.
2. Run:
   ```bash
   python scripts/auto_type_hygiene.py --check
   ```
3. Commit the resulting changes (if any).

Prefer adding *only* modules that reliably lack published stubs or where partial typing would otherwise generate persistent noise.

## Local Developer Workflow
During active development:
```bash
./scripts/dev_check.sh --changed --fix   # ultra-fast sanity (2-5s)
./scripts/validate_fast.sh --fix         # adaptive validation (5-30s)
./scripts/run_tests.sh                   # full test suite (15-25s)
```
Before pushing a feature branch:
```bash
./scripts/validate_fast.sh --fix
./scripts/run_tests.sh
```
If CI leaves an autofix commit on your branch, **pull/rebase** before adding further changes.

## When To Escalate Beyond Automation
Open a focused PR (or issue) for:
- Introducing or refactoring complex protocols / generics.
- Replacing dynamic imports with explicit optional dependency shims.
- Tightening mypy configuration (e.g. enabling `disallow-any-generics`).
- Broad import hygiene sweeps beyond the allowlist.

## Design Principles
| Principle | Rationale |
|-----------|-----------|
| Idempotent | Re-running produces no further diffs when clean. |
| Minimal Surface | Only low-risk fixes applied automatically. |
| Deterministic | Output stable across environments. |
| Transparent | All mutations committed with explicit chore message. |
| Extensible | Allowlist easily adjusted with single source of truth. |

## Troubleshooting
| Symptom | Likely Cause | Action |
|---------|--------------|-------|
| Repeated autofix commits | Unformatted notebooks (black lacks jupyter extra) | Install `black[jupyter]` locally or exclude notebooks. |
| New mypy errors after a rebase | Upstream typing tightened | Resolve manually; avoid blanket `# type: ignore` unless justified. |
| Missing ignore for known untyped lib | Not in allowlist | Add to `ALLOWLIST` in script, run autofix locally. |
| CI autofix skipped | No diff produced | Confirm local environment replicates tool versions. |

## Future Enhancements (Optional Backlog)
- Add metrics: record autofix delta lines per run.
- Dry-run mode for PR comments instead of commits (toggle by label).
- Expand hygiene to detect *unused* ignores and prune them automatically.

---
Maintainers: keep this doc aligned with any future changes to `.github/actions/autofix/action.yml` or `scripts/auto_type_hygiene.py` so contributors understand the automation contract.
