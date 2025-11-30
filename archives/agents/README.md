# Archived agent instruction files

Legacy agent guidance that referenced retired workflow names, superseded policy drafts, or **closed issues** lives here for historical context. Current automation and protection rules are documented in:

- `.github/workflows/README.md` for the active workflow topology and naming policy.
- `docs/ci/AGENTS_POLICY.md` for the protection contract covering the agents workflows.

## Archive structure

```
archives/agents/
├── ledgers/              # YAML task-tracking ledgers from .agents/
│   └── issue-NNNN-ledger.yml
├── YYYY-MM-DD-codex-NNNN.md  # Archived instruction files
└── README.md
```

## Archived on 2025-11-30 (bulk cleanup)

**Ledgers** (18 files → `ledgers/`):
- All issue ledgers from `.agents/` — issues #3011, #3203, #3209, #3213, #3218, #3219, #3279, #3284, #3309, #3318, #3321, #3333, #3352, #3363, #3428, #3442, #3490, #3498 — all CLOSED

**Instruction files** (415 files):
- All codex-*.md files referencing closed issues, numbered #721 through #3879
- Excludes only `codex-3572.md` which remains active (issue OPEN)

## Archived on 2025-11-22
- `2025-11-22-codex-2682.md` – instructions focused on removing the old Agents 61/62 consumer workflows, which are no longer present.
- `2025-11-22-codex-2684.md` – bootstrap plan for drafting the original agents policy file that has since been published and maintained elsewhere.
- `2025-11-22-codex-2729.md` – branch-protection checklist tied to the phased Gate/Health 45 setup that has been replaced by the current Gate + Agents Guard enforcement described in the CI docs.
- `2025-11-22-codex-2738.md` – repository ruleset validation plan targeting the legacy `agent63_pair*.yml` workflow names, superseded by the `agents-63-issue-intake.yml` + `agents-70-orchestrator.yml` pairing.
