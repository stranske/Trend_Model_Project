# Agents instruction index

This directory contains **active** Codex/agent guidance tied to **open** issues only.
Files for closed issues are archived to `archives/agents/` with a date prefix.

## Current active files

| File | Issue | Status |
|------|-------|--------|
| _None_ | — | — |

## Maintenance

See `MAINTENANCE.md` for archival criteria. When an issue closes:

1. Run `scripts/archive_agents.sh` to auto-archive closed-issue files
2. Or manually: `mv codex-NNNN.md ../archives/agents/$(date +%Y-%m-%d)-codex-NNNN.md`

## Canonical sources

- `.github/workflows/README.md` — workflow inventory and naming rules
- `docs/ci/AGENTS_POLICY.md` — protection contract and verification steps

## Archive history

- **2025-11-30**: Bulk archived 415 codex files and 18 ledger files (all closed issues)
- **2025-11-22**: Archived 4 files referencing retired workflow names
