# Agents instruction maintenance checklist

Use this checklist when deciding whether to retire or promote an instruction file in `agents/`.

## When to retire (archive)
- [ ] The file references workflow names, labels, or status checks that no longer exist in `.github/workflows/`.
- [ ] The tasks duplicate content now covered by `docs/ci/AGENTS_POLICY.md` or `.github/workflows/README.md`.
- [ ] The associated issue is closed and there is a newer bootstrap for the same workflow family.
- [ ] A replacement policy or runbook is published; add a short note about the successor before archiving.

## How to archive
- [ ] Create `archives/agents/` if it is missing.
- [ ] Move the retired file into `archives/agents/` and prefix the filename with the archive date (YYYY-MM-DD-...).
- [ ] Update `archives/agents/README.md` with a one-line rationale and the canonical source that replaces it.
- [ ] Refresh `agents/README.md` to make sure it points to the current instruction set.

## When to promote or keep
- [ ] The workflow names and enforcement layers match the current topology in `.github/workflows/README.md`.
- [ ] The instructions cover scenarios not addressed in the shared policy docs (e.g., bespoke diagnostics for an open issue).
- [ ] The guidance is still referenced by an open ledger entry in `.agents/`.
- [ ] The tasks remain prerequisites for the active automation surfaces or required CI protections.
