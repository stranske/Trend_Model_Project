# Repository organization audit – 2025-11-25

## Scope
- Re-ran the organizational prompt to validate whether existing checklists and archival rules remain satisfied without changing the repo structure.
- Reviewed the root log/status files, keepalive status index, and notebook directory expectations to confirm housekeeping guidance is being followed.

## Findings
1. **Root snapshots needed archival copies.** The most recent `coverage-summary.md`, `gate-summary.md`, and `keepalive_status.md` files were sitting at the root without dated copies in `archives/generated/`. The housekeeper index (`archives/ROOT_FILE_INDEX.md`) still referenced only the 2025-11-22 sweep, so historical traceability lagged behind the latest keepalive round.
2. **Keepalive status layout remains compliant.** Active checklists live under `docs/keepalive/status/` with no overlapping edits. The root `keepalive_status.md` still lists the current runs and directs writers to create per-run files to avoid PR conflicts.
3. **Notebook maintenance checklist still holds.** `notebooks/README.md` lists a single maintained notebook and an archive path for explorations. The tree matches that contract (one live notebook, archives kept under `archives/notebooks/2025`). No stray experiment notebooks were found outside the archive.

## Remediation performed
- Captured dated archival copies of the current root snapshots under `archives/generated/2025/` and updated `archives/ROOT_FILE_INDEX.md` to record the new sweep for traceability.

## Recommended follow-ups
1. **Continue quarterly sweeps.** Run the `docs/repository_housekeeping.md` checklist during the first week of each quarter to decide whether the new 2025-11-25 snapshots should remain at the root or be converted into stubs that point directly to the archives.
2. **Log future keepalive rounds promptly.** When new keepalive runs start, add a new file under `docs/keepalive/status/` and append it to `keepalive_status.md` so the status index stays current before the next archival sweep.
3. **Track emerging diagnostic catalogues.** If the diagnostics keepalive spawns additional catalogues (like `docs/keepalive/status/diagnostics_catalogue.md`), decide whether they belong in the status index or an appendix to prevent the root index from drifting from the active files.
