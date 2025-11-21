---
**Keepalive checklist snapshot (issue #3689)**

#### Scope
- [x] Add `src/utils/paths.py::proj_path(*parts)` which resolves relative to repo root (env override allowed).
- [x] Replace ad-hoc `Path.cwd()` joins in scripts and the app with `proj_path`.
- [x] Tests that simulate different CWDs to ensure stability.

#### Tasks
- [x] Implement resolver and adopt it in scripts/apps where relevant.
- [x] Add tests covering non-root CWD and Docker working dirs.

#### Acceptance criteria
- [x] All existing scripts run from any CWD without path errors.
