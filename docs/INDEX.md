# Documentation index

Use this index to find the current contributor guides and to understand which overlapping docs remain for historical context.

## Overlapping docs and their scopes
| Document | Audience | Scope/status |
| --- | --- | --- |
| `README.md` | Contributors running the CLI and demos | Quickstart for editable installs and CLI usage; retains legacy actionlint appendix that is being superseded by this index. |
| `README_APP.md` | Streamlit app users and simulation maintainers | Install, layout, and Streamlit/CLI integration notes for the packaged app. |
| `README_DATA.md` | Anyone using bundled datasets | Provenance, intended use, and validation contract for the demo CSVs. |
| `TESTING_SUMMARY.md` | Test authors and reviewers | Upload app test coverage, commands, and portfolio app coverage notes. |
| `Issues.txt` | Automation/backlog tooling | Structured list of engine/config cleanup tasks consumed by parser tests. |
| `ROBUSTNESS_GUIDE.md` | Historical reference | Stub pointing to the archived robustness how-to at `archives/docs/ROBUSTNESS_GUIDE.md`. |
| `docs/README.md` | Documentation entrypoint | Replaces the vendored actionlint README; routes readers here. |

## Canonical navigation

### Onboarding and workflow
- `README.md` for CLI quickstart and demo pipeline usage.
- `docs/WORKFLOW_GUIDE.md` plus `docs/ci/WORKFLOW_SYSTEM.md` for CI/workflow topology and maintenance.

### Testing
- `TESTING_SUMMARY.md` for the upload and portfolio app testing ledger.
- `scripts/run_tests.sh` for the standard suite entrypoint referenced across CI jobs.

### Robustness
- Archived robustness guide at `archives/docs/ROBUSTNESS_GUIDE.md` until a refreshed version replaces it. Pair it with the weighting options described in `docs/UserGuide.md` when wiring new strategies.

### Data and app flows
- `README_DATA.md` for bundled data constraints and validation helpers.
- `README_APP.md` for Streamlit app packaging and presets.

### Repository hygiene
- `docs/repository_housekeeping.md` for archiving rules, quarterly checklists, and folder ownership.
