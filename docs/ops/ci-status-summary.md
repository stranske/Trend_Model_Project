## PR Status Summary & Docker Skip Logic

This repository provides an automated status summary comment on every PR whenever
the `CI` or `Docker` workflows complete. The comment is **idempotent** – it is
updated in-place (not re-posted) and includes:

* Head commit SHA
* Triggering workflow run name / id
* Segregated tables for Required vs Optional jobs
* Job result badge + conclusion + duration
* Explicit failures section (empty if none)
* ISO timestamp for the last refresh

### Required vs Optional

Currently the heuristic set of required jobs is: `CI` (+ `Docker` if it runs).
Future refinement can externalise this list through repository variables.

### Anti-Spam Protections

* Concurrency group keyed by head SHA cancels stale in-flight summary updates.
* Single comment identified by `<!-- pr-status-summary:do-not-remove -->` marker.
* Body diff check avoids needless comment edits when content is unchanged.

### Docker Workflow Skip

The Docker workflow (`.github/workflows/pr-12-docker-smoke.yml`) is filtered using
`paths-ignore` so it **does not run** for documentation-only or metadata
changes (patterns: `docs/**`, `**/*.md`, `.github/ISSUE_TEMPLATE/**`). This
reduces CI noise and prevents non-critical failures (e.g. image build transients)
from appearing beside code-only validations.

### Adjusting Behaviour

* Add / remove required jobs: edit the `requiredSet` in
  `.github/workflows/maint-31-pr-status-summary.yml`.
* Force Docker build for a doc change: include a dummy change outside the
  ignored paths or temporarily comment the pattern locally (not recommended for
  normal operation).
* Expand skip scope: add more glob patterns under `paths-ignore`.

### Future Enhancements (Backlog)

* Externalise required jobs via `vars.REQUIRED_JOBS` (comma-separated) with a
  fallback heuristic.
* Include elapsed end-to-end pipeline time across all jobs.
* Provide a condensed status (✅ All Required Passed / ❌ Required Failures) in
  the PR title decoration via another workflow.

---

Maintainers: Avoid manually deleting the summary comment; it will be recreated
on the next workflow completion. To disable, remove the workflow file.
