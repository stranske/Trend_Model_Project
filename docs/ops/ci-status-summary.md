## Post CI Status Summary Workflow

The repository maintains a single "Automated Status Summary" comment on every
pull request using `.github/workflows/maint-30-post-ci-summary.yml`. The
workflow listens for both the **CI** and **Docker** pipelines to complete and
then refreshes the comment in-place.

### What the comment shows

* Head commit SHA for the PR tip.
* Latest run references for both CI and Docker (including links and attempt
  numbers).
* Required status indicators for CI test, style, automation, gate checks, and
  the Docker workflow.
* A consolidated job table covering both pipelines, sorted so failures float to
  the top.
* Coverage overview lines (latest averages + deltas) followed by the rendered
  `coverage_summary.md` artifact when available.
* A trailing `_Updated automatically_` footer so reviewers know it self-heals on
  subsequent runs.

### Anti-spam behaviour

* Concurrency is keyed by head SHA (`post-ci-summary-${sha}`) so only the latest
  in-flight update survives.
* The comment always starts with `<!-- post-ci-summary:do-not-edit -->` and the
  search heuristic still keys off the "Automated Status Summary" header. Either
  hint is sufficient for future maintenance scripts to locate the thread.
* Empty coverage artifacts are skipped gracefully so documentation-only PRs
  avoid noisy failures.

### Required vs optional jobs

The required signals remain identical to the legacy implementation:

* CI tests (`main / tests` matrix)
* Workflow automation smoke tests (`workflow / automation-tests`)
* Style & lint jobs (`main / style`)
* The CI gate (`gate / all-required-green`)
* Docker workflow outcome (lint + build)

Everything else is treated as informational in the combined job table. Update
these groupings directly inside `maint-30-post-ci-summary.yml` if the CI layout
changes.

### Adjusting behaviour

* To add or remove required job groupings, tweak the `groups` array in the
  "Prepare summary body" step of `maint-30-post-ci-summary.yml`.
* To append additional artifact data (e.g. coverage dashboards or failure
  snapshots), extend the download + parsing steps before the comment body is
  composed.
* To disable the comment entirely, delete the workflow file; the next CI or
  Docker completion will no longer post updates.

### Docker workflow skip reminder

`pr-12-docker-smoke.yml` still uses `paths-ignore` to avoid running for docs-only
changes. When it is skipped the summary comment shows `Docker: ⏳ pending`. This
behaviour is intentional and prevents flaky Docker builds from blocking
non-Docker work.
