# Workflow Topology & Agent Routing Guide (WFv1)

This guide documents how GitHub Actions workflows are organized under the WFv1
naming scheme, how to stage new or renamed workflows without breaking triggers,
and how post-CI reporting and agent orchestration work. Use it when reviewing
workflow contributions or opening issues that need Codex or Copilot support.

## WFv1 naming scheme

The repository follows the "Workflow File naming v1" (WFv1) convention: each
workflow filename starts with a category prefix followed by a two-digit index
and a descriptive slug. The pattern keeps related workflows grouped in GitHub's
UI and makes diffs easier to scan.

### Category buckets

| Prefix | Purpose | Examples |
| ------ | ------- | -------- |
| `pr-`  | Pull-request facing CI wrappers that call reusable jobs | `pr-10-ci-python.yml`, `pr-12-docker-smoke.yml` |
| `agents-` | Agent assignment, watchdogs, and helper utilities | `agents-41-assign.yml`, `agents-42-watchdog.yml`, `agents-40-consumer.yml` |
| `maint-` | Maintenance, reporting, governance, and health monitors (indices cluster by theme: `maint-30` post-CI, `maint-33` failure tracking, `maint-34` quarantine, etc.) | `maint-30-post-ci-summary.yml`, `maint-33-check-failure-tracker.yml`, `maint-35-repo-health-self-check.yml`, `maint-36-actionlint.yml` |
| `autofix-` | Autofix infrastructure (composite inputs, environment pins) | `autofix.yml`, `autofix-versions.env`, `autofix-residual-cleanup.yml` |
| `reuse-` | Reusable workflow entrypoints consumed by wrappers | `reusable-ci-python.yml`, `reusable-90-agents.yml` |
| `verify-` | Manual probes or validation suites | `verify-ci-stack.yml`, `verify-codex-bootstrap-matrix.yml` |
| `codeql` / `dependency-review` / `workflow-lint` | Security & linting workflows that retain legacy slugs for integration | `codeql.yml`, `dependency-review.yml`, `workflow-lint.yml` |
| `release` / `stale` / `cleanup` / `perf` | Operational jobs that predate WFv1 but retain canonical names | `release.yml`, `stale-prs.yml`, `cleanup-codex-bootstrap.yml`, `perf-benchmark.yml` |

Follow these buckets when adding new workflows. Reuse an existing prefix when
possible; introduce a new bucket only when a new lifecycle warrants it.

### Naming checklist for new workflows

1. Pick the prefix that matches the workflow's scope.
2. Use a two-digit index that slots into the existing sequence (e.g. the next
   available `maint-3x` number for new maintenance automation).
3. Keep the remainder of the slug concise but descriptive ("ci-python", "assign",
   "post-ci-summary").
4. Update the workflow inventory documentation (this guide and
   `.github/workflows/README.md`) when adding, renaming, or retiring workflows.

## Staging new or renamed workflows

Maintaining trigger fidelity requires a bit of bookkeeping whenever workflow
files move or new ones are introduced:

1. **Draft the workflow under WFv1.** Create the file in `.github/workflows/`
   using the naming checklist above. Co-locate supporting files (composite
   actions, environment manifests) under the same prefix family when possible.
2. **Run schema and repository linting.** Commit the file locally and run
   `gh workflow lint` or push to a branch to let `maint-36-actionlint.yml`
   validate the syntax. For major refactors, run
   `pytest tests/test_workflow_*.py` to exercise guard rails that protect
   required slugs.
3. **Wire the triggers.** For workflow_run followers (post-CI summary, failure
   tracker, autofix) update the `workflows:` list so the new producer job is
   observed. When renaming a workflow, update any downstream consumers—
   especially `maint-30-post-ci-summary.yml`,
   `maint-33-check-failure-tracker.yml`, and `merge-manager.yml`—to point to the
   new name. Treat the gate job (`gate / all-required-green` inside
   `pr-10-ci-python.yml`) as the single source of truth for CI requirements.
4. **Update documentation.** Add a note to this guide and cross-link it from
   `.github/workflows/README.md`. Mention behavior changes in the PR summary so
   automation reviewers can spot them.
5. **Stage via branch + PR.** Push the branch and open a PR. The CI suite and
   post-CI summary will validate the workflow wiring before merge.

### Quick staging checklist

- [ ] New or renamed workflow listed in `.github/workflows/README.md` and this
      guide.
- [ ] Downstream `workflow_run` consumers updated (`maint-30`, `maint-33`,
      `merge-manager.yml`, and any custom followers).
- [ ] Tests that lock workflow names (for example, `tests/test_workflow_*.py`)
      updated when introducing or retiring slugs.
- [ ] Failure-tracker label taxonomy confirmed or refreshed if new failure
      signatures are expected.
- [ ] Manual workflow dispatches (if applicable) verified in a fork before
      requesting review.

## Post-CI status summaries & failure tracking

Two maintenance workflows keep pull requests informed about CI health:

- **`maint-30-post-ci-summary.yml`** listens for `workflow_run` events from the
  `CI` and `Docker` workflows. It aggregates the latest runs for the head SHA
  and posts (or updates) a summary comment containing each job's result and log
  link. It cancels redundant runs per SHA to avoid spam and can be tuned via the
  `WORKFLOW_TARGETS_JSON` environment variable. The comment carries the marker
  `<!-- post-ci-summary:do-not-edit -->`; reruns replace the same comment so the
  conversation stays tidy.
- **`maint-33-check-failure-tracker.yml`** also subscribes to CI/Docker
  `workflow_run` events. When a run fails it calculates a stable signature from
  the failed jobs, ensures the `ci-failure` issue label taxonomy exists, and
  opens or updates a tracking issue with the failure table and stack-token
  hints. The issue auto-heals once a succeeding run clears the signature for 24
  hours. Each tracker update links back to the PR and run that produced the
  signature so regressions can be followed across branches.

Because the gate workflow (`gate / all-required-green`) depends on upstream jobs
being green, a failed `lint / black-fast` job will cause both the gate check and
post-CI summary to report failures. Use the summary comment to jump directly to
logs, and watch the failure-tracker issue for cross-run history. Once the gate
check flips to ✅ the failure tracker will close automatically on the next
passing cycle.

## Agent orchestration & labels

Agent automation relies on label-driven routing:

- **Assignment.** `agents-41-assign.yml` forwards issue and PR label events to
  the unified orchestrator `agents-41-assign-and-watch.yml`. When a maintainer
  applies `agent:codex` or `agent:copilot`, the orchestrator assigns the
  corresponding triage team, boots Codex issue branches when needed, and kicks
  off the watchdog timer.
- **Watchdog.** `agents-42-watchdog.yml` preserves the historical manual
  watchdog interface while delegating to the unified workflow. It ensures a PR
  is linked to the issue within the configured timeout; otherwise it posts a
  diagnostic comment and can escalate.
- **Readiness & drills.** `agents-40-consumer.yml` wraps the reusable
  `reusable-90-agents.yml` so operators can run readiness checks or diagnostic
  drills on demand.
- **Label origins.** `.github/agent-label-rules.json` ensures automation-created
  PRs receive both `from:codex`/`from:copilot` and the matching `agent:*`
  labels. Manual issues should set the right agent label up front so the
  assigner triggers immediately.

When opening a new issue, pick the agent label that matches the assistance you
need:

- `agent:codex` → Codex authored or assisted work.
- `agent:copilot` → Copilot authored or assisted work.

Only one agent label should be present on an issue at a time. Remove any stale
label before switching to the other agent to prevent conflicting assignments.

### Quick reference: agent request checklist

When filing an issue that requires Codex or Copilot support:

- [ ] Pick the issue template that matches your request (bug vs. feature).
- [ ] Ensure the sidebar shows exactly one agent label
      (`agent:codex` **or** `agent:copilot`).
- [ ] Remove the other agent label if it was auto-applied from a previous
      request so assignment does not stall.
- [ ] Submit the form only after the confirmation checkbox passes—automation
      will fail fast when both agent labels are present.

## Cross-references & additional resources

- `.github/workflows/README.md` – Architectural overview and onboarding
  checklist for CI, agents, and maintenance workflows.
- `docs/ci-failure-tracker.md` – Deep dive into the failure-tracker issue flow.
- `docs/agent-automation.md` – Additional agent automation internals and guard
  rails.
