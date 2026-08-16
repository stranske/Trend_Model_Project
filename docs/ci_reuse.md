# Shared CI & Automation Ownership

Trend Model Project consumes shared workflow behavior from
[`stranske/Workflows`](https://github.com/stranske/Workflows). The canonical
consumer template is `templates/consumer-repo/` in that repository. Reference
that template and its validation tools instead of copying shared workflow
implementations into this guide.

## Local entry points

| Concern | Local surface |
| --- | --- |
| Required PR gate | `.github/workflows/pr-00-gate.yml` |
| Python CI caller | `.github/workflows/ci.yml` |
| Agent issue intake | `.github/workflows/agents-issue-intake.yml` |
| Codex queue and delivery | `.github/workflows/agents-71-codex-belt-dispatcher.yml`, `agents-72-codex-belt-worker.yml`, `agents-73-codex-belt-conveyor.yml` |
| PR and Gate event routing | `.github/workflows/agents-80-pr-event-hub.yml`, `agents-81-gate-followups.yml` |
| Keepalive | `.github/workflows/agents-keepalive-sweep.yml`, `agents-keepalive-loop-reporter.yml` |

The files above are deployed consumer surfaces. Shared implementation changes
belong in Workflows first and reach this repository through the managed sync
path. Product-specific CI behavior remains local when it is not part of the
consumer template.

## Agent dispatch

Apply a registered agent label or dispatch the local intake workflow:

```bash
gh workflow run agents-issue-intake.yml \
  --field mode=agent_bridge \
  --field issue_number=NUMBER
```

Automation-created and reused PRs are always ready for review. Use labels,
required checks, disabled auto-merge, and exact-head lifecycle metadata for
holds; do not use draft state as a queue or dependency marker.

## Validation

1. Compare the deployed inventory with the Workflows consumer template and
   manifest.
2. Run the Workflows template-sync and completeness validators for shared
   changes.
3. Run this repository's focused workflow contract tests and Actionlint.
4. Verify the exact PR head has green required checks and no active review
   threads before merge.

See [`.github/workflows/README.md`](../.github/workflows/README.md) for the
quick local inventory and [`docs/ci/WORKFLOWS.md`](ci/WORKFLOWS.md) for the
detailed active catalog.
