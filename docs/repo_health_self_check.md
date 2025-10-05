# Repository Health Workflow (maint-02)

The `maint-02-repo-health.yml` workflow keeps the repository's
baseline governance assets (labels, secrets, and Ops issue plumbing) healthy.
It runs nightly with a weekly deep-dive, can be dispatched manually, and
re-executes automatically whenever the probe or workflow definition changes in a
PR.

## What it verifies

| Aspect | What It Verifies | Failure Handling |
| ------ | ---------------- | ---------------- |
| Workflow lint | `actionlint` pinned at v1.7.7 | Fails the job and records the failure in the Ops summary |
| Labels | Presence of required labels (`agent:*`, `priority:*`, `tech:coverage`, `workflows`) | Missing labels are listed in the Ops issue comment |
| Secrets | `SERVICE_BOT_PAT` repository secret exists | Missing secret is reported in summary + Ops issue |
| Variables | `OPS_HEALTH_ISSUE` repository variable exists | Summary warns and Ops issue update is skipped when absent |

## Trigger modes

- **Daily cron** — 04:17 UTC for routine hygiene.
- **Weekly cron** — Monday 06:45 UTC for the extended Ops report.
- **workflow_dispatch** — Manual runs for smoke tests or post-incident
  verification.
- **Scoped PR / push triggers** — Changes to `tools/repo_health_probe.py` or the
  workflow definition automatically re-run the probe on the
  `phase-2-dev` branch.

## Sample step summary

```
## Repo health nightly checks

- ✅ Workflow lint (`actionlint`) succeeded.
- ✅ Required labels, variables, and secrets are present.
```

When the probe encounters missing artefacts, the summary switches to an error
list and (if `OPS_HEALTH_ISSUE` is set) the workflow posts an update to the Ops
tracking issue via a replace-in-place comment (`<!-- repo-health-nightly -->`).

## OPS_HEALTH_ISSUE maintenance

The probe expects the repository variable (or secret) `OPS_HEALTH_ISSUE` to
contain the numeric identifier of the Operations tracking issue. Confirm the
variable is defined in production and staging forks — the workflow will warn in
its summary and skip the comment update if the value is missing or malformed.

## Offline smoke coverage

`scripts/workflow_smoke_tests.py` exercises `tools.repo_health_probe` using a
fixture payload so CI validates the probe without GitHub API access. Run the
smoke harness locally with:

```bash
python scripts/workflow_smoke_tests.py
```

The command prints the same Markdown summary that appears in Actions logs,
providing a quick regression check when adjusting probe logic.
