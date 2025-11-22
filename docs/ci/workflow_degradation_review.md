# Workflow Reliability Review

## Round 1 – Failure causes
- **Gate checkout blocked on forked PRs.** The Gate workflow checked out the pull request head SHA without switching to the contributor's repository, so forked PRs could not fetch the commit and the workflow aborted before CI ran.【F:.github/workflows/pr-00-gate.yml†L31-L38】
- **Autofix loop fails without private credentials.** The CI autofix workflow hard-requires the `ACTIONS_BOT_PAT` secret even when running on forked PRs that cannot access repository secrets, causing the workflow to stop before it can prepare a branch or patch artifact.【F:.github/workflows/autofix.yml†L353-L372】

## Round 2 – Optimization and hardening opportunities
- **Default to fork-safe checkouts.** The Gate workflow now targets the contributing repository when a PR event supplies one; this prevents fork checkouts from failing and lets the detection logic proceed consistently.【F:.github/workflows/pr-00-gate.yml†L31-L38】 Extending the same pattern to other head-specific checkouts (e.g., future helpers) would keep the CI matrix reliable across contributor types.
- **Graceful autofix fallback.** Relaxing the PAT requirement in the autofix workflow (e.g., switching to patch-only mode with `GITHUB_TOKEN` when secrets are unavailable) would keep the job green on forks while still producing actionable artifacts. The current hard failure gate is the primary source of autofix run stops.【F:.github/workflows/autofix.yml†L353-L372】
- **Compute-job reuse for CI health.** The orchestrator already enforces a concurrency gate and reuses shared scripts; applying similar reuse/caching to the Python CI installs (currently reinstalling toolchains per run) would shorten runtimes and reduce timeouts on large PRs.【F:.github/workflows/pr-00-gate.yml†L86-L103】

## Round 3 – Automation and agent-management capabilities
- **Central orchestration with keepalive guardrails.** The Agents 70 Orchestrator drives scheduled and event-driven automation, evaluating keepalive prerequisites (labels, run caps, activation) before dispatching work so multiple agents can coordinate safely.【F:.github/workflows/agents-70-orchestrator.yml†L1-L80】【F:.github/workflows/agents-70-orchestrator.yml†L430-L520】
- **Gate-level concurrency and scope detection.** The Gate workflow uses per-PR concurrency and change detection to decide which CI lanes to run (Python matrix, Docker, scripts) before aggregating a single required status, keeping repo health checks focused on relevant changes.【F:.github/workflows/pr-00-gate.yml†L6-L103】【F:.github/workflows/pr-00-gate.yml†L150-L185】
- **Scheduled health signals for sustained runs.** Keepalive and orchestrator checks enforce run caps and annotate PRs with the current round/trace, enabling long-running multi-agent tasks to progress across multiple runs without manual babysitting.【F:.github/workflows/agents-70-orchestrator.yml†L430-L520】
