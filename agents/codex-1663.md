# Issue 1663 – Docker smoke workflow upkeep

## Verification checklist

- [x] Workflow still identified as **Docker** and keeps existing triggers for push, pull_request, workflow_call, and dispatch events.
- [x] Lint job runs `hadolint` only, ensuring the Dockerfile check remains lean.
- [x] Buildx cache restored and saved with the deterministic `requirements.lock` hash key, with cache paths centralised under `/tmp/.buildx-cache`.
- [x] Docker image built once via Buildx and reused for pytest and health validation runs.
- [x] Smoke tests execute `pytest -q` (or verbose mode when debugging) inside the freshly built container image.
- [x] Health endpoint probed after tests to guarantee the container starts cleanly.
- [x] Registry login and image push steps gated so they only execute on `phase-2-dev` pushes, skipping PRs and other refs.
- [x] Step names, cache keys, and inline comments reflect repository conventions to keep maintenance straightforward.

These checks confirm the workflow satisfies the acceptance criteria while remaining aligned with repository expectations for deterministic caching and smoke coverage.
