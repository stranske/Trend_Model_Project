# Issue 2199 – Docker smoke workflow guard rails

## Scope and constraints
- Keep the existing buildx cache semantics. The restore step must always run and the save step stays disabled for pull requests.
- Total runtime must remain short; the smoke job timeout cannot exceed 10 minutes and the health probe should fail within seconds when the container is unhealthy.
- Changes apply to both CI (.github/workflows/pr-12-docker-smoke.yml) and the matching local helper (scripts/docker_smoke.sh).

## Acceptance criteria
- [x] A health check runs immediately after the image build, targeting `${HEALTH_PORT}${HEALTH_PATH}` and fails the job quickly when the service is unhealthy.
- [x] The smoke job timeout is capped at 10 minutes or less.
- [x] Cache saving only executes on non-pull_request events while restore always runs.

## Implementation checklist
- [x] Launch the container with a unique name and ensure cleanup even on failure.
- [x] Poll the health endpoint with retry + backoff capped under 10 seconds total.
- [x] Parse JSON/plain-text health responses and capture diagnostics on failure.
- [x] Share the health-response parser between CI and the local smoke helper to avoid drift.
- [x] Mirror the CI guard rails in scripts/docker_smoke.sh for local parity.
- [x] Validate YAML syntax (`python -c 'yaml.safe_load(...)'`) and shell syntax (`bash -n scripts/docker_smoke.sh`).
