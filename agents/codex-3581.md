<!-- bootstrap for codex on issue #3581 -->

## Scope
- [ ] Keep pyproject.toml as the authoritative dependency list.
- [ ] Generate a single, pinned lock file via either pip-compile or uv pip compile (pick one).
- [ ] Remove duplicate inputs (keep only pyproject.toml + one lock).
- [ ] Update DEPENDENCY_QUICKSTART.md and DOCKER_QUICKSTART.md to the new flow.

## Tasks
- [ ] Move any version pins from requirements*.txt into pyproject.toml if missing.
- [ ] Add a `make lock` target that builds the lock with the chosen tool.
- [ ] Delete redundant requirements.txt or requirements.lock after migration; keep the generated lock only.
- [ ] Update Dockerfile to install from pyproject.toml + the lock.
- [ ] Document install flows for: local dev, Docker build, CI.

## Acceptance criteria
- [ ] Fresh venv: `pip install -e .` resolves only to versions specified in the generated lock.
- [ ] Docker build uses the lock and produces identical `pip freeze` across runs.
- [ ] DEPENDENCY_QUICKSTART.md shows a single install path and it works end-to-end.
