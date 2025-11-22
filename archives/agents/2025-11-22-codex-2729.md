# Issue #2729: CI/CD and Branch Protection Checklist

## Required check before merge: Gate
- [ ] Ensure the "Gate" status context (`gate`) is required on `phase-2-dev` and passing before merge.

## Informational: Gate summary comment
- [ ] Review the Gate summary comment after Gate succeeds to confirm the aggregated CI results and keep it informational.

## How to configure branch protection in GitHub
- [ ] Follow [GitHub's branch protection guide](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-protected-branches).
- [ ] Require **Gate / gate** as a status check and keep **Health 45 Agents Guard / Enforce agents workflow protections** enforced for protected workflow edits.

## Reference documentation
- See [docs/ci/WORKFLOW_SYSTEM.md](../docs/ci/WORKFLOW_SYSTEM.md#required-vs-informational-checks-on-phase-2-dev) for full workflow details.

## Checklist
- [ ] All required checks are configured and passing.
- [ ] Branch protection rules match the documented settings.
- [ ] Gate summary comment has been reviewed.
