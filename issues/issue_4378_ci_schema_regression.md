# CI regression coverage for prompt schema validity

## Context

Acceptance criteria requires adding a CI regression check that validates schema validity
of returned prompt attachment test cases remains >= 95%. This requires updating
`.github/workflows/**`, which is protected in this agent environment.

## Required workflow change

- Add a CI job that runs the targeted schema validity test/coverage command for the
  prompt evaluation harness and fails when coverage drops below 95%.
- Ensure the job runs on PRs touching prompt evaluation logic or datasets.

<!-- needs-human: add a workflow job to run schema validity regression tests at >=95%; workflow edits required. -->
