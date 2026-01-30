# Keepalive Status — PR #4598

## Scope
PR #4598 addressed issue #4597, but verification identified concerns (verdict: CONCERNS). This follow-up addresses remaining gaps with improved task structure.

The ledger-file policy scope is limited to detecting new `issue-*-ledger.yml` files outside the approved roots (`archives/agents/ledgers` and `.workflows-lib/.agents`) via a repository test guard, without reintroducing `.agents` ledgers.

## Tasks
- [x] Remove `.agents/issue-4160-ledger.yml` and `.agents/issue-4592-ledger.yml` files from the repository and ensure no new ledger files are added unless permitted by policy.
- [x] Remove `.agents/issue-4160-ledger.yml` (verify: confirm completion in repo)
- [x] Define scope for: `.agents/issue-4592-ledger.yml` files from the repository. (verify: confirm completion in repo)
- [x] Implement focused slice for: `.agents/issue-4592-ledger.yml` files from the repository. (verify: confirm completion in repo)
- [x] Validate focused slice for: `.agents/issue-4592-ledger.yml` files from the repository. (verify: confirm completion in repo)
- [x] Define scope for: Implement a mechanism to ensure no new ledger files are added unless permitted by policy. (verify: confirm completion in repo)
- [x] Implement focused slice for: Implement a mechanism to ensure no new ledger files are added unless permitted by policy. (verify: confirm completion in repo)
- [x] Validate focused slice for: Implement a mechanism to ensure no new ledger files are added unless permitted by policy. (verify: confirm completion in repo)
- [x] Modify the `normalize_frequency_code()` function to accept an optional configuration parameter for mapping 'Q' frequency to either 'M' or 'D'. Add unit tests for both scenarios.
- [x] Refactor the missingness propagation logic in `StationaryBootstrapModel` to avoid using internal state (`_log_returns`) for mask validation. Update tests to use public interfaces for missingness information.
- [x] Integrate performance test for generating 1000 paths over 120 periods into the standard test suite or update documentation for separate execution.
- [x] Define scope for: Integrate performance test for generating 1000 paths over 120 periods into the standard test suite. (verify: confirm completion in repo)
- [x] Implement focused slice for: Integrate performance test for generating 1000 paths over 120 periods into the standard test suite. (verify: confirm completion in repo)
- [x] Validate focused slice for: Integrate performance test for generating 1000 paths over 120 periods into the standard test suite. (verify: confirm completion in repo)
- [x] Define scope for: Update documentation with instructions for separate execution of the performance test. (verify: docs updated)
- [x] Implement focused slice for: Update documentation with instructions for separate execution of the performance test. (verify: docs updated)
- [x] Validate focused slice for: Update documentation with instructions for separate execution of the performance test. (verify: docs updated)

## Acceptance Criteria
- [x] The files `.agents/issue-4160-ledger.yml` and `.agents/issue-4592-ledger.yml` are removed from the repository and no new ledger files are added unless explicitly permitted by the repository policy.
- [x] The `normalize_frequency_code()` function accepts a configuration parameter that allows mapping 'Q' frequency to either 'M' or 'D', and unit tests confirm correct behavior for both mappings without runtime errors.
- [x] The `StationaryBootstrapModel` simulates missing data (NaN positions) during bootstrapping such that the resulting mask matches the historical missing data structure per asset, without using internal states like `_log_returns`.
- [x] The performance test for generating 1000 paths over 120 periods completes in less than 10 seconds and is part of the standard test suite or is documented with instructions for separate execution.

## Progress
21/21 tasks complete
