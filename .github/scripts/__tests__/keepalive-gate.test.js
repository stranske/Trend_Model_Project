'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { countActive } = require('../keepalive_gate.js');

// The `details` parameter provides mock run details for getWorkflowRun.
// By default, it is an empty object, so getWorkflowRun will throw a 404 error
// for any runId not explicitly provided in `details`. This is intentional and
// expected behavior for test isolation, simulating the real GitHub API's response.
function makeGithubStub(registry, details = {}) {
  return {
    rest: {
      actions: {
        listWorkflowRuns: Symbol('listWorkflowRuns'),
        async getWorkflowRun({ run_id: runId }) {
          if (Object.prototype.hasOwnProperty.call(details, runId)) {
            return { data: details[runId] };
          }
          const error = new Error('not found');
          error.status = 404;
          throw error;
        },
      },
    },
    async paginate(_fn, params) {
      const key = `${params.workflow_id}|${params.status}`;
      const payload = registry[key] || [];
      return payload;
    },
  };
}

test('countActive counts queued and in-progress orchestrator runs without duplication', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      { id: 101, pull_requests: [{ number: 42 }] },
      { id: 102, pull_requests: [{ number: 999 }] }, // mismatched PR should be ignored
    ],
    'agents-70-orchestrator.yml|in_progress': [
      { id: 103, pull_requests: [{ number: 42 }] },
      { id: 101, pull_requests: [{ number: 42 }] }, // duplicate id should not double count
    ],
  };
  const github = makeGithubStub(registry);
  const result = await countActive({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 42,
    headSha: 'abc',
    headRef: 'feature/run-cap',
    currentRunId: 9999,
  });

  assert.equal(result.active, 2); // ids 101 and 103
  assert.equal(result.breakdown.get('orchestrator'), 2);
  assert.equal(result.breakdown.get('worker'), undefined);
});

test('countActive optionally includes worker runs when requested', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      { id: 201, pull_requests: [{ number: 7 }] },
    ],
    'agents-72-codex-belt-worker.yml|in_progress': [
      { id: 301, pull_requests: [{ number: 7 }] },
    ],
  };
  const github = makeGithubStub(registry);
  const withoutWorker = await countActive({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 7,
    includeWorker: false,
  });

  assert.equal(withoutWorker.active, 1);
  assert.equal(withoutWorker.breakdown.get('orchestrator'), 1);
  assert.equal(withoutWorker.breakdown.get('worker'), undefined);

  const withWorker = await countActive({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 7,
  });

  assert.equal(withWorker.active, 2);
  assert.equal(withWorker.breakdown.get('orchestrator'), 1);
  assert.equal(withWorker.breakdown.get('worker'), 1);
});

test('countActive ignores the current run id to avoid self-counting', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      { id: 555, pull_requests: [{ number: 5 }] },
    ],
  };
  const github = makeGithubStub(registry);
  const result = await countActive({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 5,
    headSha: 'sha',
    headRef: 'refs/heads/branch',
    currentRunId: 555,
  });

  assert.equal(result.active, 0);
  assert.equal(result.breakdown.size, 0);
});

test('countActive matches by branch metadata when pull requests array is empty', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      { id: 610, head_branch: 'refs/heads/feature/match-me' },
    ],
  };
  const details = {
    610: {
      id: 610,
      head_branch: 'feature/match-me',
      head_sha: 'abc123',
    },
  };
  const github = makeGithubStub(registry, details);
  const result = await countActive({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 8,
    headRef: 'feature/match-me',
    headSha: 'abc123',
  });

  assert.equal(result.active, 1);
  assert.equal(result.breakdown.get('orchestrator'), 1);
});
