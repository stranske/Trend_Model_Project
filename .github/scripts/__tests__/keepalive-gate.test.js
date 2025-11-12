'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  countActiveRuns,
  RECENT_RUN_LOOKBACK_MINUTES,
} = require('../keepalive_gate.js');

function makeGithubStub(registry) {
  return {
    rest: {
      actions: {
        listWorkflowRuns: Symbol('listWorkflowRuns'),
      },
    },
    async paginate(_fn, params, mapFn) {
      const key = `${params.workflow_id}|${params.status}`;
      const payload = registry[key] || [];
      if (typeof mapFn === 'function') {
        let stopped = false;
        const done = () => {
          stopped = true;
        };
        mapFn({ data: payload }, done);
        if (stopped) {
          return [];
        }
        return [];
      }
      return payload;
    },
  };
}

test('countActiveRuns tallies in-flight runs without duplication', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      { id: 101, pull_requests: [{ number: 42 }] },
      { id: 102, pull_requests: [{ number: 999 }] }, // mismatched PR should be ignored
    ],
    'agents-70-orchestrator.yml|in_progress': [
      { id: 103, pull_requests: [{ number: 42 }] },
      { id: 101, pull_requests: [{ number: 42 }] }, // duplicate id should not double count
    ],
    'agents-70-orchestrator.yml|completed': [],
  };
  const github = makeGithubStub(registry);
  const result = await countActiveRuns({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 42,
    headSha: 'abc',
    headRef: 'feature/run-cap',
    currentRunId: 9999,
    workflowFile: 'agents-70-orchestrator.yml',
    recentWindowMinutes: RECENT_RUN_LOOKBACK_MINUTES,
  });

  assert.equal(result.activeRuns, 2); // ids 101 and 103
  assert.equal(result.inFlightRuns, 2);
  assert.equal(result.recentRuns, 0);
});

test('countActiveRuns treats recent completed runs as active within the window', async () => {
  const now = Date.now();
  const recentIso = new Date(now - 2 * 60 * 1000).toISOString();
  const oldIso = new Date(now - 15 * 60 * 1000).toISOString();

  const registry = {
    'agents-70-orchestrator.yml|queued': [],
    'agents-70-orchestrator.yml|in_progress': [],
    'agents-70-orchestrator.yml|completed': [
      { id: 201, pull_requests: [{ number: 7 }], updated_at: recentIso },
      { id: 202, pull_requests: [{ number: 7 }], updated_at: oldIso },
    ],
  };
  const github = makeGithubStub(registry);
  const result = await countActiveRuns({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 7,
    headSha: 'xyz',
    headRef: 'feature/recent-run',
    currentRunId: 0,
    workflowFile: 'agents-70-orchestrator.yml',
    recentWindowMinutes: 5,
  });

  assert.equal(result.inFlightRuns, 0);
  assert.equal(result.recentRuns, 1);
  assert.equal(result.activeRuns, 1);
});

test('countActiveRuns ignores the current run id to avoid self-counting', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      { id: 555, pull_requests: [{ number: 5 }] },
    ],
    'agents-70-orchestrator.yml|in_progress': [],
    'agents-70-orchestrator.yml|completed': [],
  };
  const github = makeGithubStub(registry);
  const result = await countActiveRuns({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 5,
    headSha: 'sha',
    headRef: 'refs/heads/branch',
    currentRunId: 555,
    workflowFile: 'agents-70-orchestrator.yml',
    recentWindowMinutes: RECENT_RUN_LOOKBACK_MINUTES,
  });

  assert.equal(result.activeRuns, 0);
  assert.equal(result.inFlightRuns, 0);
  assert.equal(result.recentRuns, 0);
});

test('countActiveRuns recognises orchestrator concurrency tags for keepalive runs', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      {
        id: 610,
        concurrency: 'agents-70-orchestrator-42-keepalive-trace1234',
        head_branch: 'phase-2-dev',
        pull_requests: [],
      },
    ],
    'agents-70-orchestrator.yml|in_progress': [],
    'agents-70-orchestrator.yml|completed': [],
  };
  const github = makeGithubStub(registry);
  const result = await countActiveRuns({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 42,
    headSha: 'different-sha',
    headRef: 'codex/issue-42',
    currentRunId: 0,
    workflowFile: 'agents-70-orchestrator.yml',
    recentWindowMinutes: RECENT_RUN_LOOKBACK_MINUTES,
  });

  assert.equal(result.activeRuns, 1);
  assert.equal(result.inFlightRuns, 1);
  assert.equal(result.recentRuns, 0);
});

test('countActiveRuns detects PR markers in run display titles', async () => {
  const registry = {
    'agents-70-orchestrator.yml|queued': [
      {
        id: 702,
        display_title: 'Agents 70 Orchestrator (#99) keepalive sweep',
        name: 'agents-70-orchestrator',
        head_branch: 'phase-2-dev',
        pull_requests: [],
      },
    ],
    'agents-70-orchestrator.yml|in_progress': [],
    'agents-70-orchestrator.yml|completed': [],
  };
  const github = makeGithubStub(registry);
  const result = await countActiveRuns({
    github,
    owner: 'stranske',
    repo: 'Trend_Model_Project',
    prNumber: 99,
    headSha: 'sha',
    headRef: 'feature/other-branch',
    currentRunId: 0,
    workflowFile: 'agents-70-orchestrator.yml',
    recentWindowMinutes: RECENT_RUN_LOOKBACK_MINUTES,
  });

  assert.equal(result.activeRuns, 1);
  assert.equal(result.inFlightRuns, 1);
  assert.equal(result.recentRuns, 0);
});
