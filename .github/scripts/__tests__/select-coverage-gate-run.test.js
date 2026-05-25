const test = require('node:test');
const assert = require('node:assert/strict');

const {
  selectCoverageGateRun,
  sortCompletedGateRuns,
  successfulGateRuns,
} = require('../select_coverage_gate_run');

function createCore() {
  return {
    infos: [],
    warnings: [],
    info(message) {
      this.infos.push(message);
    },
    warning(message) {
      this.warnings.push(message);
    },
  };
}

test('sortCompletedGateRuns orders newest completed runs first', () => {
  const runs = [
    { id: 1, status: 'completed', created_at: '2026-01-01T00:00:00Z' },
    { id: 2, status: 'in_progress', created_at: '2026-01-03T00:00:00Z' },
    { id: 3, status: 'completed', created_at: '2026-01-02T00:00:00Z' },
  ];

  assert.deepEqual(
    sortCompletedGateRuns(runs).map((run) => run.id),
    [3, 1],
  );
});

test('successfulGateRuns keeps success and neutral only', () => {
  const runs = [
    { id: 1, status: 'completed', conclusion: 'failure', created_at: '2026-01-03T00:00:00Z' },
    { id: 2, status: 'completed', conclusion: 'success', created_at: '2026-01-02T00:00:00Z' },
    { id: 3, status: 'completed', conclusion: 'neutral', created_at: '2026-01-01T00:00:00Z' },
  ];

  assert.deepEqual(
    successfulGateRuns(runs).map((run) => run.id),
    [2, 3],
  );
});

test('selectCoverageGateRun skips newer Gate runs without coverage artifacts', async () => {
  const core = createCore();
  const artifactByRun = new Map([
    [30, []],
    [20, [{ name: 'gate-coverage-trend-history' }, { name: 'gate-coverage-trend' }]],
  ]);
  const github = {
    rest: {
      actions: {
        listWorkflowRuns: async () => ({
          data: {
            workflow_runs: [
              {
                id: 30,
                status: 'completed',
                conclusion: 'success',
                created_at: '2026-02-03T00:00:00Z',
              },
              {
                id: 20,
                status: 'completed',
                conclusion: 'success',
                created_at: '2026-02-02T00:00:00Z',
              },
            ],
          },
        }),
        listWorkflowRunArtifacts: async ({ run_id }) => ({
          data: { artifacts: artifactByRun.get(run_id) || [] },
        }),
      },
    },
  };

  const selected = await selectCoverageGateRun({
    github,
    context: { repo: { owner: 'octo', repo: 'demo' } },
    core,
  });

  assert.equal(selected.run.id, 20);
  assert.deepEqual(selected.artifacts, ['gate-coverage-trend', 'gate-coverage-trend-history']);
  assert.match(core.infos[0], /Skipping Gate run 30/);
});

test('selectCoverageGateRun returns null when no run has required coverage artifacts', async () => {
  const core = createCore();
  const github = {
    rest: {
      actions: {
        listWorkflowRuns: async () => ({
          data: {
            workflow_runs: [
              {
                id: 10,
                status: 'completed',
                conclusion: 'success',
                created_at: '2026-02-01T00:00:00Z',
              },
            ],
          },
        }),
        listWorkflowRunArtifacts: async () => ({ data: { artifacts: [] } }),
      },
    },
  };

  const selected = await selectCoverageGateRun({
    github,
    context: { repo: { owner: 'octo', repo: 'demo' } },
    core,
  });

  assert.equal(selected, null);
  assert.match(core.warnings.at(-1), /No recent Gate workflow run/);
});

test('selectCoverageGateRun ignores expired coverage artifacts', async () => {
  const core = createCore();
  const artifactByRun = new Map([
    [44, [{ name: 'gate-coverage-trend', expired: true }]],
    [43, [{ name: 'gate-coverage-trend', expired: false }]],
  ]);
  const github = {
    rest: {
      actions: {
        listWorkflowRuns: async () => ({
          data: {
            workflow_runs: [
              {
                id: 44,
                status: 'completed',
                conclusion: 'success',
                created_at: '2026-03-02T00:00:00Z',
              },
              {
                id: 43,
                status: 'completed',
                conclusion: 'success',
                created_at: '2026-03-01T00:00:00Z',
              },
            ],
          },
        }),
        listWorkflowRunArtifacts: async ({ run_id }) => ({
          data: { artifacts: artifactByRun.get(run_id) || [] },
        }),
      },
    },
  };

  const selected = await selectCoverageGateRun({
    github,
    context: { repo: { owner: 'octo', repo: 'demo' } },
    core,
  });

  assert.equal(selected.run.id, 43);
  assert.match(core.infos[0], /Skipping Gate run 44/);
});
