const test = require('node:test');
const assert = require('node:assert/strict');

const {
  listCompletedGateRuns,
  selectCoverageGateRun,
  sortCompletedGateRuns,
  successfulGateRuns,
} = require('../select_coverage_gate_run');

function createPaginatedListWorkflowRuns(pages) {
  let callCount = 0;
  return async () => {
    const page = pages[callCount] || { data: { workflow_runs: [] } };
    callCount += 1;
    return page;
  };
}

function createPaginateIterator(pages) {
  return {
    iterator(_method, _params) {
      let index = 0;
      return {
        [Symbol.asyncIterator]() {
          return {
            async next() {
              if (index >= pages.length) {
                return { value: undefined, done: true };
              }
              const value = pages[index];
              index += 1;
              return { value, done: false };
            },
          };
        },
      };
    },
  };
}

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
  assert.deepEqual(selected.requiredArtifacts, ['gate-coverage-trend']);
  assert.deepEqual(selected.skippedRunIds, [30]);
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
  assert.deepEqual(selected.skippedRunIds, [44]);
  assert.match(core.infos[0], /Skipping Gate run 44/);
});

test('selectCoverageGateRun paginates across pages when only an older run has coverage artifacts', async () => {
  const core = createCore();
  const pages = [
    {
      data: {
        workflow_runs: Array.from({ length: 3 }, (_, idx) => ({
          id: 200 + idx,
          status: 'completed',
          conclusion: 'success',
          created_at: `2026-04-${String(20 - idx).padStart(2, '0')}T00:00:00Z`,
        })),
      },
    },
    {
      data: {
        workflow_runs: [
          {
            id: 150,
            status: 'completed',
            conclusion: 'success',
            created_at: '2026-04-10T00:00:00Z',
          },
        ],
      },
    },
  ];
  const artifactByRun = new Map([
    [150, [{ name: 'gate-coverage-trend' }]],
  ]);
  const github = {
    paginate: createPaginateIterator(pages),
    rest: {
      actions: {
        listWorkflowRuns: createPaginatedListWorkflowRuns(pages),
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

  assert.equal(selected.run.id, 150);
  assert.deepEqual(selected.skippedRunIds, [200, 201, 202]);
});

test('selectCoverageGateRun honors maxPages cap when walking pages', async () => {
  const core = createCore();
  const pages = [
    {
      data: {
        workflow_runs: [
          {
            id: 301,
            status: 'completed',
            conclusion: 'success',
            created_at: '2026-04-05T00:00:00Z',
          },
        ],
      },
    },
    {
      data: {
        workflow_runs: [
          {
            id: 302,
            status: 'completed',
            conclusion: 'success',
            created_at: '2026-04-04T00:00:00Z',
          },
        ],
      },
    },
    {
      data: {
        workflow_runs: [
          {
            id: 303,
            status: 'completed',
            conclusion: 'success',
            created_at: '2026-04-03T00:00:00Z',
          },
        ],
      },
    },
  ];
  const github = {
    paginate: createPaginateIterator(pages),
    rest: {
      actions: {
        listWorkflowRuns: createPaginatedListWorkflowRuns(pages),
        listWorkflowRunArtifacts: async () => ({ data: { artifacts: [] } }),
      },
    },
  };

  const selected = await selectCoverageGateRun({
    github,
    context: { repo: { owner: 'octo', repo: 'demo' } },
    core,
    maxPages: 2,
  });

  assert.equal(selected, null);
  assert.deepEqual(core.warnings.at(-1).match(/required coverage artifact/) !== null, true);
});

test('listCompletedGateRuns falls back to a single page when paginate.iterator is absent', async () => {
  const runs = [
    {
      id: 1,
      status: 'completed',
      conclusion: 'success',
      created_at: '2026-04-01T00:00:00Z',
    },
  ];
  const github = {
    rest: {
      actions: {
        listWorkflowRuns: async () => ({ data: { workflow_runs: runs } }),
      },
    },
  };

  const result = await listCompletedGateRuns({
    github,
    owner: 'octo',
    repo: 'demo',
    workflowId: '.github/workflows/pr-00-gate.yml',
    perPage: 100,
    maxPages: 5,
  });

  assert.deepEqual(
    result.map((run) => run.id),
    [1],
  );
});
