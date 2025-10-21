'use strict';

const DEFAULT_PYTHON_VERSIONS = ['3.11'];

function parsePythonVersions(raw) {
  if (!raw || !raw.trim()) {
    return [...DEFAULT_PYTHON_VERSIONS];
  }
  try {
    const parsed = JSON.parse(raw.trim());
    if (Array.isArray(parsed) && parsed.length) {
      return parsed.map(String);
    }
  } catch (error) {
    // ignore parse errors and fall back
  }
  return [...DEFAULT_PYTHON_VERSIONS];
}

function parseScenarioList(raw) {
  if (!raw) {
    return [];
  }
  return raw
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);
}

function expectedArtifactsFor(scenario, pythonVersions, { warn } = {}) {
  const prefix = (suffix) => `sf-${scenario}-${suffix}`;
  const base = pythonVersions.map((version) => prefix(`coverage-${version}`));
  switch (scenario) {
    case 'minimal':
      return base;
    case 'metrics_only':
      return [...base, prefix('ci-metrics')];
    case 'metrics_history':
      return [...base, prefix('ci-metrics'), prefix('metrics-history')];
    case 'classification_only':
      return [...base, prefix('classification')];
    case 'coverage_delta':
      return [...base, prefix('coverage-delta')];
    case 'full_soft_gate':
      return [
        ...base,
        prefix('ci-metrics'),
        prefix('metrics-history'),
        prefix('classification'),
        prefix('coverage-delta'),
        prefix('coverage-summary'),
        prefix('coverage-trend'),
        prefix('coverage-trend-history'),
      ];
    default:
      if (typeof warn === 'function') {
        warn(`Unknown scenario '${scenario}' encountered; treating as minimal.`);
      }
      return base;
  }
}

function statusEmoji(conclusion) {
  switch ((conclusion || '').toLowerCase()) {
    case 'success':
      return '✅';
    case 'failure':
      return '❌';
    default:
      return 'ℹ️';
  }
}

function summarizeArtifacts({
  artifactNames,
  jobs,
  scenarioNames,
  pythonVersions,
  runId,
  warn,
}) {
  const rows = ['| Scenario | Status | Missing | Unexpected |', '|---|---|---|---|'];
  const summary = [];
  let failures = 0;

  const python = pythonVersions && pythonVersions.length ? pythonVersions : [...DEFAULT_PYTHON_VERSIONS];

  const jobStatusByScenario = Object.fromEntries(
    (jobs || [])
      .filter((job) => job && typeof job.name === 'string' && job.name.startsWith('Scenario - '))
      .map((job) => {
        const scenarioName = job.name.replace('Scenario - ', '').trim();
        return [scenarioName, statusEmoji(job.conclusion)];
      }),
  );

  const artifactList = artifactNames || [];
  const nameSet = new Set(artifactList);

  const expectedUniverse = new Set();

  scenarioNames.forEach((scenario) => {
    expectedArtifactsFor(scenario, python, { warn }).forEach((name) => expectedUniverse.add(name));
  });

  for (const scenario of scenarioNames) {
    const expected = expectedArtifactsFor(scenario, python, { warn });
    const missing = expected.filter((name) => !nameSet.has(name));
    const prefix = `sf-${scenario}-`;
    const actual = artifactList.filter((name) => name.startsWith(prefix));
    const unexpected = actual.filter((name) => !expected.includes(name));
    const status = jobStatusByScenario[scenario] || 'ℹ️';
    const ok = missing.length === 0 && unexpected.length === 0 && status === '✅';
    if (!ok) {
      failures += 1;
    }
    rows.push(
      `| ${scenario} | ${status} | ${missing.join('<br>') || '—'} | ${unexpected.join('<br>') || '—'} |`,
    );
    summary.push({ scenario, status, missing, unexpected, ok });
  }

  const stray = artifactList.filter((name) => name.startsWith('sf-') && !expectedUniverse.has(name));
  if (stray.length) {
    rows.push(`| (stray) | ❌ | (n/a) | ${stray.join('<br>')} |`);
    summary.push({ scenario: '_stray_', status: 'unexpected', missing: [], unexpected: stray, ok: false });
    failures += 1;
  }

  const table = rows.join('\n');
  const report = {
    run_id: runId,
    python_versions: python,
    scenarios: summary,
    artifact_count: artifactList.length,
    failures,
  };

  return { table, failures, summary, report };
}

async function normalizeCoverageArtifacts(
  { github, context, core },
  { pythonVersionsRaw, scenarioListRaw } = {},
) {
  const pythonVersions = parsePythonVersions(pythonVersionsRaw);
  const scenarioNames = parseScenarioList(scenarioListRaw);
  const { owner, repo } = context.repo;
  const runId = context.runId;

  const artifactNames = await github.paginate(
    github.rest.actions.listWorkflowRunArtifacts,
    { owner, repo, run_id: runId, per_page: 100 },
    (response) => response.data.map((artifact) => artifact.name),
  );

  const jobs = await github.paginate(
    github.rest.actions.listJobsForWorkflowRun,
    { owner, repo, run_id: runId, per_page: 100 },
    (response) => response.data,
  );

  return summarizeArtifacts({
    artifactNames,
    jobs,
    scenarioNames,
    pythonVersions,
    runId,
    warn: typeof core?.warning === 'function' ? core.warning.bind(core) : undefined,
  });
}

module.exports = {
  normalizeCoverageArtifacts,
  summarizeArtifacts,
  parsePythonVersions,
  parseScenarioList,
  expectedArtifactsFor,
};
