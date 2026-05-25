'use strict';

const DEFAULT_ARTIFACT_NAMES = ['gate-coverage-trend'];
const DEFAULT_WORKFLOW_ID = '.github/workflows/pr-00-gate.yml';
const DEFAULT_PER_PAGE = 100;
const DEFAULT_MAX_PAGES = 5;

function runStartedAt(run) {
  return new Date(run?.run_started_at || run?.created_at || 0).getTime();
}

function sortCompletedGateRuns(runs) {
  return [...(runs || [])]
    .filter((run) => run && run.status === 'completed')
    .sort((a, b) => runStartedAt(b) - runStartedAt(a));
}

function successfulGateRuns(runs) {
  const allowed = new Set(['success', 'neutral']);
  return sortCompletedGateRuns(runs).filter((run) => allowed.has(run.conclusion || ''));
}

function artifactNames(artifacts) {
  return new Set(
    (artifacts || [])
      .filter((artifact) => artifact && artifact.expired !== true)
      .map((artifact) => String(artifact?.name || '').trim())
      .filter(Boolean),
  );
}

async function listArtifactsForRun({ github, owner, repo, runId }) {
  const params = {
    owner,
    repo,
    run_id: runId,
    per_page: 100,
  };
  const listArtifacts = github.rest.actions.listWorkflowRunArtifacts;
  if (typeof github.paginate === 'function') {
    return github.paginate(listArtifacts, params);
  }
  const response = await listArtifacts(params);
  return response?.data?.artifacts || [];
}

async function listCompletedGateRuns({
  github,
  owner,
  repo,
  workflowId,
  perPage,
  maxPages,
}) {
  const params = {
    owner,
    repo,
    workflow_id: workflowId,
    status: 'completed',
    per_page: perPage,
  };
  const listRuns = github.rest.actions.listWorkflowRuns;
  const iterator = github.paginate?.iterator;
  if (typeof iterator === 'function') {
    const collected = [];
    let pagesFetched = 0;
    for await (const response of iterator.call(github.paginate, listRuns, params)) {
      const pageRuns = response?.data?.workflow_runs || [];
      collected.push(...pageRuns);
      pagesFetched += 1;
      if (maxPages && pagesFetched >= maxPages) {
        break;
      }
    }
    return collected;
  }
  const response = await listRuns(params);
  return response?.data?.workflow_runs || [];
}

async function selectCoverageGateRun({
  github,
  context,
  core,
  workflowId = DEFAULT_WORKFLOW_ID,
  requiredArtifacts = DEFAULT_ARTIFACT_NAMES,
  perPage = DEFAULT_PER_PAGE,
  maxPages = DEFAULT_MAX_PAGES,
} = {}) {
  const { owner, repo } = context.repo;
  const runs = await listCompletedGateRuns({
    github,
    owner,
    repo,
    workflowId,
    perPage,
    maxPages,
  });
  const candidates = successfulGateRuns(runs);

  if (!candidates.length) {
    core?.warning?.('No successful or neutral Gate workflow runs found.');
    return null;
  }

  const skippedRunIds = [];

  for (const run of candidates) {
    let artifacts = [];
    try {
      artifacts = await listArtifactsForRun({ github, owner, repo, runId: run.id });
    } catch (error) {
      core?.warning?.(`Unable to inspect artifacts for Gate run ${run.id}: ${error.message}`);
      continue;
    }
    const names = artifactNames(artifacts);
    const missing = requiredArtifacts.filter((name) => !names.has(name));
    if (!missing.length) {
      return {
        run,
        artifacts: [...names].sort(),
        requiredArtifacts: [...requiredArtifacts],
        skippedRunIds,
      };
    }
    skippedRunIds.push(run.id);
    core?.info?.(
      `Skipping Gate run ${run.id}: missing coverage artifact(s): ${missing.join(', ')}`,
    );
  }

  core?.warning?.(
    `No recent Gate workflow run published required coverage artifact(s): ${requiredArtifacts.join(', ')}`,
  );
  return null;
}

module.exports = {
  artifactNames,
  listArtifactsForRun,
  listCompletedGateRuns,
  selectCoverageGateRun,
  sortCompletedGateRuns,
  successfulGateRuns,
};
