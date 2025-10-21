'use strict';

const DOC_EXTENSIONS = [
  '.md',
  '.mdx',
  '.markdown',
  '.rst',
  '.txt',
  '.qmd',
  '.adoc',
];

const DOC_BASENAMES = new Set([
  'readme',
  'changelog',
  'contributing',
  'code_of_conduct',
  'code-of-conduct',
  'security',
  'guidelines',
  'mkdocs',
  'docfx',
  'antora-playbook',
]);

const DOC_PREFIXES = [
  'docs/',
  'docs\\',
  'docs_',
  'doc/',
  'doc\\',
  'assets/docs/',
  'assets/docs\\',
  'documentation/',
  'documentation\\',
  'guides/',
  'handbook/',
  'manual/',
];

const DOC_SEGMENTS = [
  '/docs/',
  '/doc/',
  '/documentation/',
  '/manual/',
  '/design-docs/',
  '/handbook/',
  '/guide/',
  '/guides/',
  '/adr/',
  '/rfcs/',
  '/specs/',
  '/notes/',
  '\\docs\\',
  '\\doc\\',
  '\\documentation\\',
  '\\manual\\',
  '\\design-docs\\',
  '\\handbook\\',
  '\\guide\\',
  '\\guides\\',
  '\\adr\\',
  '\\rfcs\\',
  '\\specs\\',
  '\\notes\\',
];

const DOCKERFILE_SUFFIXES = ['/dockerfile', '\\dockerfile'];
const DOCKER_PREFIXES = ['docker/', 'docker\\', '.docker/', '.docker\\'];
const DOCKER_SEGMENTS = ['/docker/', '\\docker\\', '/.docker/', '\\.docker\\'];

function normaliseSeparators(value) {
  return value.replace(/\\/g, '/');
}

function getBasename(filename) {
  const normalised = normaliseSeparators(filename);
  const parts = normalised.split('/');
  return parts.length ? parts[parts.length - 1] : filename;
}

function classifyDocument(filename) {
  const lower = filename.toLowerCase();
  if (DOC_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return true;
  }

  const basename = getBasename(lower);
  if (basename) {
    const nameWithoutExt = basename.includes('.')
      ? basename.slice(0, basename.lastIndexOf('.'))
      : basename;
    if (DOC_BASENAMES.has(nameWithoutExt)) {
      return true;
    }
  }

  if (DOC_PREFIXES.some((prefix) => lower.startsWith(prefix))) {
    return true;
  }

  if (DOC_SEGMENTS.some((segment) => lower.includes(segment))) {
    return true;
  }

  return false;
}

function isDockerRelated(filename) {
  const lower = filename.toLowerCase();
  if (lower === 'dockerfile') {
    return true;
  }

  if (DOCKERFILE_SUFFIXES.some((suffix) => lower.endsWith(suffix))) {
    return true;
  }

  const basename = getBasename(lower);
  if (basename && basename.startsWith('dockerfile')) {
    return true;
  }

  if (lower === '.dockerignore') {
    return true;
  }

  if (DOCKER_PREFIXES.some((prefix) => lower.startsWith(prefix))) {
    return true;
  }

  if (DOCKER_SEGMENTS.some((segment) => lower.includes(segment))) {
    return true;
  }

  return false;
}

function analyseChangedFiles(filenames) {
  const changedFiles = filenames.map((name) => name.toLowerCase());
  const hasChanges = changedFiles.length > 0;
  const nonDocFiles = changedFiles.filter((filename) => !classifyDocument(filename));
  const docOnly = hasChanges ? nonDocFiles.length === 0 : true;
  const dockerChanged = changedFiles.some((filename) => isDockerRelated(filename));

  let reason = 'code_changes';
  if (!hasChanges) {
    reason = 'no_changes';
  } else if (docOnly) {
    reason = 'docs_only';
  }

  return {
    hasChanges,
    docOnly,
    dockerChanged,
    reason,
    runCore: !docOnly,
  };
}

async function detectChanges({ github, context, core }) {
  if (context.eventName !== 'pull_request') {
    core.setOutput('doc_only', 'false');
    core.setOutput('run_core', 'true');
    core.setOutput('reason', 'non_pr_event');
    core.setOutput('docker_changed', 'true');
    return;
  }

  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.payload.pull_request.number,
    per_page: 100,
  });

  const filenames = files.map((file) => file.filename || '').filter(Boolean);
  const result = analyseChangedFiles(filenames);

  core.setOutput('doc_only', result.docOnly ? 'true' : 'false');
  core.setOutput('run_core', result.runCore ? 'true' : 'false');
  core.setOutput('reason', result.reason);
  core.setOutput('docker_changed', result.dockerChanged ? 'true' : 'false');
}

module.exports = {
  detectChanges,
  analyseChangedFiles,
  classifyDocument,
  isDockerRelated,
};
