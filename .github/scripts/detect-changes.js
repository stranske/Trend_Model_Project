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

const DOCKER_SUFFIXES = ['/dockerfile', '\\dockerfile'];
const DOCKER_PREFIXES = ['docker/', 'docker\\', '.docker/', '.docker\\'];
const DOCKER_SEGMENTS = ['/docker/', '\\docker\\', '/.docker/', '\\.docker\\'];

function normalisePath(value) {
  return String(value || '').toLowerCase();
}

function toForwardSlashes(value) {
  return value.replace(/\\/g, '/');
}

function basename(value) {
  const normalised = toForwardSlashes(value);
  const parts = normalised.split('/');
  return parts[parts.length - 1] || '';
}

function stripExtension(name) {
  if (!name.includes('.')) {
    return name;
  }
  return name.slice(0, name.lastIndexOf('.'));
}

function isDocFile(filename) {
  const normalised = normalisePath(filename);

  if (DOC_EXTENSIONS.some((ext) => normalised.endsWith(ext))) {
    return true;
  }

  const base = basename(normalised);
  if (base) {
    const withoutExt = stripExtension(base);
    if (DOC_BASENAMES.has(withoutExt)) {
      return true;
    }
  }

  if (DOC_PREFIXES.some((prefix) => normalised.startsWith(prefix))) {
    return true;
  }

  if (DOC_SEGMENTS.some((segment) => normalised.includes(segment))) {
    return true;
  }

  return false;
}

function isDockerRelated(filename) {
  const normalised = normalisePath(filename);
  if (normalised === 'dockerfile') {
    return true;
  }

  if (DOCKER_SUFFIXES.some((suffix) => normalised.endsWith(suffix))) {
    return true;
  }

  const base = basename(normalised);
  if (base && base.startsWith('dockerfile')) {
    return true;
  }

  if (normalised === '.dockerignore') {
    return true;
  }

  if (DOCKER_PREFIXES.some((prefix) => normalised.startsWith(prefix))) {
    return true;
  }

  if (DOCKER_SEGMENTS.some((segment) => normalised.includes(segment))) {
    return true;
  }

  return false;
}

async function detectChanges({ github, context, core }) {
  const eventName = context.eventName;
  if (eventName !== 'pull_request') {
    core.setOutput('doc_only', 'false');
    core.setOutput('run_core', 'true');
    core.setOutput('reason', 'non_pr_event');
    core.setOutput('docker_changed', 'true');
    return {
      docOnly: false,
      runCore: true,
      reason: 'non_pr_event',
      dockerChanged: true,
    };
  }

  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.payload.pull_request.number,
    per_page: 100,
  });

  const changedFiles = files.map((file) => file.filename);
  const hasChanges = changedFiles.length > 0;
  const nonDocFiles = changedFiles.filter((filename) => !isDocFile(filename));
  const docOnly = hasChanges ? nonDocFiles.length === 0 : true;
  const dockerChanged = changedFiles.some((filename) => isDockerRelated(filename));

  let reason = 'code_changes';
  if (!hasChanges) {
    reason = 'no_changes';
  } else if (docOnly) {
    reason = 'docs_only';
  }

  core.setOutput('doc_only', docOnly ? 'true' : 'false');
  core.setOutput('run_core', docOnly ? 'false' : 'true');
  core.setOutput('reason', reason);
  core.setOutput('docker_changed', dockerChanged ? 'true' : 'false');

  return {
    docOnly,
    runCore: !docOnly,
    reason,
    dockerChanged,
  };
}

module.exports = {
  detectChanges,
  isDocFile,
  isDockerRelated,
};
