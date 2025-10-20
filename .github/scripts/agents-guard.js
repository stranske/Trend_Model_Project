// @ts-check

const DEFAULT_MARKER = '<!-- agents-guard-marker -->';

const DEFAULT_PROTECTED_PATHS = [
  '.github/workflows/agents-63-chatgpt-issue-sync.yml',
  '.github/workflows/agents-63-codex-issue-bridge.yml',
  '.github/workflows/agents-70-orchestrator.yml',
];

function escapeRegex(text) {
  return text.replace(/[.+^${}()|[\]\\]/g, '\\$&');
}

function globToRegExp(glob) {
  let result = '';
  let i = 0;
  while (i < glob.length) {
    const char = glob[i];
    if (char === '*') {
      const nextChar = glob[i + 1];
      if (nextChar === '*') {
        result += '.*';
        i += 2;
      } else {
        result += '[^/]*';
        i += 1;
      }
    } else if (char === '?') {
      result += '[^/]';
      i += 1;
    } else {
      result += escapeRegex(char);
      i += 1;
    }
  }
  return new RegExp(`^${result}$`);
}

function normalizePattern(pattern) {
  return pattern.replace(/^\/+/, '');
}

function parseCodeowners(content) {
  if (!content) {
    return [];
  }

  const entries = [];
  const lines = content.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) {
      continue;
    }

    const parts = line.split(/\s+/).filter(Boolean);
    if (parts.length < 2) {
      continue;
    }

    const pattern = parts[0];
    const normalized = normalizePattern(pattern);
    const owners = parts.slice(1);
    entries.push({
      pattern,
      owners,
      regex: globToRegExp(normalized),
    });
  }

  return entries;
}

function findCodeowners(entries, filePath) {
  const normalizedPath = filePath.replace(/^\/+/, '');
  let owners = [];
  for (const entry of entries) {
    if (entry.regex.test(normalizedPath)) {
      owners = entry.owners;
    }
  }
  return owners;
}

function listRelevantFiles(files) {
  return files.filter((file) => {
    if (!file || typeof file !== 'object') {
      return false;
    }

    const current = file.filename || '';
    const previous = file.previous_filename || '';

    if (current.startsWith('.github/workflows/agents-')) {
      return true;
    }
    if (previous && previous.startsWith('.github/workflows/agents-')) {
      return true;
    }
    return false;
  });
}

function summarizeTouchedFiles(files) {
  if (!files.length) {
    return '- (no files in scope detected)';
  }

  return files
    .map((file) => {
      const current = file.filename || '';
      const previous = file.previous_filename || '';
      const status = file.status || '';

      if (status === 'renamed' && previous) {
        return `- ${previous} → ${current} (${status})`;
      }

      return `- ${current} (${status})`;
    })
    .join('\n');
}

function collectLatestApprovals(reviews) {
  const latestReviewStates = new Map();
  for (const review of reviews || []) {
    if (!review || typeof review !== 'object') {
      continue;
    }

    const login = review.user && review.user.login
      ? String(review.user.login).toLowerCase()
      : '';
    if (!login) {
      continue;
    }

    const state = review.state ? String(review.state).toUpperCase() : '';
    if (!state) {
      continue;
    }

    latestReviewStates.set(login, state);
  }

  return new Set(
    [...latestReviewStates.entries()]
      .filter(([, state]) => state === 'APPROVED')
      .map(([login]) => login),
  );
}

function extractLabelNames(labels) {
  return new Set(
    (labels || [])
      .map((label) => (label && label.name ? String(label.name).toLowerCase() : ''))
      .filter(Boolean),
  );
}

function evaluateGuard({
  files = [],
  labels = [],
  reviews = [],
  codeownersContent = '',
  protectedPaths = DEFAULT_PROTECTED_PATHS,
  labelName = 'agents:allow-change',
  marker = DEFAULT_MARKER,
} = {}) {
  const normalizedLabelName = String(labelName).toLowerCase();
  const protectedSet = new Set(protectedPaths);

  const relevantFiles = listRelevantFiles(files);
  const fatalViolations = [];
  const modifiedProtectedPaths = new Set();
  const touchedProtectedPaths = new Set();

  for (const file of relevantFiles) {
    const current = file.filename || '';
    const previous = file.previous_filename || '';
    const status = file.status || '';

    const protectedPath = protectedSet.has(current)
      ? current
      : (previous && protectedSet.has(previous) ? previous : null);

    if (protectedPath) {
      touchedProtectedPaths.add(protectedPath);
      if (status === 'removed') {
        fatalViolations.push(`• ${current} was deleted.`);
        continue;
      }

      if (status === 'renamed' && previous) {
        fatalViolations.push(`• ${previous} was renamed to ${current}.`);
        continue;
      }

      if (status === 'modified') {
        modifiedProtectedPaths.add(protectedPath);
      }
    }
  }

  const labelNames = extractLabelNames(labels);
  const hasAllowLabel = labelNames.has(normalizedLabelName);

  const approvedLogins = collectLatestApprovals(reviews);

  const codeownerEntries = parseCodeowners(codeownersContent);
  const codeownerLogins = new Set();
  const relevantCodeownerPaths = touchedProtectedPaths.size > 0
    ? [...touchedProtectedPaths]
    : [...protectedSet];

  for (const path of relevantCodeownerPaths) {
    const owners = findCodeowners(codeownerEntries, path);
    for (const ownerSlug of owners) {
      if (!ownerSlug || !ownerSlug.startsWith('@')) {
        continue;
      }
      const name = ownerSlug.slice(1).trim();
      if (!name || name.includes('/')) {
        // Team owners cannot be expanded without additional permissions.
        continue;
      }
      codeownerLogins.add(name.toLowerCase());
    }
  }

  const hasCodeownerApproval = [...codeownerLogins].some((login) => approvedLogins.has(login));

  const needsLabel = modifiedProtectedPaths.size > 0 && !hasAllowLabel;
  const needsApproval = modifiedProtectedPaths.size > 0 && !hasCodeownerApproval;

  const failureReasons = [];
  if (fatalViolations.length > 0) {
    failureReasons.push(...fatalViolations);
  }

  if (modifiedProtectedPaths.size > 0 && (needsLabel || needsApproval)) {
    const modifiedList = [...modifiedProtectedPaths].map((path) => `• ${path}`).join('\n');
    failureReasons.push(`Protected workflows modified:\n${modifiedList}`);
    if (needsLabel) {
      failureReasons.push('Missing `agents:allow-change` label.');
    }
    if (needsApproval) {
      const codeownerHint = codeownerLogins.size > 0
        ? `Request approval from a CODEOWNER (${[...codeownerLogins].map((login) => `@${login}`).join(', ')}).`
        : 'Request approval from a CODEOWNER.';
      failureReasons.push(codeownerHint);
    }
  }

  const blocked = failureReasons.length > 0;
  const plainFirstReason = blocked
    ? failureReasons[0].replace(/^[\s•*-]+/, '').trim()
    : '';
  const summary = blocked
    ? (plainFirstReason
      ? `Health 45 Agents Guard blocked this PR: ${plainFirstReason}`
      : 'Health 45 Agents Guard blocked this PR.')
    : 'Health 45 Agents Guard passed.';

  let commentBody = null;
  let instructions = [];
  const touchedFilesText = summarizeTouchedFiles(relevantFiles);
  if (blocked) {
    instructions = [];
    if (fatalViolations.length > 0) {
      instructions.push('Restore the deleted or renamed workflows. These files cannot be moved or removed.');
    }
    if (needsLabel) {
      instructions.push('Apply the `agents:allow-change` label to this pull request once the change is justified.');
    }
    if (needsApproval) {
      if (codeownerLogins.size > 0) {
        const ownersList = [...codeownerLogins].map((login) => `@${login}`).join(', ');
        instructions.push(`Ask a CODEOWNER (${ownersList}) to review and approve the change.`);
      } else {
        instructions.push('Ask a CODEOWNER to review and approve the change.');
      }
    }
    instructions.push('Push an update or re-run this workflow after addressing the issues.');

    commentBody = [
      marker,
  '**Health 45 Agents Guard** stopped this pull request.',
      '',
      '**What we found**',
      ...failureReasons.map((reason) => `- ${reason}`),
      '',
      '**Next steps**',
      ...instructions.map((step) => `- ${step}`),
      '',
      '**Files seen in this run**',
      touchedFilesText,
    ].join('\n');
  }

  const warnings = [];
  if (blocked && fatalViolations.length === 0 && modifiedProtectedPaths.size === 0) {
    warnings.push('Guard triggered but no protected file changes were found.');
  }

  return {
    blocked,
    summary,
    marker,
    failureReasons,
    instructions,
    commentBody,
    touchedFilesText,
    warnings,
    hasAllowLabel,
    hasCodeownerApproval,
    needsLabel,
    needsApproval,
    modifiedProtectedPaths: [...modifiedProtectedPaths],
    touchedProtectedPaths: [...touchedProtectedPaths],
    fatalViolations,
    codeownerLogins: [...codeownerLogins],
    relevantFiles,
  };
}

module.exports = {
  DEFAULT_MARKER,
  DEFAULT_PROTECTED_PATHS,
  evaluateGuard,
  parseCodeowners,
  globToRegExp,
};

