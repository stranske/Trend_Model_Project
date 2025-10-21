'use strict';

const fs = require('fs');

function trim(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPullRequestEvent(context) {
  return context?.eventName === 'pull_request';
}

function selectMarkerComment(comments, { marker, baseMessage }) {
  const normalizedMarker = marker || '';
  const normalizedBase = trim(baseMessage || '');
  let target = null;
  const duplicates = [];

  for (const comment of comments || []) {
    if (!comment || typeof comment.body !== 'string') {
      continue;
    }
    const body = comment.body;
    const trimmed = trim(body);
    const hasMarker = normalizedMarker && body.includes(normalizedMarker);
    const isLegacy = normalizedBase && (trimmed === normalizedBase || trimmed.startsWith(normalizedBase));
    if (!hasMarker && !isLegacy) {
      continue;
    }
    if (!target) {
      target = { comment, hasMarker };
      continue;
    }
    if (!target.hasMarker && hasMarker) {
      duplicates.push(target.comment);
      target = { comment, hasMarker };
    } else {
      duplicates.push(comment);
    }
  }

  return {
    target: target ? target.comment : null,
    targetHasMarker: Boolean(target?.hasMarker),
    duplicates,
  };
}

function info(core, message) {
  if (core && typeof core.info === 'function') {
    core.info(message);
  } else {
    console.log(message);
  }
}

function warn(core, message) {
  if (core && typeof core.warning === 'function') {
    core.warning(message);
  } else {
    console.warn(message);
  }
}

async function ensureMarkerComment({ github, context, core, commentBody, marker, baseMessage }) {
  if (!isPullRequestEvent(context)) {
    info(core, 'Not a pull_request event; skipping comment management.');
    return;
  }

  const body = trim(commentBody);
  if (!body) {
    const message = 'Docs-only comment body is missing.';
    if (core) {
      core.setFailed(message);
    }
    throw new Error(message);
  }

  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const issue_number = context.payload.pull_request.number;

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number,
    per_page: 100,
  });

  const { target, duplicates } = selectMarkerComment(comments, { marker, baseMessage });
  const desired = body;
  let targetId = target?.id;

  if (targetId) {
    const current = trim(target.body);
    if (current === desired) {
      info(core, `Existing docs-only comment ${targetId} is up to date.`);
    } else {
      await github.rest.issues.updateComment({ owner, repo, comment_id: targetId, body: desired });
      info(core, `Updated docs-only comment ${targetId}.`);
    }
  } else {
    const created = await github.rest.issues.createComment({ owner, repo, issue_number, body: desired });
    targetId = created?.data?.id;
    info(core, `Created docs-only comment ${targetId}.`);
  }

  for (const duplicate of duplicates) {
    if (!duplicate || duplicate.id === targetId) {
      continue;
    }
    await github.rest.issues.deleteComment({ owner, repo, comment_id: duplicate.id });
    info(core, `Removed duplicate docs-only comment ${duplicate.id}.`);
  }
}

async function removeMarkerComments({ github, context, core, marker, baseMessages = [] }) {
  if (!isPullRequestEvent(context)) {
    info(core, 'Not a pull_request event; nothing to clean up.');
    return;
  }

  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const issue_number = context.payload.pull_request.number;
  const legacyBodies = new Set(baseMessages.map(value => trim(value)).filter(Boolean));

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number,
    per_page: 100,
  });

  const targets = comments.filter(comment => {
    if (!comment || typeof comment.body !== 'string') {
      return false;
    }
    if (marker && comment.body.includes(marker)) {
      return true;
    }
    const trimmed = trim(comment.body);
    if (legacyBodies.has(trimmed)) {
      return true;
    }
    if (legacyBodies.size > 0) {
      for (const legacy of legacyBodies) {
        if (trimmed.startsWith(legacy)) {
          return true;
        }
      }
    }
    return false;
  });

  if (!targets.length) {
    info(core, 'No docs-only fast-pass comment found to remove.');
    return;
  }

  for (const comment of targets) {
    await github.rest.issues.deleteComment({ owner, repo, comment_id: comment.id });
    info(core, `Removed docs-only fast-pass comment ${comment.id}.`);
  }
}

function extractAnchoredMetadata(body, anchorPattern) {
  const pattern = anchorPattern instanceof RegExp
    ? anchorPattern
    : new RegExp(anchorPattern || '', 'i');
  const match = typeof body === 'string' ? body.match(pattern) : null;
  if (!match) {
    return null;
  }
  const content = match[1] || '';
  const prMatch = content.match(/pr=([0-9]+)/i);
  const headMatch = content.match(/head=([0-9a-f]+)/i);
  return {
    raw: match[0],
    pr: prMatch ? prMatch[1] : null,
    head: headMatch ? headMatch[1] : null,
  };
}

function findAnchoredComment(comments, { anchorPattern, fallbackMarker, targetAnchor }) {
  const marker = fallbackMarker || '';
  if (targetAnchor) {
    const anchored = comments.find(comment => {
      const info = extractAnchoredMetadata(comment?.body, anchorPattern);
      if (!info) {
        return false;
      }
      if (targetAnchor.pr && info.pr && info.pr !== targetAnchor.pr) {
        return false;
      }
      if (targetAnchor.head && info.head && info.head !== targetAnchor.head) {
        return false;
      }
      return true;
    });
    if (anchored) {
      return anchored;
    }
  }

  if (marker) {
    return comments.find(comment => typeof comment?.body === 'string' && comment.body.includes(marker)) || null;
  }

  return null;
}

async function upsertAnchoredComment({
  github,
  context,
  core,
  prNumber,
  commentPath,
  body,
  anchorPattern = /<!--\s*maint-46-post-ci:([^>]*)-->/i,
  fallbackMarker = '<!-- maint-46-post-ci:',
}) {
  const pr = Number(prNumber || 0);
  if (!Number.isFinite(pr) || pr <= 0) {
    warn(core, 'PR number missing; skipping comment update.');
    return;
  }

  let commentBody = body;
  if (!commentBody && commentPath) {
    commentBody = fs.readFileSync(commentPath, 'utf8');
  }
  commentBody = trim(commentBody);
  if (!commentBody) {
    warn(core, 'Comment body empty; skipping update.');
    return;
  }

  const owner = context.repo.owner;
  const repo = context.repo.repo;

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: pr,
    per_page: 100,
  });

  const targetAnchor = extractAnchoredMetadata(commentBody, anchorPattern);
  const existing = findAnchoredComment(comments, { anchorPattern, fallbackMarker, targetAnchor });

  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body: commentBody });
    info(core, 'Updated existing consolidated status comment.');
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number: pr, body: commentBody });
    info(core, 'Created consolidated status comment.');
  }
}

module.exports = {
  selectMarkerComment,
  ensureMarkerComment,
  removeMarkerComments,
  extractAnchoredMetadata,
  findAnchoredComment,
  upsertAnchoredComment,
};
