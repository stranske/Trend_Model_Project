'use strict';

function extractAnchor(text) {
  if (typeof text !== 'string' || !text) {
    return null;
  }
  const anchorPattern = /<!--\s*maint-46-post-ci:([^>]*)-->/i;
  const match = text.match(anchorPattern);
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

function findExistingComment(comments, targetAnchor, markerPrefix) {
  if (targetAnchor) {
    const candidate = comments.find((comment) => {
      if (!comment || typeof comment.body !== 'string') {
        return false;
      }
      const info = extractAnchor(comment.body);
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
    if (candidate) {
      return candidate;
    }
  }

  return comments.find((comment) => {
    if (!comment || typeof comment.body !== 'string') {
      return false;
    }
    return comment.body.includes(markerPrefix);
  }) || null;
}

async function upsertSummaryComment({
  github,
  context,
  core,
  issueNumber,
  body,
  markerPrefix = '<!-- maint-46-post-ci:',
}) {
  const pr = Number(issueNumber);
  if (!Number.isFinite(pr) || pr <= 0) {
    core.warning('PR number missing; skipping comment update.');
    return { updated: false, created: false };
  }
  if (typeof body !== 'string' || !body.trim()) {
    core.warning('Comment body empty; skipping comment update.');
    return { updated: false, created: false };
  }

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: pr,
    per_page: 100,
  });

  const targetAnchor = extractAnchor(body);
  const existing = findExistingComment(comments, targetAnchor, markerPrefix);

  if (existing && existing.id) {
    await github.rest.issues.updateComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      comment_id: existing.id,
      body,
    });
    core.info('Updated existing consolidated status comment.');
    return { updated: true, created: false, id: existing.id };
  }

  const created = await github.rest.issues.createComment({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: pr,
    body,
  });
  const createdId = created?.data?.id;
  core.info('Created consolidated status comment.');
  return { updated: false, created: true, id: createdId };
}

module.exports = {
  extractAnchor,
  findExistingComment,
  upsertSummaryComment,
};
