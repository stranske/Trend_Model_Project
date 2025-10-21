'use strict';

function extractAnchor(text, pattern) {
  if (!text) {
    return null;
  }
  const match = text.match(pattern);
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

async function upsertSummaryComment(
  { github, context, core },
  {
    body,
    prNumber,
    anchorPattern = /<!--\s*maint-46-post-ci:([^>]*)-->/i,
    fallbackMarker = '<!-- maint-46-post-ci:',
  } = {},
) {
  if (!prNumber) {
    core.warning('PR number missing; skipping comment update.');
    return { action: 'skip', reason: 'missing-pr' };
  }

  if (!body || !body.trim()) {
    core.setFailed('Summary comment body is empty.');
    return { action: 'error', reason: 'empty-body' };
  }

  const targetAnchor = extractAnchor(body, anchorPattern);
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: prNumber,
    per_page: 100,
  });

  let existing = null;
  if (targetAnchor) {
    existing = comments.find((comment) => {
      const info = extractAnchor(comment.body || '', anchorPattern);
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
  }

  if (!existing) {
    existing = comments.find((comment) => (comment.body || '').includes(fallbackMarker));
  }

  if (existing) {
    const trimmedExisting = typeof existing.body === 'string' ? existing.body.trim() : '';
    const trimmedDesired = body.trim();
    if (trimmedExisting === trimmedDesired) {
      core.info(`Existing consolidated status comment ${existing.id} is up to date.`);
      return { action: 'noop', commentId: existing.id };
    }

    await github.rest.issues.updateComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      comment_id: existing.id,
      body,
    });
    core.info(`Updated existing consolidated status comment ${existing.id}.`);
    return { action: 'updated', commentId: existing.id };
  }

  const created = await github.rest.issues.createComment({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: prNumber,
    body,
  });
  core.info(`Created consolidated status comment ${created.data.id}.`);
  return { action: 'created', commentId: created.data.id };
}

module.exports = {
  upsertSummaryComment,
  extractAnchor,
};
