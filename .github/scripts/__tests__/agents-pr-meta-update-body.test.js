'use strict';

const {
  parseCheckboxStates,
  mergeCheckboxStates,
  ensureChecklist,
  extractBlock,
} = require('../agents_pr_meta_update_body.js');

describe('parseCheckboxStates', () => {
  it('extracts checked items from a checkbox list', () => {
    const block = `
- [x] Task one completed
- [ ] Task two pending
- [x] Task three completed
- [ ] Task four pending
    `.trim();

    const states = parseCheckboxStates(block);

    expect(states.size).toBe(2);
    expect(states.get('task one completed')).toBe(true);
    expect(states.get('task three completed')).toBe(true);
    expect(states.has('task two pending')).toBe(false);
  });

  it('normalizes text by stripping leading dashes', () => {
    const block = `
- [x] - Tests fail if weight bounds...
- [ ] - Existing functionality remains
    `.trim();

    const states = parseCheckboxStates(block);

    expect(states.size).toBe(1);
    expect(states.get('tests fail if weight bounds...')).toBe(true);
  });

  it('handles case-insensitive matching', () => {
    const block = `
- [X] UPPERCASE checked
- [x] lowercase checked
    `.trim();

    const states = parseCheckboxStates(block);

    expect(states.size).toBe(2);
    expect(states.get('uppercase checked')).toBe(true);
    expect(states.get('lowercase checked')).toBe(true);
  });

  it('returns empty map for empty input', () => {
    expect(parseCheckboxStates('')).toEqual(new Map());
    expect(parseCheckboxStates(null)).toEqual(new Map());
    expect(parseCheckboxStates(undefined)).toEqual(new Map());
  });
});

describe('mergeCheckboxStates', () => {
  it('restores checked state for unchecked items', () => {
    const newContent = `
- [ ] Task one
- [ ] Task two
- [ ] Task three
    `.trim();

    const existingStates = new Map([
      ['task one', true],
      ['task three', true],
    ]);

    const result = mergeCheckboxStates(newContent, existingStates);

    expect(result).toContain('- [x] Task one');
    expect(result).toContain('- [ ] Task two');
    expect(result).toContain('- [x] Task three');
  });

  it('preserves already checked items in new content', () => {
    const newContent = `
- [x] Already checked in new content
- [ ] Unchecked in new
    `.trim();

    const existingStates = new Map([
      ['unchecked in new', true],
    ]);

    const result = mergeCheckboxStates(newContent, existingStates);

    // Already checked stays checked
    expect(result).toContain('- [x] Already checked in new content');
    // Unchecked gets restored
    expect(result).toContain('- [x] Unchecked in new');
  });

  it('handles items with leading dashes in text', () => {
    const newContent = `
- [ ] - Tests fail if bounds violated
- [ ] - Functionality remains unchanged
    `.trim();

    const existingStates = new Map([
      ['tests fail if bounds violated', true],
    ]);

    const result = mergeCheckboxStates(newContent, existingStates);

    expect(result).toContain('- [x] - Tests fail if bounds violated');
    expect(result).toContain('- [ ] - Functionality remains unchanged');
  });

  it('returns original content if no existing states', () => {
    const content = '- [ ] Task one\n- [ ] Task two';
    
    expect(mergeCheckboxStates(content, null)).toBe(content);
    expect(mergeCheckboxStates(content, new Map())).toBe(content);
  });

  it('handles real-world acceptance criteria format', () => {
    const prBody = `
#### Acceptance criteria
- [ ] - Tests fail if weight bounds or turnover calculations allow negative weights
- [ ] - Existing functionality remains unchanged outside the stronger test coverage
    `.trim();

    // Agent completes first criterion and posts with checked box
    const agentReply = `
#### Acceptance criteria
- [x] - Tests fail if weight bounds or turnover calculations allow negative weights
- [ ] - Existing functionality remains unchanged outside the stronger test coverage
    `.trim();

    const existingStates = parseCheckboxStates(agentReply);
    expect(existingStates.size).toBe(1);

    // PR-meta refreshes from issue (unchecked) and merges agent's checked state
    const merged = mergeCheckboxStates(prBody, existingStates);

    expect(merged).toContain('- [x] - Tests fail if weight bounds or turnover calculations allow negative weights');
    expect(merged).toContain('- [ ] - Existing functionality remains unchanged outside the stronger test coverage');
  });
});

describe('ensureChecklist', () => {
  it('adds checkbox prefix to plain text lines', () => {
    const text = 'Task one\nTask two\nTask three';
    const result = ensureChecklist(text);

    expect(result).toBe('- [ ] Task one\n- [ ] Task two\n- [ ] Task three');
  });

  it('preserves existing checkbox formatting', () => {
    const text = '- [x] Completed task\n- [ ] Pending task';
    const result = ensureChecklist(text);

    expect(result).toBe('- [x] Completed task\n- [ ] Pending task');
  });

  it('returns placeholder for empty input', () => {
    expect(ensureChecklist('')).toBe('- [ ] —');
    expect(ensureChecklist('   ')).toBe('- [ ] —');
    expect(ensureChecklist(null)).toBe('- [ ] —');
  });
});

describe('extractBlock', () => {
  it('extracts content between markers', () => {
    const body = `
Some preamble text

<!-- auto-status-summary:start -->
#### Tasks
- [ ] Task one
- [x] Task two
<!-- auto-status-summary:end -->

Some footer text
    `.trim();

    const block = extractBlock(body, 'auto-status-summary');

    expect(block).toContain('#### Tasks');
    expect(block).toContain('- [ ] Task one');
    expect(block).toContain('- [x] Task two');
  });

  it('returns empty string if markers not found', () => {
    expect(extractBlock('no markers here', 'auto-status-summary')).toBe('');
    expect(extractBlock('', 'auto-status-summary')).toBe('');
    expect(extractBlock(null, 'auto-status-summary')).toBe('');
  });
});
