import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { sanitizeText, PathSanitizeStream, sanitizeToolCall } from '../src/sanitize.js';

const MARKER = '<workspace>';

// Leaked Windsurf paths are redacted to a short structural placeholder.
// The marker MUST be visibly intentional to humans/LLMs and MUST NOT look
// like a real absolute path — see sanitize.js header for the full marker
// history and why both shell-loop and prose-loop failure modes matter.

describe('sanitizeText', () => {
  it('redacts /tmp/windsurf-workspace paths', () => {
    assert.equal(sanitizeText('/tmp/windsurf-workspace/src/index.js'), MARKER);
  });

  it('redacts bare /tmp/windsurf-workspace', () => {
    assert.equal(sanitizeText('/tmp/windsurf-workspace'), MARKER);
  });

  it('redacts per-account workspace paths', () => {
    assert.equal(
      sanitizeText('/home/user/projects/workspace-abc12345/package.json'),
      MARKER
    );
  });

  it('redacts /opt/windsurf', () => {
    assert.equal(sanitizeText('/opt/windsurf/language_server'), MARKER);
  });

  it('leaves normal text unchanged', () => {
    const text = 'Hello, this is a normal response.';
    assert.equal(sanitizeText(text), text);
  });

  it('handles multiple patterns in one string', () => {
    const input = 'Editing /tmp/windsurf-workspace/a.js and /opt/windsurf/bin';
    const result = sanitizeText(input);
    assert.equal(result, `Editing ${MARKER} and ${MARKER}`);
  });

  // Issue #86 follow-up: oaskdosakdoakd reported `C:\home\user\projects\workspace-devinxse`
  // leaking despite the Unix-only regex catching `/home/user/projects/workspace-skxwsx01`.
  // The model (often GLM running on Windows clients) hallucinates Windows-prefixed
  // forms of the workspace path. Defensive: cover backslash, drive prefix, and
  // mixed separators.
  it('redacts Windows-style workspace path with C: drive prefix', () => {
    assert.equal(
      sanitizeText('C:\\home\\user\\projects\\workspace-devinxse\\src\\index.js'),
      MARKER
    );
  });

  it('redacts Windows-style workspace path with backslashes only', () => {
    assert.equal(
      sanitizeText('\\home\\user\\projects\\workspace-skxwsx01\\src\\index.js'),
      MARKER
    );
  });

  it('redacts mixed-separator workspace path (drive prefix + forward slashes)', () => {
    assert.equal(
      sanitizeText('C:\\home/user/projects/workspace-devinxse/src/index.js'),
      MARKER
    );
  });

  it('redacts lowercase-drive Windows workspace path', () => {
    assert.equal(
      sanitizeText('d:\\home\\user\\projects\\workspace-x12345\\file.txt'),
      MARKER
    );
  });

  it('returns non-strings unchanged', () => {
    assert.equal(sanitizeText(null), null);
    assert.equal(sanitizeText(undefined), undefined);
    assert.equal(sanitizeText(''), '');
  });
});

describe('PathSanitizeStream', () => {
  it('sanitizes a complete path in one chunk', () => {
    const stream = new PathSanitizeStream();
    const out = stream.feed('/tmp/windsurf-workspace/file.js is here');
    const rest = stream.flush();
    assert.equal(out + rest, `${MARKER} is here`);
  });

  it('handles path split across chunks', () => {
    const stream = new PathSanitizeStream();
    let result = '';
    result += stream.feed('Look at /tmp/windsurf');
    result += stream.feed('-workspace/config.yaml for details');
    result += stream.flush();
    assert.equal(result, `Look at ${MARKER} for details`);
  });

  it('handles partial prefix at buffer end', () => {
    const stream = new PathSanitizeStream();
    let result = '';
    result += stream.feed('path is /tmp/win');
    result += stream.feed('dsurf-workspace/x.js done');
    result += stream.flush();
    assert.equal(result, `path is ${MARKER} done`);
  });

  it('flushes clean text immediately', () => {
    const stream = new PathSanitizeStream();
    const out = stream.feed('Hello world ');
    assert.equal(out, 'Hello world ');
  });
});

describe('PathSanitizeStream — incremental cut equivalence (audit S5)', () => {
  // The audit-S5 O(N²)→O(N) optimization adds a fast path that resumes the
  // held-construct scan across chunks instead of re-walking the whole buffer.
  // The guarantee is that streaming an input in ANY chunking must produce the
  // exact same bytes as one-shot sanitizeText() of the whole input. This oracle
  // exhausts every chunk-boundary placement for representative inputs — the
  // strongest possible pin against the fast path ever diverging.
  function streamAll(input, cutPoints) {
    const s = new PathSanitizeStream();
    let out = '';
    let prev = 0;
    for (const c of cutPoints) { out += s.feed(input.slice(prev, c)); prev = c; }
    out += s.feed(input.slice(prev));
    out += s.flush();
    return out;
  }

  const CASES = [
    'plain text with no sensitive content at all, streamed piecewise',
    'edit /tmp/windsurf-workspace/src/deeply/nested/path/file.js then continue talking',
    'a very long path /home/user/projects/workspace-abc12345/' + 'sub/'.repeat(40) + 'end.js done',
    'before <workspace_layout>' + 'tree line\n'.repeat(30) + '</workspace_layout> after',
    'mix /opt/windsurf/x and <user_information>secret\n'.repeat(10) + '</user_information> tail',
    'two paths /tmp/windsurf-workspace/a.js and /tmp/windsurf-workspace/b.js',
    'unterminated tail ends right at a prefix /tmp/win',
    'unclosed block never terminates <workspace_information>partial content forever',
  ];

  it('every single-cut position matches one-shot sanitizeText', () => {
    for (const input of CASES) {
      const expected = sanitizeText(input);
      for (let i = 0; i <= input.length; i++) {
        const got = streamAll(input, [i]);
        assert.equal(got, expected, `single cut @${i} of ${JSON.stringify(input.slice(0, 40))}…`);
      }
    }
  });

  it('character-by-character (worst-case chunking) matches one-shot', () => {
    for (const input of CASES) {
      const cuts = Array.from({ length: input.length }, (_, i) => i + 1);
      assert.equal(streamAll(input, cuts), sanitizeText(input), `char-split of ${JSON.stringify(input.slice(0, 40))}…`);
    }
  });

  it('randomized multi-cut chunkings match one-shot (fuzz)', () => {
    let seed = 1234567;
    const rand = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    for (const input of CASES) {
      const expected = sanitizeText(input);
      for (let trial = 0; trial < 50; trial++) {
        const cuts = [];
        let pos = 0;
        while (pos < input.length) {
          pos += 1 + Math.floor(rand() * 5);
          if (pos < input.length) cuts.push(pos);
        }
        assert.equal(streamAll(input, cuts), expected, `fuzz trial ${trial}`);
      }
    }
  });

  it('long streamed path stays O(N): scales roughly linearly, not quadratically', () => {
    // Feed a growing unterminated path one char at a time and confirm total work
    // scales sub-quadratically. We can't assert wall-clock reliably, but we can
    // assert the fast path holds correctness at large N cheaply enough to finish.
    for (const N of [2000, 8000]) {
      const path = '/home/user/projects/workspace-x/' + 'a'.repeat(N);
      const s = new PathSanitizeStream();
      let out = '';
      for (const ch of path) out += s.feed(ch);
      out += s.feed(' end');
      out += s.flush();
      assert.equal(out, sanitizeText(path + ' end'), `N=${N} incremental path`);
    }
  });
});

describe('sanitizeToolCall', () => {
  it('sanitizes argumentsJson paths', () => {
    const tc = { name: 'Read', argumentsJson: '{"path":"/tmp/windsurf-workspace/f.js"}' };
    const result = sanitizeToolCall(tc);
    assert.equal(result.argumentsJson, `{"path":"${MARKER}"}`);
  });

  it('sanitizes input object string values', () => {
    const tc = { name: 'Read', input: { file_path: '/home/user/projects/workspace-abc12345/src/x.ts' } };
    const result = sanitizeToolCall(tc);
    assert.equal(result.input.file_path, MARKER);
  });

  it('returns null/undefined unchanged', () => {
    assert.equal(sanitizeToolCall(null), null);
    assert.equal(sanitizeToolCall(undefined), undefined);
  });
});

describe('REDACTED_PATH marker shape (prose-loop regression)', () => {
  // The marker is emitted verbatim into model-facing text. It must be
  // visibly structural so a user who asks for the project path does not
  // mistake the redaction marker for a real one-character answer.
  const marker = sanitizeText('/tmp/windsurf-workspace');

  it('is distinguishable from a natural ellipsis', () => {
    assert.notEqual(marker, '…');
  });

  it('is wrapped in structural delimiters', () => {
    assert.match(marker, /^(<[^<>]+>|\([^()]+\)|\[[^\[\]]+\])$/);
  });

  it('stays terse', () => {
    assert.ok(marker.length < 16, `marker must stay short: got ${JSON.stringify(marker)}`);
  });

  it('is not absolute-path-shaped', () => {
    assert.ok(!marker.includes('/'), 'marker must not contain / (looks like a Unix path)');
    assert.ok(!marker.includes('\\'), 'marker must not contain \\ (looks like a Windows path)');
  });
});
