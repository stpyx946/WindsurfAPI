import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SCRIPT = join(ROOT, 'scripts', 'cline-compat-verify.mjs');

// The Cline OpenAI-Compatible contract verifier ships as a partner-facing
// self-check. Its OFFLINE mode runs the full assertion suite against an
// in-process mock that emits the exact OpenAI-compatible shapes Cline needs.
// A green offline run proves the assertions are internally consistent — if
// someone breaks an assertion (or the mock), this test catches it in CI
// rather than letting a partner discover a bogus verifier.
describe('cline-compat-verify (offline contract self-test)', () => {
  it('all Cline contract checks pass against the in-process mock', () => {
    const r = spawnSync(process.execPath, [SCRIPT], {
      cwd: ROOT,
      encoding: 'utf8',
      env: { ...process.env, CLINE_VERIFY_REAL: '', VERIFY_TIMEOUT_MS: '30000' },
    });
    assert.equal(r.status, 0, `verifier exited ${r.status}:\n${r.stdout}\n${r.stderr}`);
    assert.match(r.stdout, /7\/7 checks passed/, `expected 7/7:\n${r.stdout}`);
    assert.match(r.stdout, /All Cline contract checks passed/);
  });
});
