import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { getVersionInfo, VERSION } from '../src/version.js';

const ENV_KEYS = [
  'WINDSURFAPI_BUILD_VERSION',
  'WINDSURFAPI_BUILD_COMMIT',
  'WINDSURFAPI_BUILD_COMMIT_MESSAGE',
  'WINDSURFAPI_BUILD_COMMIT_DATE',
  'WINDSURFAPI_BUILD_BRANCH',
];

function withEnv(vars, fn) {
  const prev = Object.fromEntries(ENV_KEYS.map(k => [k, process.env[k]]));
  try {
    for (const k of ENV_KEYS) delete process.env[k];
    Object.assign(process.env, vars);
    return fn();
  } finally {
    for (const [k, v] of Object.entries(prev)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  }
}

describe('version metadata', () => {
  it('uses build env metadata for archive/container deployments', () => withEnv({
    WINDSURFAPI_BUILD_VERSION: '2.0.98',
    WINDSURFAPI_BUILD_COMMIT: '682a7001234567890abcdef',
    WINDSURFAPI_BUILD_COMMIT_MESSAGE: 'fix: smoke',
    WINDSURFAPI_BUILD_COMMIT_DATE: '2026-06-05T00:00:00Z',
    WINDSURFAPI_BUILD_BRANCH: 'master',
  }, () => {
    const info = getVersionInfo();
    assert.equal(info.version, '2.0.98');
    assert.equal(info.commit, '682a70012345');
    assert.equal(info.commitMessage, 'fix: smoke');
    assert.equal(info.commitDate, '2026-06-05T00:00:00Z');
    assert.equal(info.branch, 'master');
  }));

  it('falls back to package version', () => withEnv({}, () => {
    assert.equal(getVersionInfo().version, VERSION);
  }));
});
