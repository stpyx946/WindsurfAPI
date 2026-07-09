// Backend-switch hot-swap — env → runtime-config migration (v3.0.2).
//
// Complements test/runtime-config-hardening.test.js (which unit-tests the
// getter/setter with INJECTED env) by covering the two seams that migration
// actually depends on in production:
//
//   1. Three-tier fallback through the DEFAULT process.env path (no env arg):
//      runtime-config explicit override > process.env > historical default.
//      The hardening suite only exercises injected env; here we mutate the real
//      process.env to prove the default-arg resolution wired up correctly.
//   2. selectBackend() HOT-SWAP: a runtime-config override must steer routing
//      even when env is empty ({}), while an UNSET override falls back to the
//      injected env — preserving backend-router.test.js's __testing semantics
//      and the hard backward-compat contract (old env-only deploys unchanged).
//
// In-memory only: _resetRuntimeConfigForTests reseeds state; persist writes into
// the temp DATA_DIR from test/setup-env.mjs, so no real project file is touched.
// process.env keys are saved/restored per test so nothing leaks across cases.

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  _resetRuntimeConfigForTests,
  getBackendSwitch, getBackendSwitches,
  getBackendSwitchOverrides, setBackendSwitches,
} from '../src/runtime-config.js';
import { selectBackend, usesCascadeFlow, BACKEND } from '../src/backend-router.js';

// Every env var the backend switches map to. Saved/restored around each test so
// mutating the real process.env (to test the default-arg path) can't leak.
const SWITCH_ENV_VARS = [
  'DEVIN_CONNECT', 'DEVIN_ONLY', 'DEVIN_CLI_MODE',
  'DEVIN_CLI_ALLOW_CLIENT_TOOLS', 'DEVIN_CONNECT_LOGIN_HOST_FALLBACK',
  'DEVIN_CONNECT_AUTO_RELOGIN',
];

describe('backend switch: three-tier fallback via default process.env', () => {
  let saved;
  beforeEach(() => {
    _resetRuntimeConfigForTests();
    saved = {};
    for (const k of SWITCH_ENV_VARS) {
      saved[k] = process.env[k];
      delete process.env[k]; // clean slate → historical default
    }
  });
  afterEach(() => {
    for (const k of SWITCH_ENV_VARS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  it('tier 3: no override + no env → historical default', () => {
    assert.equal(getBackendSwitch('devinConnect'), false);
    assert.equal(getBackendSwitch('devinOnly'), false);
    assert.equal(getBackendSwitch('devinCliMode'), 'print');
  });

  it('tier 2: no override, real process.env=1 → env wins (no env arg)', () => {
    process.env.DEVIN_CONNECT = '1';
    process.env.DEVIN_CLI_MODE = 'acp';
    assert.equal(getBackendSwitch('devinConnect'), true, 'reads real process.env');
    assert.equal(getBackendSwitch('devinCliMode'), 'acp');
  });

  it('tier 1: runtime-config override beats real process.env', () => {
    process.env.DEVIN_CONNECT = '1';       // env says ON
    setBackendSwitches({ devinConnect: false }); // override says OFF
    assert.equal(getBackendSwitch('devinConnect'), false, 'override wins over process.env');

    process.env.DEVIN_ONLY = '0';          // env says OFF
    setBackendSwitches({ devinOnly: true });     // override says ON
    assert.equal(getBackendSwitch('devinOnly'), true, 'override wins over process.env');
  });

  it('null clears override → resolution drops back to real process.env', () => {
    process.env.DEVIN_ONLY = '1';
    setBackendSwitches({ devinOnly: false });
    assert.equal(getBackendSwitch('devinOnly'), false, 'override active');
    setBackendSwitches({ devinOnly: null });
    assert.equal(getBackendSwitch('devinOnly'), true, 'cleared → falls back to process.env=1');
  });

  it('getBackendSwitches() reflects the effective mix of override + process.env', () => {
    process.env.DEVIN_CLI_MODE = 'acp';
    process.env.DEVIN_CONNECT = '1';
    setBackendSwitches({ devinConnect: false }); // override masks the env
    const all = getBackendSwitches();
    assert.equal(all.devinConnect, false, 'override wins');
    assert.equal(all.devinCliMode, 'acp', 'env fallback');
    assert.equal(all.devinOnly, false, 'default');
  });
});

describe('setBackendSwitches: whitelist + coercion + null-clear', () => {
  beforeEach(() => _resetRuntimeConfigForTests());

  it('rejects unknown keys, including the credential-store env switches', () => {
    const out = setBackendSwitches({
      devinConnect: true,
      DEVIN_CONNECT_ALLOW_REMOTE_CRED_STORE: true, // security boundary — must NOT be settable
      DEVIN_CONNECT_CRED_KEY: 'abc',
      bogusSwitch: true,
    });
    assert.strictEqual(out.devinConnect, true, 'known key applied');
    assert.equal('DEVIN_CONNECT_ALLOW_REMOTE_CRED_STORE' in out, false, 'cred-store flag rejected');
    assert.equal('DEVIN_CONNECT_CRED_KEY' in out, false, 'cred-store key rejected');
    assert.equal('bogusSwitch' in out, false, 'junk key rejected');
  });

  it('devinCliMode accepts only acp/print; junk leaves prior override intact', () => {
    assert.equal(setBackendSwitches({ devinCliMode: 'acp' }).devinCliMode, 'acp');
    assert.equal(setBackendSwitches({ devinCliMode: 'print' }).devinCliMode, 'print');
    setBackendSwitches({ devinCliMode: 'acp' });
    setBackendSwitches({ devinCliMode: 'garbage' }); // ignored
    assert.equal(getBackendSwitchOverrides().devinCliMode, 'acp', 'junk mode did not overwrite');
    setBackendSwitches({ devinCliMode: 42 }); // non-string junk ignored too
    assert.equal(getBackendSwitchOverrides().devinCliMode, 'acp');
  });

  it('boolean switches coerce truthy/falsy input to real booleans', () => {
    assert.strictEqual(setBackendSwitches({ allowClientTools: 'yes' }).allowClientTools, true);
    assert.strictEqual(setBackendSwitches({ allowClientTools: 0 }).allowClientTools, false);
    assert.strictEqual(setBackendSwitches({ loginHostFallback: 1 }).loginHostFallback, true);
    assert.strictEqual(setBackendSwitches({ autoRelogin: '' }).autoRelogin, false);
  });

  it('null clears an override back to unset (env fallback)', () => {
    setBackendSwitches({ devinConnect: true });
    assert.equal(getBackendSwitchOverrides().devinConnect, true);
    setBackendSwitches({ devinConnect: null });
    assert.equal(getBackendSwitchOverrides().devinConnect, null, 'null → unset');
  });

  it('non-object patch is a no-op that returns current overrides', () => {
    setBackendSwitches({ devinOnly: true });
    const out = setBackendSwitches(null);
    assert.equal(out.devinOnly, true, 'state untouched by null patch');
  });
});

// selectBackend HOT-SWAP: the payoff of the migration. A runtime-config override
// must drive routing with env:{} (no env var present), and an unset override must
// fall through to the injected env exactly as before the migration.
describe('selectBackend hot-swap via runtime-config (env not required)', () => {
  let saved;
  beforeEach(() => {
    _resetRuntimeConfigForTests();
    // Isolate from any real process.env — selectBackend is called with env:{} so
    // only runtime-config should decide. Belt-and-suspenders: clear the vars too.
    saved = {};
    for (const k of SWITCH_ENV_VARS) { saved[k] = process.env[k]; delete process.env[k]; }
  });
  afterEach(() => {
    for (const k of SWITCH_ENV_VARS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  it('runtime-config devinConnect=true routes to DEVIN_CONNECT with env:{}', () => {
    setBackendSwitches({ devinConnect: true });
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_CLAUDE_4_5_SONNET' }, env: {} });
    assert.equal(sel.backend, BACKEND.DEVIN_CONNECT, 'override alone selected DEVIN_CONNECT');
    assert.equal(sel.flow, 'devin_connect');
    assert.equal(sel.reason, 'devin_connect');
  });

  it('runtime-config devinOnly=true forces special-agent with env:{}', () => {
    setBackendSwitches({ devinOnly: true });
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_GPT_5' }, env: {} });
    assert.equal(sel.flow, 'special_agent');
    assert.equal(sel.reason, 'devin_only');
    assert.equal(sel.backend, BACKEND.DEVIN_PRINT, 'default sub-mode');
    assert.ok(!usesCascadeFlow(sel));
  });

  it('runtime-config devinCliMode=acp steers the DEVIN_ONLY sub-mode with env:{}', () => {
    setBackendSwitches({ devinOnly: true, devinCliMode: 'acp' });
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: {} });
    assert.equal(sel.flow, 'special_agent');
    assert.equal(sel.backend, BACKEND.DEVIN_ACP, 'override sub-mode honoured');
  });

  it('runtime-config devinConnect wins over env DEVIN_ONLY (precedence preserved)', () => {
    setBackendSwitches({ devinConnect: true });
    const sel = selectBackend({ modelInfo: { backend: 'special_agent' }, env: { DEVIN_ONLY: '1' } });
    assert.equal(sel.backend, BACKEND.DEVIN_CONNECT, 'connect still highest precedence');
    assert.equal(sel.flow, 'devin_connect');
  });

  it('override=false MASKS an env var that is set to 1 (kill-switch off wins)', () => {
    setBackendSwitches({ devinConnect: false });
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: { DEVIN_CONNECT: '1' } });
    assert.equal(sel.flow, 'cascade', 'explicit off override beats env=1');
    assert.notEqual(sel.backend, BACKEND.DEVIN_CONNECT);
  });
});

// Backward-compat regression: with an EMPTY runtime-config (every override null,
// the fresh-deploy state), selectBackend must behave byte-identically to the
// pre-migration env-only reads. This is the contract that keeps old deploys safe.
describe('selectBackend backward-compat: empty runtime-config → env-only behaviour', () => {
  beforeEach(() => _resetRuntimeConfigForTests()); // all overrides null

  it('env DEVIN_CONNECT=1 still routes to DEVIN_CONNECT (unchanged)', () => {
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: { DEVIN_CONNECT: '1' } });
    assert.equal(sel.backend, BACKEND.DEVIN_CONNECT);
    assert.equal(sel.flow, 'devin_connect');
  });

  it('env DEVIN_ONLY=1 still forces special-agent (unchanged)', () => {
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: { DEVIN_ONLY: '1' } });
    assert.equal(sel.flow, 'special_agent');
    assert.equal(sel.reason, 'devin_only');
  });

  it('env DEVIN_ONLY=1 + DEVIN_CLI_MODE=acp still yields devin-acp (unchanged)', () => {
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: { DEVIN_ONLY: '1', DEVIN_CLI_MODE: 'acp' } });
    assert.equal(sel.backend, BACKEND.DEVIN_ACP);
  });

  it('no env, no override → cascade for a real model (unchanged)', () => {
    const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: {} });
    assert.equal(sel.flow, 'cascade');
  });

  it('truthy-string env values do NOT enable (exact "1" guard preserved)', () => {
    for (const v of ['true', 'yes', '2', 'on', ' ']) {
      const sel = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: { DEVIN_CONNECT: v } });
      assert.notEqual(sel.flow, 'devin_connect', `DEVIN_CONNECT=${JSON.stringify(v)} must not enable`);
    }
    const padded = selectBackend({ modelInfo: { modelUid: 'MODEL_X' }, env: { DEVIN_CONNECT: ' 1 ' } });
    assert.equal(padded.flow, 'devin_connect', 'whitespace-padded 1 tolerated');
  });
});
