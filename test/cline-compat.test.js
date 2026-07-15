import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  normalizeToolCallArgs,
  detectClineClient,
  resolveClineCompat,
  recordArgRepair,
  getClineCompatStats,
  resetClineCompatStats,
  stripClineNamespace,
} from '../src/handlers/cline-compat.js';

// cline-compat is the standalone, pluggable Cline compatibility layer. It never
// changes standard /v1 behavior unless a request is explicitly routed through
// /v1/cline/* (endpoint opt-in) or detected as Cline with the master toggle on.

describe('normalizeToolCallArgs', () => {
  it('coalesces empty / whitespace / unparseable to "{}"', () => {
    // @ai-sdk/openai-compatible drops a tool_call whose arguments fail
    // isParsableJson (vercel/ai#6687). Claude sends "" for parameterless tools.
    assert.equal(normalizeToolCallArgs(''), '{}');
    assert.equal(normalizeToolCallArgs('   '), '{}');
    assert.equal(normalizeToolCallArgs('\n\t '), '{}');
    assert.equal(normalizeToolCallArgs('{bad'), '{}');
    assert.equal(normalizeToolCallArgs('not json'), '{}');
    assert.equal(normalizeToolCallArgs(undefined), '{}');
    assert.equal(normalizeToolCallArgs(null), '{}');
  });

  it('preserves already-valid JSON objects and arrays verbatim', () => {
    assert.equal(normalizeToolCallArgs('{"city":"Paris"}'), '{"city":"Paris"}');
    assert.equal(normalizeToolCallArgs('{}'), '{}');
    assert.equal(normalizeToolCallArgs('{"a":1,"b":[2,3]}'), '{"a":1,"b":[2,3]}');
  });

  it('returns a form that is ALWAYS JSON.parse-able (the @ai-sdk gate), even for JSON-illegal whitespace', () => {
    // JS String.trim() strips the full Unicode White_Space set, but JSON only
    // permits space/tab/LF/CR. If the function returned the original padded
    // string, @ai-sdk's isParsableJson would still reject it and drop the tool
    // call. So the RESULT must itself parse. ASCII-space padding is collapsed
    // (semantically identical, still valid); JSON-illegal whitespace (NBSP, BOM,
    // line/para separators) must not survive in a way that breaks the parse.
    for (const input of ['  {"a":1}  ', ' {}', '﻿{"x":1}', '{} ', '　{"y":2}　']) {
      const out = normalizeToolCallArgs(input);
      assert.doesNotThrow(() => JSON.parse(out), `result must parse for input ${JSON.stringify(input)} → ${JSON.stringify(out)}`);
    }
    // ASCII-space padding collapses to the trimmed valid JSON.
    assert.equal(normalizeToolCallArgs('  {"a":1}  '), '{"a":1}');
  });

  it('does NOT coerce a bare JSON scalar/array to {} (only object args are valid tool args, but a parseable value passes the SDK gate)', () => {
    // The SDK gate is isParsableJson, not is-object. A parseable scalar passes
    // it, so we must not rewrite it (that would change the model's intent).
    assert.equal(normalizeToolCallArgs('[1,2]'), '[1,2]');
    assert.equal(normalizeToolCallArgs('123'), '123');
    assert.equal(normalizeToolCallArgs('"x"'), '"x"');
  });
});

describe('detectClineClient', () => {
  it('detects Cline by User-Agent', () => {
    assert.equal(detectClineClient({ 'user-agent': 'Cline/3.2.0' }), true);
    assert.equal(detectClineClient({ 'user-agent': 'cline vscode extension' }), true);
    assert.equal(detectClineClient({ 'user-agent': 'Mozilla/5.0 ... Cline/1.0' }), true);
  });

  it('does not misfire on non-Cline clients', () => {
    assert.equal(detectClineClient({ 'user-agent': 'curl/8.4.0' }), false);
    assert.equal(detectClineClient({ 'user-agent': 'claude-cli/1.0' }), false);
    assert.equal(detectClineClient({ 'user-agent': '' }), false);
    assert.equal(detectClineClient({}), false);
    assert.equal(detectClineClient(null), false);
  });
});

describe('resolveClineCompat', () => {
  it('endpoint path is active even when master toggle is OFF (explicit opt-in namespace)', () => {
    const r = resolveClineCompat({ path: '/v1/cline/chat/completions', headers: {}, masterEnabled: false });
    assert.equal(r.active, true);
    assert.equal(r.source, 'endpoint');
  });

  it('standard path with master OFF and non-Cline UA is inactive (byte-identical)', () => {
    const r = resolveClineCompat({ path: '/v1/chat/completions', headers: { 'user-agent': 'curl' }, masterEnabled: false });
    assert.equal(r.active, false);
  });

  it('detection path requires master ON', () => {
    const off = resolveClineCompat({ path: '/v1/chat/completions', headers: { 'user-agent': 'Cline/3.2' }, masterEnabled: false });
    assert.equal(off.active, false);
    const on = resolveClineCompat({ path: '/v1/chat/completions', headers: { 'user-agent': 'Cline/3.2' }, masterEnabled: true });
    assert.equal(on.active, true);
    assert.equal(on.source, 'detect');
  });

  it('master ON but non-Cline client on standard path stays inactive (no blanket behavior change)', () => {
    const r = resolveClineCompat({ path: '/v1/chat/completions', headers: { 'user-agent': 'curl' }, masterEnabled: true });
    assert.equal(r.active, false);
  });
});

describe('stripClineNamespace', () => {
  it('rewrites /v1/cline/* to canonical /v1/*', () => {
    assert.equal(stripClineNamespace('/v1/cline/chat/completions'), '/v1/chat/completions');
    assert.equal(stripClineNamespace('/v1/cline/models'), '/v1/models');
  });
  it('leaves non-cline paths unchanged', () => {
    assert.equal(stripClineNamespace('/v1/chat/completions'), '/v1/chat/completions');
    assert.equal(stripClineNamespace('/v1/models'), '/v1/models');
    assert.equal(stripClineNamespace('/dashboard/api/x'), '/dashboard/api/x');
  });
});

describe('cline-compat stats', () => {
  beforeEach(() => resetClineCompatStats());

  it('counts arg repairs', () => {
    assert.equal(getClineCompatStats().argRepairs, 0);
    recordArgRepair();
    recordArgRepair();
    assert.equal(getClineCompatStats().argRepairs, 2);
  });
});
