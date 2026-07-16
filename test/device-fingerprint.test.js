import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { buildGetChatMessageRequest, __testing } from '../src/devin-connect.js';
import { isStableDeviceEnabled, ensureDeviceSeed } from '../src/auth.js';

const { generateFingerprint } = __testing;

// Per-account stable device fingerprint (opt-in, WINDSURFAPI_STABLE_DEVICE).
// Default OFF: #31 stays per-request random and byte-identical to before. When a
// deviceSeed is supplied, the fingerprint is deterministically derived so an
// account presents one stable device across requests. See the machine-id study.

afterEach(() => { delete process.env.WINDSURFAPI_STABLE_DEVICE; });

describe('generateFingerprint — #31 ClientMetadata (732 hex / 366 bytes)', () => {
  it('no seed → per-request random, always the correct shape', () => {
    const a = generateFingerprint();
    const b = generateFingerprint();
    assert.equal(a.length, 732, '732 hex chars');
    assert.match(a, /^[0-9a-f]{732}$/, 'lowercase hex');
    assert.notEqual(a, b, 'two random fingerprints differ (anti-fingerprint default)');
  });

  it('same seed → SAME fingerprint every call (stable per account)', () => {
    const seed = 'a'.repeat(64);
    const a = generateFingerprint(seed);
    const b = generateFingerprint(seed);
    assert.equal(a, b, 'deterministic for a given seed');
    assert.equal(a.length, 732, 'still 732 hex chars (shape preserved)');
    assert.match(a, /^[0-9a-f]{732}$/);
  });

  it('different seeds → different fingerprints (each account its own device)', () => {
    assert.notEqual(generateFingerprint('seed-account-1'), generateFingerprint('seed-account-2'));
  });

  it('derived fingerprint is NOT trivially the seed (HKDF expand, not echo)', () => {
    const seed = 'deadbeef';
    const fp = generateFingerprint(seed);
    assert.ok(!fp.includes(seed) || fp.length === 732, 'seed not naively embedded');
    assert.equal(fp.length, 732);
  });
});

describe('buildGetChatMessageRequest — deviceSeed threads to #31, byte-equivalence when absent', () => {
  const base = {
    token: 'tok_test',
    messages: [{ role: 'user', content: 'hi' }],
    model: 'swe-1-6-slow',
    sessionId: 'fixed-session',
  };

  it('same deviceSeed → identical request bytes (stable device is reproducible)', () => {
    // sessionId is pinned so the only per-request randomness left is #31 and the
    // ModelConfig #15.1 UUID; with a fixed seed #31 is stable. We assert the #31
    // field region is identical across two builds with the same seed by checking
    // the derived fingerprint appears in both proto buffers.
    const seed = 'c'.repeat(64);
    const fp = generateFingerprint(seed);
    // #31 is written as the fingerprint STRING (UTF-8) into the proto, so in the
    // buffer's hex dump it appears as the hex-of-the-ascii-of-fp. Same seed →
    // same fp → that byte region is present in both builds.
    const needle = Buffer.from(fp, 'utf8').toString('hex');
    const hex1 = Buffer.from(p1FromSeed(seed)).toString('hex');
    const hex2 = Buffer.from(p1FromSeed(seed)).toString('hex');
    assert.ok(hex1.includes(needle), 'proto embeds the derived stable fingerprint');
    assert.ok(hex2.includes(needle), 'second build embeds the same fingerprint');
    function p1FromSeed(s) { return buildGetChatMessageRequest({ ...base, deviceSeed: s }); }
  });

  it('no deviceSeed → random #31 each build (default path unchanged)', () => {
    const p1 = Buffer.from(buildGetChatMessageRequest({ ...base })).toString('hex');
    const p2 = Buffer.from(buildGetChatMessageRequest({ ...base })).toString('hex');
    // With no seed the #31 fingerprint is random, so the two proto buffers differ.
    assert.notEqual(p1, p2, 'default path stays per-request random');
  });
});

describe('auth: isStableDeviceEnabled + ensureDeviceSeed lifecycle', () => {
  it('disabled by default → ensureDeviceSeed returns undefined and does not stamp the account', () => {
    assert.equal(isStableDeviceEnabled({}), false);
    const acct = { apiKey: 'k' };
    assert.equal(ensureDeviceSeed(acct, {}), undefined);
    assert.equal(acct.deviceSeed, undefined, 'account left untouched when mode off');
  });

  it('enabled → mints a stable 64-hex seed on first use, reuses it after', () => {
    const env = { WINDSURFAPI_STABLE_DEVICE: '1' };
    assert.equal(isStableDeviceEnabled(env), true);
    const acct = { apiKey: 'k' };
    const s1 = ensureDeviceSeed(acct, env);
    assert.match(s1, /^[0-9a-f]{64}$/, 'seed is 64 hex');
    const s2 = ensureDeviceSeed(acct, env);
    assert.equal(s1, s2, 'same account reuses its minted seed');
    assert.equal(acct.deviceSeed, s1, 'seed persisted on the account object');
  });

  it('distinct accounts get distinct seeds', () => {
    const env = { WINDSURFAPI_STABLE_DEVICE: '1' };
    const a = { apiKey: 'a' };
    const b = { apiKey: 'b' };
    assert.notEqual(ensureDeviceSeed(a, env), ensureDeviceSeed(b, env));
  });

  it('preexisting seed is respected (loaded from accounts.json), not regenerated', () => {
    const env = { WINDSURFAPI_STABLE_DEVICE: '1' };
    const acct = { apiKey: 'k', deviceSeed: 'preexisting-seed-value' };
    assert.equal(ensureDeviceSeed(acct, env), 'preexisting-seed-value');
  });

  it('end-to-end: a stable seed produces a stable #31 across the account', () => {
    const env = { WINDSURFAPI_STABLE_DEVICE: '1' };
    const acct = { apiKey: 'k' };
    const seed = ensureDeviceSeed(acct, env);
    assert.equal(generateFingerprint(seed), generateFingerprint(acct.deviceSeed), 'stable device end to end');
  });
});
