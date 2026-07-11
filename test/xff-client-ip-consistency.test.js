// audit S4 — the "trusted client IP" (X-Forwarded-For hop counting) logic used
// by the per-caller pool/cache scope (src/caller-key.js) and by the dashboard
// brute-force lockout bucket (src/dashboard/api.js) MUST stay identical. They
// used to be two byte-for-byte private copies kept in sync only by a "MUST stay
// identical" comment, with a latent drift: caller-key.js captured
// TRUST_PROXY_X_FORWARDED_FOR into a module-load const while api.js read it live.
//
// Both now delegate to the single source of truth net-safety.js:trustedClientIp.
// These tests pin that function's contract AND assert the two former call sites
// still agree, so a future edit can't silently re-fork them and re-open XFF-1
// (an attacker rotating a spoofed leftmost XFF to dodge the lockout).

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { trustedClientIp, trustedProxyHops } from '../src/net-safety.js';

function mkReq(xff, ip, ua = 'ua') {
  return { headers: { 'x-forwarded-for': xff, 'user-agent': ua }, socket: { remoteAddress: ip } };
}

describe('net-safety trustedClientIp — XFF trust policy', () => {
  const off = {}; // TRUST_PROXY_X_FORWARDED_FOR unset

  it('default: ignores XFF, returns socket peer', () => {
    assert.equal(trustedClientIp(mkReq('1.2.3.4', '203.0.113.1'), off), '203.0.113.1');
  });

  it('default: spoofed XFF cannot change the returned IP', () => {
    const victim = trustedClientIp(mkReq(undefined, '203.0.113.1'), off);
    const attacker = trustedClientIp(mkReq('203.0.113.1', '198.51.100.9'), off);
    assert.notEqual(victim, attacker);
  });

  it('trust mode: reads the entry before the trusted hop (from the RIGHT)', () => {
    const env = { TRUST_PROXY_X_FORWARDED_FOR: '1' };
    // one trusted proxy appended 203.0.113.9 on the right; attacker prepended a spoof
    assert.equal(trustedClientIp(mkReq('evil, 203.0.113.9', '10.0.0.1'), env), '203.0.113.9');
    // rotating the spoofed leftmost value must NOT change the result
    assert.equal(
      trustedClientIp(mkReq('9.9.9.1, 203.0.113.9', '10.0.0.1'), env),
      trustedClientIp(mkReq('9.9.9.2, 203.0.113.9', '10.0.0.1'), env)
    );
  });

  it('trust mode: honours TRUST_PROXY_HOPS=2', () => {
    const env = { TRUST_PROXY_X_FORWARDED_FOR: '1', TRUST_PROXY_HOPS: '2' };
    assert.equal(trustedClientIp(mkReq('evil, 203.0.113.20, 172.16.0.1', '10.0.0.1'), env), '203.0.113.20');
    assert.equal(trustedProxyHops(env), 2);
  });

  it('trust mode: header shorter than hop count falls back to socket peer', () => {
    const env = { TRUST_PROXY_X_FORWARDED_FOR: '1', TRUST_PROXY_HOPS: '3' };
    assert.equal(trustedClientIp(mkReq('203.0.113.30', '198.51.100.9'), env), '198.51.100.9');
  });

  it('trustedProxyHops: default 1, rejects non-int / <1', () => {
    assert.equal(trustedProxyHops({}), 1);
    assert.equal(trustedProxyHops({ TRUST_PROXY_HOPS: '0' }), 1);
    assert.equal(trustedProxyHops({ TRUST_PROXY_HOPS: 'x' }), 1);
    assert.equal(trustedProxyHops({ TRUST_PROXY_HOPS: '5' }), 5);
  });

  it('env is read LIVE (not captured) — flipping trust mid-process takes effect', () => {
    const req = mkReq('203.0.113.9', '10.0.0.1');
    assert.equal(trustedClientIp(req, {}), '10.0.0.1');                                  // off → peer
    assert.equal(trustedClientIp(req, { TRUST_PROXY_X_FORWARDED_FOR: '1' }), '203.0.113.9'); // on → XFF
  });
});

describe('audit S4 — caller-key.js and dashboard/api.js agree on the client IP', () => {
  // The dashboard's dashboardClientIp is module-private, so we verify agreement
  // through the shared function both delegate to: for any request + env, the IP
  // that scopes the caller-key must equal the IP that buckets the lockout. If a
  // future edit re-introduces a divergent private copy in either file, that copy
  // won't be exercised here — but the intent is preserved by keeping BOTH files
  // importing trustedClientIp (asserted structurally below).
  it('trustedClientIp is deterministic for the same req+env', () => {
    const req = mkReq('evil, 203.0.113.9', '10.0.0.1');
    const env = { TRUST_PROXY_X_FORWARDED_FOR: '1' };
    assert.equal(trustedClientIp(req, env), trustedClientIp(req, env));
  });

  it('both caller-key.js and dashboard/api.js import the shared trustedClientIp', async () => {
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const root = fileURLToPath(new URL('..', import.meta.url));
    const callerKeySrc = readFileSync(root + 'src/caller-key.js', 'utf8');
    const apiSrc = readFileSync(root + 'src/dashboard/api.js', 'utf8');
    assert.match(callerKeySrc, /trustedClientIp/, 'caller-key.js must use the shared trustedClientIp');
    assert.match(apiSrc, /trustedClientIp/, 'dashboard/api.js must use the shared trustedClientIp');
    // Guard against a re-introduced private copy of the socket-peer read pattern.
    assert.doesNotMatch(callerKeySrc, /function\s+clientIp\s*\(/, 'no private clientIp copy in caller-key.js');
    assert.doesNotMatch(apiSrc, /function\s+dashboardTrustedProxyHops\s*\(/, 'no private hop copy in api.js');
  });
});
