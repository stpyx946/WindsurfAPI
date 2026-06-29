// Regression for the misleading "probe_pending" gate (homecloud finding 2026-06-29).
//
// When the only callable models are unavailable, chat.js returns 403. The
// message branches on hasUnprobedActive: a TRULY fresh account → "账号刚添加，
// 等 tier 检测" / type=probe_pending; an already-probed account → plan/tier
// "model_not_entitled". The original predicate keyed solely off
// userStatusLastFetched, so a canary-probed account whose tier resolved to
// 'expired' (but whose GetUserStatus came back empty) was mislabeled as
// "just added, still detecting tier" — sending users to re-probe forever.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { accountIsProbed } from '../src/handlers/chat.js';

describe('accountIsProbed — probe_pending gating predicate', () => {
  it('treats a genuinely fresh account as NOT probed', () => {
    assert.equal(accountIsProbed({ tier: 'unknown', lastProbed: 0, userStatusLastFetched: 0 }), false);
    assert.equal(accountIsProbed({}), false);
    assert.equal(accountIsProbed(null), false);
  });

  it('treats a canary-probed account (lastProbed set) as probed even with empty GetUserStatus', () => {
    // The live homecloud repro: tier flipped to 'expired', userStatusLastFetched
    // stayed 0, but lastProbed was stamped by the canary sweep.
    assert.equal(accountIsProbed({ tier: 'expired', lastProbed: Date.now(), userStatusLastFetched: 0 }), true);
  });

  it('treats any resolved (non-unknown) tier as probed', () => {
    assert.equal(accountIsProbed({ tier: 'expired', lastProbed: 0, userStatusLastFetched: 0 }), true);
    assert.equal(accountIsProbed({ tier: 'free', lastProbed: 0, userStatusLastFetched: 0 }), true);
    assert.equal(accountIsProbed({ tier: 'pro' }), true);
  });

  it('treats a landed GetUserStatus as probed regardless of tier label', () => {
    assert.equal(accountIsProbed({ tier: 'unknown', userStatusLastFetched: Date.now() }), true);
  });
});
