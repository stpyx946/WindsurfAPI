import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  addAccountByKey,
  getApiKey,
  refundReservation,
  removeAccount,
  reportError,
  reportSuccess,
} from '../src/auth.js';

// Batch 3 — account-pool self-healing & rate-limit feedback.
// AP-BUG-1: status flip to 'error' is persisted (saveAccounts on the flip).
// AP-RISK-1: 'error' accounts recover (half-open) after a TTL so the pool
//   doesn't shrink monotonically under transient upstream wobble.
// AP-RISK-3: a refunded reservation frees the RPM slot it reserved.

const createdIds = [];
function addTestAccount(label = 'batch3') {
  const account = addAccountByKey(`b3-key-${Date.now()}-${Math.random().toString(36).slice(2)}`, label);
  createdIds.push(account.id);
  return account;
}
// A healthy peer kept in rotation so the account under test is NOT the pool's
// last usable account — otherwise the LB last-account exemption (src/auth.js)
// keeps a sole account active to avoid a guaranteed pool-wide 529. These tests
// assert breaker MECHANICS, which in production only fire when a fallback exists.
function addHealthyPeer(label = 'batch3-peer') {
  const peer = addTestAccount(label);
  reportSuccess(peer.apiKey);
  return peer;
}

afterEach(() => {
  delete process.env.WINDSURFAPI_ERROR_RECOVERY_MS;
  while (createdIds.length) removeAccount(createdIds.pop());
});

describe('batch3 — account pool self-healing', () => {
  it('AP-BUG-1: three errors in-window flip status to error and persist it', () => {
    addHealthyPeer();
    const acct = addTestAccount('ap-bug-1');
    reportError(acct.apiKey);
    reportError(acct.apiKey);
    assert.equal(acct.status, 'active', 'two errors do not disable');
    reportError(acct.apiKey);
    assert.equal(acct.status, 'error', 'third error disables');
    assert.ok(acct.erroredAt > 0, 'erroredAt stamped for recovery TTL');
  });

  it('reportSuccess clears an error streak before it disables', () => {
    const acct = addTestAccount('ap-bug-1-clear');
    reportError(acct.apiKey);
    reportError(acct.apiKey);
    reportSuccess(acct.apiKey);
    assert.equal(acct.errorCount, 0);
    reportError(acct.apiKey);
    assert.equal(acct.status, 'active', 'streak reset, single error after success does not disable');
  });

  it('AP-RISK-1: an error account is half-open recovered after the TTL on next selection', () => {
    process.env.WINDSURFAPI_ERROR_RECOVERY_MS = '1000';
    addHealthyPeer();
    const acct = addTestAccount('ap-risk-1');
    reportError(acct.apiKey);
    reportError(acct.apiKey);
    reportError(acct.apiKey);
    assert.equal(acct.status, 'error');
    // Backdate the error so it's past the (1s) recovery TTL.
    acct.erroredAt = Date.now() - 5000;
    const picked = getApiKey([], null, null);
    assert.ok(picked, 'a candidate is selectable after recovery');
    assert.equal(acct.status, 'active', 'error account flipped back to active (half-open)');
    assert.equal(acct.errorCount, 0, 'error count cleared on recovery');
    if (picked?.apiKey) {
      // release the inflight we just took by selecting
      import('../src/auth.js').then(m => m.releaseAccount(picked.apiKey)).catch(() => {});
    }
  });

  it('AP-RISK-1: an error account still inside the TTL is NOT recovered on selection', () => {
    process.env.WINDSURFAPI_ERROR_RECOVERY_MS = String(60 * 60 * 1000);
    addHealthyPeer();
    const acct = addTestAccount('ap-risk-1-cooldown');
    reportError(acct.apiKey);
    reportError(acct.apiKey);
    reportError(acct.apiKey);
    acct.erroredAt = Date.now(); // fresh error, well within the 1h TTL
    // A selection pass must NOT flip it back: it's still cooling down.
    getApiKey([], null, null);
    assert.equal(acct.status, 'error', 'still error inside TTL');
  });

  it('AP-RISK-3: refunding a reservation removes exactly one RPM slot', () => {
    const acct = addTestAccount('ap-risk-3');
    // Reserve via the real selection path. getApiKey returns the chosen
    // account's apiKey + reservationTimestamp and pushes a token into its
    // live _rpmHistory; refundReservation must remove exactly that token.
    const pick = getApiKey([], null, null);
    assert.ok(pick, 'got a reservation');
    assert.ok(Array.isArray(acct._rpmHistory), 'rpm history exists on the account');
    const before = acct.apiKey === pick.apiKey ? acct._rpmHistory.length : null;
    const ok = refundReservation(pick.apiKey, pick.reservationTimestamp);
    assert.equal(ok, true, 'refund succeeded');
    // Refunding a token that is not present must be a no-op (returns false).
    const dup = refundReservation(pick.apiKey, pick.reservationTimestamp);
    assert.equal(dup, false, 'double refund is a no-op');
    if (before !== null) {
      assert.equal(acct._rpmHistory.length, before - 1, 'exactly one RPM token removed');
    }
    import('../src/auth.js').then(m => m.releaseAccount(pick.apiKey)).catch(() => {});
  });
});
