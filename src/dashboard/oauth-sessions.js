// In-memory OAuth onboarding session store.
//
// Ported from CLIProxyAPI's oauth_sessions.go three-layer decoupling
// (start / callback / poll), collapsed to a single-process in-memory Map — no
// file mailbox or fs-watcher needed here. This is what lets a PUBLIC deploy
// onboard a Devin account without the operator's machine ever receiving a
// localhost OAuth callback: the user finishes login in their own browser and
// pastes the whole redirect URL back; the server extracts the token from it.
//
// Contract (mirrors the workflow-poll shape the dashboard already uses):
//   registerSession()      -> state (opaque, URL-safe)
//   getStatus(state)       -> { status: 'wait' | 'error' | 'ok', error? }
//   completeSession(state) -> drop (success)
//   failSession(state,msg) -> mark error with a message
//
// A missing session reads as 'ok' (already consumed / never existed) — the same
// terminal semantics CLIProxyAPI uses, so a poll after success cleanly ends. The
// one exception: a session that expired while still pending is remembered as an
// 'error' (ERR_SESSION_EXPIRED) so a slow browser login can't masquerade as ok.

import { randomBytes } from 'crypto';

const sessions = new Map(); // state -> { provider, status, error, expiresAt }
const TTL_MS = 10 * 60 * 1000; // 10 minutes to finish a browser login

// Tombstones for sessions that ran out their TTL while still pending. Without
// this, a purged pending session is indistinguishable from a consumed-success
// one (both just "missing"), so getStatus would report a false 'ok' — the
// browser login timed out but the dashboard would claim the account was added.
// We remember the state briefly so a late poll gets a truthful expired-error.
const expired = new Map(); // state -> tombstoneExpiresAt
const TOMBSTONE_MS = 30 * 60 * 1000; // keep the expired verdict long enough for a poll to observe it

// State charset guard (from oauth_sessions.go:208-233): the state is used as a
// map key and as an anti-CSRF token echoed through the OAuth round-trip. Reject
// anything that isn't a short, URL-safe token so it can't smuggle path
// traversal or oversized junk.
const STATE_RE = /^[A-Za-z0-9._-]{1,128}$/;

export function validateState(state) {
  return typeof state === 'string' && STATE_RE.test(state);
}

export function registerSession(provider = 'windsurf') {
  purgeExpired();
  // 24 random bytes -> 32-char base64url, well within STATE_RE.
  const state = randomBytes(24).toString('base64url');
  sessions.set(state, {
    provider: String(provider || 'windsurf'),
    status: 'pending',
    error: '',
    expiresAt: Date.now() + TTL_MS,
  });
  return state;
}

export function getSession(state) {
  purgeExpired();
  return sessions.get(state) || null;
}

export function getStatus(state) {
  purgeExpired();
  const s = sessions.get(state);
  if (!s) {
    // A pending session that ran out its TTL leaves a tombstone: report the
    // timeout instead of a false 'ok'. Anything else missing is consumed/unknown.
    if (expired.has(state)) return { status: 'error', error: 'ERR_SESSION_EXPIRED' };
    return { status: 'ok' }; // consumed or unknown -> terminal
  }
  if (s.status === 'error') return { status: 'error', error: s.error || 'unknown_error' };
  if (s.status === 'ok') return { status: 'ok' };
  return { status: 'wait' };
}

export function completeSession(state) {
  sessions.delete(state);
  expired.delete(state); // a real success must never read back as expired
}

export function failSession(state, message) {
  const s = sessions.get(state);
  if (s) {
    s.status = 'error';
    s.error = String(message || 'unknown_error');
  }
}

function purgeExpired() {
  const now = Date.now();
  for (const [k, v] of sessions) {
    if (v.expiresAt < now) {
      // Only pending sessions get a tombstone — a completed/failed one already
      // carries its own terminal verdict (dropped, or status:'error').
      if (v.status === 'pending') expired.set(k, now + TOMBSTONE_MS);
      sessions.delete(k);
    }
  }
  for (const [k, deadline] of expired) {
    if (deadline < now) expired.delete(k);
  }
}

// Periodic sweep so abandoned sessions don't accumulate. unref so it never keeps
// the process alive on its own.
const _sweep = setInterval(purgeExpired, 60 * 1000);
if (typeof _sweep.unref === 'function') _sweep.unref();

// Test seam.
export const __testing = { sessions, expired, TTL_MS, TOMBSTONE_MS, STATE_RE, purgeExpired };
