import { createHash } from 'crypto';
import { log } from './config.js';
import { trustedClientIp } from './net-safety.js';

function sha256Hex(value) {
  return createHash('sha256').update(String(value || '')).digest('hex');
}

// A body field only carries a usable scope signal when it's a string with
// non-whitespace content. An empty/whitespace value must NOT mint a scope:
// user:"" would otherwise hash to the constant sha256("") prefix
// (e3b0c44298fc1c14...), collapsing every distinct end user of a shared key
// into one :user: segment and re-enabling cross-tenant cascade/cache bleed.
// Returns the TRIMMED non-empty string, else ''. Trimming matters: returning the
// raw value made " alice " and "alice" hash into different :user: buckets, so the
// same end user got split tenant scopes across requests with incidental
// whitespace. (audit S6)
function usableSignal(value) {
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : '';
}

// Extract a per-user / per-session signal from the request body so two
// different end users sharing one API key get different conversation pool
// scopes. v2.0.25 HIGH-3: chat & responses now look at body.user /
// conversation / previous_response_id / metadata.{conversation_id,session_id}.
//
// metadata.user_id is INTENTIONALLY NOT inspected here — handlers/messages.js
// has a specialized parser for it (Claude Code's JSON-encoded
// {device_id, session_id, account_uuid} shape) and appends its own
// `:user:<digest>` to keep the two extraction paths from double-stamping
// the same callerKey.
//
// The returned subkey is appended to the API-key callerKey so reuse stays
// pinned to (apiKey, user/session). Returns '' when no usable signal.
export function extractBodyCallerSubKey(body) {
  if (!body || typeof body !== 'object') return '';
  const user = usableSignal(body.user);
  if (user) return sha256Hex(user).slice(0, 16);
  const candidates = [
    usableSignal(body?.metadata?.conversation_id),
    usableSignal(body.conversation),
    usableSignal(body.previous_response_id),
    usableSignal(body?.metadata?.session_id),
  ].filter(Boolean);
  if (!candidates.length) return '';
  return sha256Hex(candidates.join('|')).slice(0, 16);
}

// IP + UA fallback used when an apiKey-mode caller has no explicit body
// user signal. Without this, every Claude Code / claudecode CLI on a
// self-hosted single-user setup hits "shared API key, no per-user scope"
// and cascade reuse stays disabled — exactly the symptom reported in
// #93 follow-up by zhangzhang-bit (claude-opus-4-6-thinking, msgs growing
// 33→97 across turns, reuse=false on every Cascade started).
//
// Two physical clients sharing one apiKey will land on different IP/UA
// hashes and stay isolated; same client across turns lands on the same
// hash and lets the cascade pool reuse the upstream session.
//
// Client-IP trust policy (XFF hop counting) lives in net-safety.js:trustedClientIp
// — the single source of truth shared with src/dashboard/api.js so the caller-key
// fingerprint and the dashboard lockout bucket can't drift apart. See that
// function for the audit-H2 (default-ignore XFF) and audit-P1/XFF-1 (count from
// the right, never take the spoofable leftmost) rationale.
//
// ⚠️ Behaviour change (audit S4): this module previously captured
// TRUST_PROXY_X_FORWARDED_FOR into a module-load `const`, so flipping the env at
// runtime had no effect here while the dashboard copy read it live — a latent
// drift. trustedClientIp reads it live, matching the dashboard and this module's
// own already-live hop reader. The xff-spoof test still fresh-imports per case,
// so its coverage is unchanged.

function ipUaFingerprint(req) {
  const ip = trustedClientIp(req);
  const ua = req?.headers?.['user-agent'] || '';
  if (!ip && !ua) return '';
  return sha256Hex(`${ip}\0${ua}`).slice(0, 16);
}

export function callerKeyFromRequest(req, apiKey = '', body = null) {
  const bodySubKey = body ? extractBodyCallerSubKey(body) : '';
  const hasUserInBody = !!(body && usableSignal(body.user));
  // Don't log the raw body.user — OpenAI's `user` field is often an end-user
  // email or stable account id (PII). bodySubKey is already its hash.
  log.info('[caller-key] hasUser=%s subKey=%s', hasUserInBody ? 'yes' : 'no', bodySubKey || '(none)');
  if (apiKey) {
    const base = `api:${sha256Hex(apiKey).slice(0, 32)}`;
    if (bodySubKey) return `${base}:user:${bodySubKey}`;
    const ipua = ipUaFingerprint(req);
    return ipua ? `${base}:client:${ipua}` : base;
  }
  const sessionId = req?.headers?.['x-dashboard-session'] || req?.headers?.['x-session-id'] || '';
  if (sessionId) {
    const base = `session:${sha256Hex(sessionId).slice(0, 32)}`;
    return bodySubKey ? `${base}:user:${bodySubKey}` : base;
  }
  const ip = trustedClientIp(req);
  const ua = req?.headers?.['user-agent'] || '';
  const base = `client:${sha256Hex(`${ip}\0${ua}`).slice(0, 32)}`;
  return bodySubKey ? `${base}:user:${bodySubKey}` : base;
}

// NOTE: a `hasCallerScope()` export used to live here. It was DEAD (zero
// production imports — grep) and had silently diverged from the LIVE gate
// `hasPerUserScope()` in src/handlers/chat.js (the live one gates the guessed
// `:client:` bucket behind SINGLE_TENANT_CACHE; this dead twin trusted it
// unconditionally). Keeping a diverged, more-permissive copy exported was a
// cross-tenant-cache landmine if anyone had imported it. Removed 2026-07-11
// (audit S1). The authoritative scope gate is chat.js:hasPerUserScope.
