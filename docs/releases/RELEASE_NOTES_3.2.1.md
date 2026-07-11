# v3.2.1 — Audit-fix pass (security, correctness, ops, perf)

2026-07-11 (UTC+9)

A hardening release from the strongest multi-angle audit of v3.2.0. Findings were
verified against source by adversarial reviewers, fixes landed test-first, and a
second adversarial pre-commit review caught a real blind spot in the headline
security fix (see #1) before it shipped. Full suite green (2532), i18n green.

## Fixed — security

- **#1 Dashboard API-key / no-auth fallback now gates on the trusted client IP,
  not the bind host or the raw socket peer.** The dashboard let a caller present
  the shared chat API key as the admin password when `isLocalBindHost()` was
  true — but behind a same-host reverse proxy (`openresty → 127.0.0.1`) that is
  always true, and so is the raw socket peer (the proxy connects from loopback),
  so a *remote* user proxied in looked local. The gate now uses
  `dashboardClientIp()` (the real client via `X-Forwarded-For` when
  `TRUST_PROXY_X_FORWARDED_FOR=1`), applied to both `checkAuth` and
  `confirmReauth`. With XFF trust configured, a remote client is rejected (401)
  and a genuinely local one still works (200).
- **#7 Startup warning when the brute-force lockout can collapse to one bucket.**
  When bound to loopback with `X-Forwarded-For` trust off (the documented proxy
  topology), every request's IP is `127.0.0.1`, so failed logins share one
  lockout bucket — one bad actor can lock everyone out (including the operator,
  before auth). The server now warns at startup to enable
  `TRUST_PROXY_X_FORWARDED_FOR=1` + `TRUST_PROXY_HOPS`. (The default is not
  flipped — blindly trusting XFF is itself spoofable.)

## Fixed — correctness

- **#9 `top_k` is now part of the response cache key.** It is a live sampling
  knob on the DEVIN_CONNECT completion config; two requests differing only in
  `top_k` previously shared a cache slot and the second got a reply sampled under
  the first's `top_k`.
- **Caller sub-key whitespace no longer splits one tenant into two.**
  `usableSignal` now trims, so `" alice "` and `"alice"` map to the same
  `:user:` scope instead of different cache/cascade buckets.
- **OTP send-success message is HTML-escaped** before going into the dashboard
  DOM.

## Fixed — ops

- **#11 Docker self-update detects a failed pull.** `/images/create` streams
  JSONL and returns HTTP 200 even when the pull fails (unknown tag, registry
  error) — the failure is an `error`/`errorDetail` line in the body. The deployer
  used to treat 200 as success and silently stay on the old image; it now scans
  the stream and rejects on an error line (both pull sites).
- **`update.sh` health check no longer always passes.** `curl -sf … | head`
  returned `head`'s exit code (always 0), so a 500 `/health` or an unbound port
  passed; the curl output is now captured first so `curl -sf`'s own exit code is
  authoritative.

## Changed — cleanup / perf

- **Removed dead `hasCallerScope()`** (zero production imports) which had
  diverged from the live scope gate `hasPerUserScope()` — a latent cross-tenant
  cache landmine if ever imported.
- **`recordRequest` caches the current-hour stats bucket** (`O(n)` linear find →
  `O(1)`), invalidated on reset / import-replace.
- **Shutdown `saveAccountsSync` failure is logged** instead of silently
  swallowed.

## Verification

- `npm test` → **2532/0**.
- `node src/dashboard/check-i18n.js` green.
- Adversarial pre-commit review (3 scoped reviewers + verifier) — 1 HIGH blind
  spot in #1 caught and corrected, 1 LOW (classifier drift) fixed, rest clean.
- Live end-to-end: XFF-trust proxy scenario returns 401 for a remote client and
  200 for a local one.

Not in this release (require a real upstream token / trace to do safely):
account-pool in-flight-leak and `reportError` transient-guard fixes, and the
`sanitize.js` streaming-buffer perf change.
