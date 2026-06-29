#!/usr/bin/env node
// cascade-survival-probe — track whether the Windsurf/Cascade backend keeps
// serving requests across the July 1 2026 Cascade retirement deadline.
//
// WHY: Cognition retires Cascade (binary-level) on 2026-07-01, replacing the
// local agent with Devin Local + ACP. The open question that decides whether
// WindsurfAPI's gRPC reflection path survives is: does the *backend* API
// (server.self-serve.windsurf.com) keep serving, or does it go down with the
// local Cascade agent? Blogs don't answer this. We measure it directly.
//
// SAFETY MODEL — two modes:
//   * DEFAULT (health-only): ZERO billable model calls. Hits only the running
//     WindsurfAPI /health?verbose=1 for account / tier / LS / drought status.
//     This catches catastrophic backend death (auth path down, accounts flip
//     to expired/dead, LS pool gone) at zero cost.
//   * OPT-IN (--chat / PROBE_CHAT=1): additionally fires ONE cheap entitled
//     gemini-2.5-flash completion (max_tokens 8) to confirm the Cascade *chat*
//     path still yields a token. This IS a real, billable, free-tier call — it
//     spends the account's prompt allowance, so it is deliberately off by
//     default (mirrors the opt-in canary policy from commit 21393b9, which was
//     introduced after a force-probe sweep degraded a working free account).
//
// Each run appends a timestamped JSONL line so we can see the exact moment (if
// any) the backend stops working.
//
// Flags / env:
//   --dry-run   (or DRY_RUN=1)   print resolved config + planned requests, make
//                                NO network call at all, exit 0. Use this to
//                                validate the script anywhere, incl. a box with
//                                no server / no API key.
//   --chat      (or PROBE_CHAT=1) enable the billable gemini chat probe.
//
// Usage (on homecloud, key never leaves the box):
//   KEY=$(sed -n 's/^API_KEY=//p' ~/WindsurfAPI/.env)
//   # zero-billable continuous monitor (recommended cadence: every 30min):
//   API_KEY=$KEY node scripts/cascade-survival-probe.mjs
//   # opt-in chat-path liveness (spends free-tier credit — run sparingly):
//   API_KEY=$KEY node scripts/cascade-survival-probe.mjs --chat
// Schedule via cron every ~30min across 2026-06-30 .. 2026-07-02.

import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const argv = new Set(process.argv.slice(2));
const dryRun = argv.has('--dry-run') || process.env.DRY_RUN === '1';
const chatEnabled = argv.has('--chat') || process.env.PROBE_CHAT === '1';

const baseUrl = (process.env.BASE_URL || 'http://127.0.0.1:3003').replace(/\/+$/, '');
const apiKey = process.env.API_KEY || process.env.WINDSURFAPI_API_KEY || '';
const model = process.env.PROBE_MODEL || 'gemini-2.5-flash';
const outFile = process.env.PROBE_OUT || '.workflow-results/cascade-survival.jsonl';
const timeoutMs = Math.max(5_000, Number(process.env.PROBE_TIMEOUT_MS || 60_000));

if (dryRun) {
  // Make NO network call. Just show exactly what a real run would do.
  const plan = {
    mode: chatEnabled ? 'health+chat (billable opt-in)' : 'health-only (zero billable)',
    dry_run: true,
    baseUrl,
    api_key_present: !!apiKey,
    outFile,
    timeoutMs,
    planned_requests: [
      { step: 1, method: 'GET', path: '/health?verbose=1', billable: false },
      ...(chatEnabled
        ? [{ step: 2, method: 'POST', path: '/v1/chat/completions', model, max_tokens: 8, billable: true }]
        : [{ step: 2, skipped: true, reason: 'chat probe is opt-in (--chat / PROBE_CHAT=1)' }]),
    ],
  };
  console.log(JSON.stringify(plan, null, 2));
  process.exit(0);
}

if (!apiKey) {
  console.error('API_KEY is required (read it locally; do not paste it anywhere).');
  process.exit(2);
}

async function call(path, opts = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const started = Date.now();
  try {
    const res = await fetch(`${baseUrl}${path}`, {
      ...opts,
      signal: controller.signal,
      headers: { authorization: `Bearer ${apiKey}`, ...(opts.headers || {}) },
    });
    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch {}
    return { status: res.status, body, ms: Date.now() - started };
  } catch (err) {
    return { status: 0, error: String(err?.message || err), ms: Date.now() - started };
  } finally {
    clearTimeout(timer);
  }
}

const record = { ts: new Date().toISOString() };

// 1. health — accounts + LS state (no upstream call)
const health = await call('/health?verbose=1');
record.health_status = health.status;
record.accounts = health.body?.accounts || null;
record.version = health.body?.version || null;
record.drought = health.body?.drought || null;
record.health_ok = health.status === 200;

// 2. cheap entitled chat probe — OPT-IN ONLY (billable, spends free-tier credit).
//    Confirms the Cascade chat path still yields a token. Off by default so the
//    every-30min monitor never burns the account (see commit 21393b9 canary policy).
if (chatEnabled) {
  const chat = await call('/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      max_tokens: 8,
      messages: [{ role: 'user', content: 'Reply with the single word: ALIVE' }],
    }),
  });
  record.chat_status = chat.status;
  record.chat_ms = chat.ms;
  const content = chat.body?.choices?.[0]?.message?.content;
  record.chat_text = typeof content === 'string' ? content.slice(0, 40) : null;
  record.chat_error = chat.body?.error?.type || chat.error || null;
  record.cascade_alive = chat.status === 200 && !!content;
} else {
  record.chat_skipped = 'opt-in (--chat / PROBE_CHAT=1)';
}

mkdirSync(dirname(outFile), { recursive: true });
appendFileSync(outFile, JSON.stringify(record) + '\n');
console.log(JSON.stringify(record, null, 2));

// Exit code: in chat mode, success = chat path alive; in health-only mode,
// success = health endpoint reachable + returning 200 (the zero-cost signal).
const ok = chatEnabled ? !!record.cascade_alive : record.health_ok;
process.exit(ok ? 0 : 1);
