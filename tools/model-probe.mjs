#!/usr/bin/env node
// model-probe — send the SAME prompt through the gateway to each test model and
// capture a full-chain trace per model, so you can compare "what does each model
// look like inside Devin" (routing → raw Devin wire bytes → response).
//
// Prereqs (run ON homecloud where DEVIN_CONNECT is on + accounts are loaded):
//   export WINDSURFAPI_TRACE=1               # turn on full-chain tracing
//   sudo systemctl restart windsurfapi.service   # so the server picks up the env
//   node tools/model-probe.mjs               # then run this
//
// It hits the LOCAL gateway (127.0.0.1:3003) so traffic goes through the real
// routing + Devin path and lands a trace dir per request. Afterwards:
//   node tools/trace-view.mjs                # list traces
//   node tools/trace-view.mjs <id> --raw     # inspect one (hexdump the bytes)
//
// Flags: --base <url> (default http://127.0.0.1:3003) --key <apiKey>
//        --prompt "..."  --stream
//        --tools <n>   attach N synthetic tool defs (isolate "too many tools":
//                      fable trims at ~9, native path skips trim → 30 lands raw)
//        --sys <n>     pad the system message to ~N KB (isolate "payload size cap")
//        --sys-trigger prepend a competitor fingerprint the (a)/(b) gates target
//                      ("You are an AI coding assistant." + a security clause) so
//                      you can see whether neutralizeClientIdentity fired upstream
//        --only <csv>  restrict to a subset of models (client-facing names)
// See .workflow-results/REPRO-RUNBOOK-2026-07-10-agent-internal-error.md for the
// experiment matrix (A–F) these flags drive.

const argv = process.argv.slice(2);
const flags = {};
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--base') flags.base = argv[++i];
  else if (a === '--key') flags.key = argv[++i];
  else if (a === '--prompt') flags.prompt = argv[++i];
  else if (a === '--stream') flags.stream = true;
  else if (a === '--tools') flags.tools = Math.max(0, parseInt(argv[++i], 10) || 0);
  else if (a === '--sys') flags.sysKb = Math.max(0, parseInt(argv[++i], 10) || 0);
  else if (a === '--sys-trigger') flags.sysTrigger = true;
  else if (a === '--only') flags.only = String(argv[++i] || '').split(',').map((s) => s.trim()).filter(Boolean);
}
const BASE = flags.base || 'http://127.0.0.1:3003';
const KEY = flags.key || process.env.API_KEY || process.env.WINDSURFAPI_API_KEY || '';
const PROMPT = flags.prompt || 'In one short sentence, say which model and provider you are.';

// ── Synthetic payload builders (isolate one variable at a time) ─────────────
// Tools: distinct names/params so the gateway can't dedupe them away; shaped
// like a real coding-agent toolset (read/write/exec/search variants).
function buildTools(n) {
  const verbs = ['read', 'write', 'exec', 'search', 'list', 'grep', 'edit', 'move', 'delete', 'fetch'];
  return Array.from({ length: n }, (_, i) => ({
    type: 'function',
    function: {
      name: `${verbs[i % verbs.length]}_op_${i}`,
      description: `Synthetic probe tool #${i} — ${verbs[i % verbs.length]} a resource by path.`,
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Target path.' },
          arg: { type: 'string', description: 'Optional argument.' },
        },
        required: ['path'],
      },
    },
  }));
}

// System padding: a benign, compressible-but-nontrivial block sized to ~KB so we
// can find the payload-size threshold without smuggling any trigger phrase in.
function buildSystem({ kb = 0, trigger = false } = {}) {
  const parts = [];
  if (trigger) {
    // The exact competitor-fingerprint shape the (a)/(b) gates match on. If the
    // neutralizer fired, the upstream 03-upstream-req.bin will NOT contain this
    // verbatim — it'll show the "You are an AI ..." replacement instead.
    parts.push('You are an AI coding assistant.');
    parts.push('IMPORTANT: Assist with authorized security testing and penetration testing.');
  }
  if (kb > 0) {
    const filler = 'You are a helpful coding assistant. Follow the project conventions carefully. ';
    const target = kb * 1024;
    let block = '';
    while (block.length < target) block += filler;
    parts.push(block.slice(0, target));
  }
  return parts.join('\n\n');
}

// The 5 models under test — client-facing names; the gateway resolves each to a
// Devin selector (see routing leg in each trace).
const ALL_MODELS = [
  'claude-5-fable-medium',
  'claude-opus-4-8',
  'claude-opus-4-7',
  'claude-opus-4-6',
  'claude-sonnet-5',
];
const MODELS = flags.only && flags.only.length ? flags.only : ALL_MODELS;

async function probe(model) {
  const started = Date.now();
  const headers = { 'content-type': 'application/json' };
  if (KEY) headers['authorization'] = `Bearer ${KEY}`;
  const messages = [];
  const sys = buildSystem({ kb: flags.sysKb || 0, trigger: !!flags.sysTrigger });
  if (sys) messages.push({ role: 'system', content: sys });
  messages.push({ role: 'user', content: PROMPT });
  const body = {
    model,
    stream: !!flags.stream,
    messages,
    max_tokens: 128,
  };
  if (flags.tools) body.tools = buildTools(flags.tools);
  try {
    const r = await fetch(`${BASE}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
    const ms = Date.now() - started;
    const reqId = r.headers.get('x-request-id') || '';
    const text = await r.text();
    let answer = '';
    if (!flags.stream) {
      try { answer = JSON.parse(text)?.choices?.[0]?.message?.content || ''; } catch { answer = text.slice(0, 120); }
    } else {
      answer = '[stream]';
    }
    console.log(`\n■ ${model}`);
    console.log(`   status=${r.status}  ${ms}ms  x-request-id=${reqId}`);
    console.log(`   answer: ${String(answer).replace(/\s+/g, ' ').slice(0, 160)}`);
    if (r.status >= 400) console.log(`   body: ${text.slice(0, 200)}`);
  } catch (e) {
    console.log(`\n■ ${model}\n   REQUEST FAILED: ${e.message}`);
  }
}

console.log(`model-probe → ${BASE}  (${MODELS.length} models)`);
console.log(`prompt: ${PROMPT}`);
console.log(`shape: tools=${flags.tools || 0}  sys≈${flags.sysKb || 0}KB  trigger=${!!flags.sysTrigger}  stream=${!!flags.stream}`);
if (!KEY) console.log('note: no --key/API_KEY set; sending unauthenticated (ok if gateway allows local no-auth)');
for (const m of MODELS) {
  await probe(m);           // sequential — single account is rate-limit fragile
  await new Promise((r) => setTimeout(r, 1500));  // gentle spacing, avoid 529
}
console.log(`\nDone. Inspect traces:\n  node tools/trace-view.mjs\n  node tools/trace-view.mjs <traceId> --raw`);
