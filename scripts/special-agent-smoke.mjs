#!/usr/bin/env node

const baseUrl = (process.env.BASE_URL || process.env.WINDSURFAPI_BASE_URL || 'http://127.0.0.1:3003').replace(/\/+$/, '');
const apiKey = process.env.API_KEY || process.env.WINDSURFAPI_API_KEY || '';
const model = process.env.MODEL || process.env.SPECIAL_AGENT_SMOKE_MODEL || 'swe-1.6-fast';
const requestTimeoutMs = Math.max(5_000, Number(process.env.SPECIAL_AGENT_SMOKE_TIMEOUT_MS || 180_000));
const requireEnabled = process.env.SPECIAL_AGENT_SMOKE_REQUIRE_ENABLED !== '0';
const prompt = process.env.SPECIAL_AGENT_SMOKE_PROMPT || 'Reply exactly SPECIAL_AGENT_OK.';

if (!apiKey) {
  console.error('API_KEY is required.');
  process.exit(2);
}

function compactText(text, max = 1200) {
  const s = String(text || '').replace(/\s+/g, ' ').trim();
  return s.length > max ? `${s.slice(0, max)}...<truncated ${s.length - max} chars>` : s;
}

async function fetchJson(path, opts = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), requestTimeoutMs);
  try {
    const res = await fetch(`${baseUrl}${path}`, {
      ...opts,
      signal: controller.signal,
      headers: {
        authorization: `Bearer ${apiKey}`,
        ...(opts.headers || {}),
      },
    });
    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch {}
    return { status: res.status, body, text };
  } finally {
    clearTimeout(timer);
  }
}

const health = await fetchJson('/health?verbose=1');
const specialAgent = health.body?.specialAgent || null;
if (requireEnabled && !specialAgent?.enabled) {
  console.log(JSON.stringify({
    ok: false,
    stage: 'preflight',
    error: 'special-agent backend is disabled',
    specialAgent,
    hint: 'Set WINDSURFAPI_SPECIAL_AGENT_BACKEND=devin-cli and DEVIN_CLI_MODE=print or acp before running this smoke.',
  }, null, 2));
  process.exit(1);
}

const started = Date.now();
const chat = await fetchJson('/v1/chat/completions', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    model,
    stream: false,
    max_tokens: 128,
    messages: [{ role: 'user', content: prompt }],
  }),
});

const content = chat.body?.choices?.[0]?.message?.content || '';
const out = {
  stage: 'positive_text_chat',
  ok: chat.status >= 200 && chat.status < 300 && !chat.body?.error,
  model,
  status: chat.status,
  latencyMs: Date.now() - started,
  specialAgent,
  content: compactText(content),
  error: chat.body?.error || null,
};

console.log(JSON.stringify(out, null, 2));
if (!out.ok) process.exit(1);

async function runNegativeSmoke(name, payload, expectedType) {
  const started = Date.now();
  const result = await fetchJson('/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const error = result.body?.error || null;
  const ok = result.status === 400 && error?.type === expectedType;
  console.log(JSON.stringify({
    stage: name,
    result: ok ? 'PASS' : 'FAIL',
    ok,
    expectedStatus: 400,
    expectedType,
    status: result.status,
    latencyMs: Date.now() - started,
    error,
    body: ok ? undefined : result.body,
    text: ok ? undefined : compactText(result.text),
  }, null, 2));
  return ok;
}

const negativeResults = [];
negativeResults.push(await runNegativeSmoke('negative_tools', {
  model,
  stream: false,
  max_tokens: 128,
  messages: [{ role: 'user', content: 'read package.json' }],
  tools: [{
    type: 'function',
    function: {
      name: 'Read',
      description: 'Read tool',
      parameters: { type: 'object', properties: {} },
    },
  }],
}, 'unsupported_tool_boundary'));

negativeResults.push(await runNegativeSmoke('negative_media', {
  model,
  stream: false,
  max_tokens: 128,
  messages: [{
    role: 'user',
    content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,xxx' } }],
  }],
}, 'unsupported_media'));

if (negativeResults.some(ok => !ok)) process.exit(1);
