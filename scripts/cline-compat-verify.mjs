#!/usr/bin/env node
/**
 * Cline OpenAI-Compatible contract verifier.
 *
 * Cline's "OpenAI Compatible" provider is Vercel `@ai-sdk/openai-compatible`
 * (includeUsage:true). It talks standard `/v1/chat/completions` with streaming,
 * native `tools`/`tool_calls` deltas, `stream_options.include_usage`, and an
 * optional `/v1/models` listing. This script asserts a gateway satisfies every
 * request shape Cline sends so we can hand a partner a green self-check.
 *
 * Two modes:
 *   - OFFLINE (default): spins up an in-process mock that returns the exact
 *     OpenAI-compatible shapes, then runs the assertion suite against it. This
 *     proves the ASSERTIONS themselves are correct (a green offline run means the
 *     contract checks are sound), with zero upstream billing.
 *   - REAL (`--real` or CLINE_VERIFY_REAL=1): runs the identical suite against a
 *     live gateway (BASE_URL + API_KEY). A green real run means the gateway
 *     honors the Cline contract end to end.
 *
 * Zero-dep: built-in `node:http` + global `fetch` only.
 *
 * Env / flags:
 *   --real | CLINE_VERIFY_REAL=1   hit a live gateway instead of the mock
 *   BASE_URL   gateway base, no trailing slash (default http://127.0.0.1:3003)
 *   API_KEY    downstream auth (default 'test'; required in real mode)
 *   MODEL      model id to exercise (default 'swe-1-6-slow', a free selector)
 *   VERIFY_TIMEOUT_MS   per-request timeout (default 120000)
 *
 * Exit code 0 = all checks passed, 1 = one or more failed, 2 = bad usage.
 */

import { createServer } from 'node:http';

const REAL = process.argv.includes('--real') || process.env.CLINE_VERIFY_REAL === '1';
const API_KEY = process.env.API_KEY || 'test';
const MODEL = process.env.MODEL || 'swe-1-6-slow';
const TIMEOUT_MS = Number(process.env.VERIFY_TIMEOUT_MS || 120000);

// ── result plumbing ────────────────────────────────────────────────
const results = [];
function record(name, ok, detail) {
  results.push({ name, ok, detail });
  const tag = ok ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
  console.log(`  ${tag}  ${name}${detail ? ` — ${detail}` : ''}`);
}
async function check(name, fn) {
  try {
    const detail = await fn();
    record(name, true, detail || '');
  } catch (err) {
    record(name, false, String(err?.message || err));
  }
}
function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// ── HTTP client ────────────────────────────────────────────────────
async function post(base, path, body, { stream = false } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(base + path, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (stream) return res; // caller reads the SSE body
    const text = await res.text();
    let json = null;
    try { json = JSON.parse(text); } catch {}
    return { status: res.status, json, text };
  } finally {
    clearTimeout(timer);
  }
}

async function get(base, path) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(base + path, {
      headers: { authorization: `Bearer ${API_KEY}` },
      signal: ctrl.signal,
    });
    const text = await res.text();
    let json = null;
    try { json = JSON.parse(text); } catch {}
    return { status: res.status, json, text };
  } finally {
    clearTimeout(timer);
  }
}

// Parse an SSE `/v1/chat/completions` stream into ordered `data:` payloads.
// Returns { events: [obj...], sawDone: bool }. `[DONE]` is recorded, not parsed.
async function readSSE(res) {
  assert(res.ok, `stream HTTP ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  const events = [];
  let sawDone = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trimEnd();
      buf = buf.slice(idx + 1);
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') { sawDone = true; continue; }
      try { events.push(JSON.parse(payload)); } catch {}
    }
  }
  return { events, sawDone };
}

// ── the Cline contract suite ───────────────────────────────────────
const WEATHER_TOOL = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get the current weather for a city.',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
      additionalProperties: false,
    },
  },
};

async function runSuite(base) {
  // 1. Non-stream chat: OpenAI envelope with choices[0].message.content.
  await check('non-stream /v1/chat/completions returns OpenAI envelope', async () => {
    const { status, json } = await post(base, '/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'Say hello in one word.' }],
      stream: false,
    });
    assert(status === 200, `HTTP ${status}`);
    assert(json?.object === 'chat.completion', `object=${json?.object}`);
    assert(Array.isArray(json.choices) && json.choices.length > 0, 'no choices');
    assert(typeof json.choices[0].message?.content === 'string', 'no message.content');
    assert(json.choices[0].finish_reason != null, 'no finish_reason');
    return `finish_reason=${json.choices[0].finish_reason}`;
  });

  // 2. Streaming chat: role-first delta chain + [DONE].
  await check('streaming /v1/chat/completions emits deltas + [DONE]', async () => {
    const res = await post(base, '/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'Count: one two three.' }],
      stream: true,
    }, { stream: true });
    const { events, sawDone } = await readSSE(res);
    assert(events.length > 0, 'no SSE events');
    assert(events.every(e => e.object === 'chat.completion.chunk'), 'non-chunk object in stream');
    const text = events.map(e => e.choices?.[0]?.delta?.content || '').join('');
    assert(text.length > 0, 'empty streamed content');
    assert(sawDone, 'missing [DONE] terminator');
    return `${events.length} chunks, ${text.length}c`;
  });

  // 3. stream_options.include_usage → a usage-bearing tail frame.
  await check('stream_options.include_usage yields a usage tail frame', async () => {
    const res = await post(base, '/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'Hi.' }],
      stream: true,
      stream_options: { include_usage: true },
    }, { stream: true });
    const { events } = await readSSE(res);
    const usageFrame = events.find(e => e.usage && typeof e.usage.total_tokens === 'number');
    assert(usageFrame, 'no frame carried usage.total_tokens');
    assert(typeof usageFrame.usage.prompt_tokens === 'number', 'usage.prompt_tokens missing');
    assert(typeof usageFrame.usage.completion_tokens === 'number', 'usage.completion_tokens missing');
    return `usage total=${usageFrame.usage.total_tokens}`;
  });

  // 4. Native tools → streamed tool_calls fragments that reassemble.
  await check('tools → streamed tool_calls (name + id + reassembled arguments)', async () => {
    const res = await post(base, '/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'What is the weather in Paris? Use the tool.' }],
      tools: [WEATHER_TOOL],
      tool_choice: 'auto',
      stream: true,
    }, { stream: true });
    const { events } = await readSSE(res);
    // Collect tool_call deltas by index and reassemble arguments.
    const byIndex = new Map();
    let finishReason = null;
    for (const e of events) {
      const choice = e.choices?.[0];
      const tcs = choice?.delta?.tool_calls;
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      if (!Array.isArray(tcs)) continue;
      for (const tc of tcs) {
        const i = tc.index ?? 0;
        const acc = byIndex.get(i) || { id: '', name: '', args: '' };
        if (tc.id) acc.id = tc.id;
        if (tc.function?.name) acc.name = tc.function.name;
        if (tc.function?.arguments) acc.args += tc.function.arguments;
        byIndex.set(i, acc);
      }
    }
    assert(byIndex.size > 0, 'no tool_calls in stream');
    const call = byIndex.get(0);
    assert(call.id, 'tool_call missing id');
    assert(call.name === 'get_weather', `tool name=${call.name}`);
    const parsed = JSON.parse(call.args); // must be valid JSON once reassembled
    assert(typeof parsed === 'object' && parsed !== null, 'arguments not a JSON object');
    assert(finishReason === 'tool_calls', `finish_reason=${finishReason}`);
    return `${call.name}(${call.args})`;
  });

  // 5. reasoning_effort passes through without a 4xx/5xx.
  await check('reasoning_effort passthrough does not error', async () => {
    const { status, json } = await post(base, '/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'Briefly: 2+2?' }],
      reasoning_effort: 'low',
      stream: false,
    });
    assert(status === 200, `HTTP ${status} (${json?.error?.message || 'no body'})`);
    assert(json?.choices?.[0]?.message, 'no message on reasoning_effort response');
    return 'accepted reasoning_effort=low';
  });

  // 6. /v1/models returns an OpenAI-shaped catalog.
  await check('/v1/models returns a catalog', async () => {
    const { status, json } = await get(base, '/v1/models');
    assert(status === 200, `HTTP ${status}`);
    assert(json?.object === 'list', `object=${json?.object}`);
    assert(Array.isArray(json.data) && json.data.length > 0, 'empty model list');
    assert(json.data.every(m => typeof m.id === 'string'), 'a model entry lacks id');
    return `${json.data.length} models`;
  });

  // 7. Multi-turn history with a role:tool turn must not 500 (Cline's tool loop).
  await check('multi-turn history incl. role:tool does not 500', async () => {
    const { status, json } = await post(base, '/v1/chat/completions', {
      model: MODEL,
      messages: [
        { role: 'user', content: 'What is the weather in Paris?' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [{
            id: 'call_1', type: 'function',
            function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
          }],
        },
        { role: 'tool', tool_call_id: 'call_1', content: '{"tempC":21,"sky":"clear"}' },
      ],
      stream: false,
    });
    assert(status < 500, `HTTP ${status} (${json?.error?.message || 'no body'})`);
    assert(status === 200, `HTTP ${status} (expected 200 for a valid tool-result turn)`);
    return `follow-up turn accepted (HTTP ${status})`;
  });
}

// ── offline mock: a minimal OpenAI-compatible gateway ───────────────
// Returns exactly the shapes the suite asserts, so a green offline run means
// the assertions are self-consistent (not that any real gateway passes).
function sseChunk(obj) { return `data: ${JSON.stringify(obj)}\n\n`; }

function startMock() {
  const server = createServer((req, res) => {
    const chunks = [];
    req.on('data', d => chunks.push(d));
    req.on('end', () => {
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
      const url = req.url;

      if (req.method === 'GET' && url === '/v1/models') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({
          object: 'list',
          data: [
            { id: MODEL, object: 'model', owned_by: 'windsurfapi' },
            { id: 'claude-opus-4-8', object: 'model', owned_by: 'windsurfapi' },
          ],
        }));
        return;
      }

      if (req.method === 'POST' && url === '/v1/chat/completions') {
        const wantsTools = Array.isArray(body.tools) && body.tools.length > 0;
        const wantsUsage = body.stream_options?.include_usage === true;

        if (!body.stream) {
          // Non-stream envelope.
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({
            id: 'chatcmpl-mock', object: 'chat.completion', created: 1,
            model: body.model,
            choices: [{
              index: 0,
              message: { role: 'assistant', content: 'hello' },
              finish_reason: 'stop',
            }],
            usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
          }));
          return;
        }

        // Streaming.
        res.writeHead(200, {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        });
        if (wantsTools) {
          // Split tool_call across frames to exercise reassembly.
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: { role: 'assistant', tool_calls: [{ index: 0, id: 'call_mock', type: 'function', function: { name: 'get_weather', arguments: '' } }] }, finish_reason: null }] }));
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] }, finish_reason: null }] }));
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"Paris"}' } }] }, finish_reason: null }] }));
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }));
        } else {
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: { role: 'assistant', content: 'one ' }, finish_reason: null }] }));
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: { content: 'two three' }, finish_reason: null }] }));
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
        }
        if (wantsUsage) {
          res.write(sseChunk({ object: 'chat.completion.chunk', choices: [], usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 } }));
        }
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }

      res.writeHead(404, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'not found' } }));
    });
  });
  return new Promise(resolve => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, base: `http://127.0.0.1:${port}` });
    });
  });
}

// ── main ───────────────────────────────────────────────────────────
async function main() {
  let base;
  let mock = null;
  if (REAL) {
    base = (process.env.BASE_URL || 'http://127.0.0.1:3003').replace(/\/+$/, '');
    console.log(`Cline compat verify — REAL mode against ${base} (model=${MODEL})\n`);
  } else {
    mock = await startMock();
    base = mock.base;
    console.log(`Cline compat verify — OFFLINE mock mode (proves the contract assertions)\n`);
  }

  try {
    await runSuite(base);
  } finally {
    if (mock) mock.server.close();
  }

  const failed = results.filter(r => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  if (failed.length) {
    console.log(`\x1b[31m${failed.length} FAILED:\x1b[0m ${failed.map(f => f.name).join('; ')}`);
    process.exit(1);
  }
  console.log('\x1b[32mAll Cline contract checks passed.\x1b[0m');
}

main().catch(err => {
  console.error('verifier crashed:', err);
  process.exit(1);
});

