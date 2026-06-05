#!/usr/bin/env node

const baseUrl = (process.env.BASE_URL || process.env.WINDSURFAPI_BASE_URL || 'http://127.0.0.1:3003').replace(/\/+$/, '');
const apiKey = process.env.API_KEY || process.env.WINDSURFAPI_API_KEY || '';
const model = process.env.MODEL || process.env.WINDSURFAPI_SMOKE_MODEL || 'claude-sonnet-4.6';
const marker = `NATIVE_BRIDGE_SMOKE_${Date.now().toString(36)}`;

if (!apiKey) {
  console.error('API_KEY is required. Run the server with WINDSURFAPI_NATIVE_TOOL_BRIDGE=all_mapped before this smoke.');
  process.exit(2);
}

const bashTool = {
  type: 'function',
  function: {
    name: 'Bash',
    description: 'Run a shell command in the configured workspace.',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string' },
        cwd: { type: 'string' },
      },
      required: ['command'],
      additionalProperties: false,
    },
  },
};

function requestBody(stream) {
  return {
    model,
    stream,
    messages: [
      {
        role: 'user',
        content: `Use the Bash tool exactly once with command: printf ${marker}. Do not answer in prose.`,
      },
    ],
    tools: [bashTool],
    tool_choice: { type: 'function', function: { name: 'Bash' } },
    max_tokens: 512,
  };
}

async function post(body) {
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${apiKey}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, text };
}

function assertNoNativeXml(text, label) {
  if (/<\/?function_calls\b|<invoke\b/i.test(text)) {
    throw new Error(`${label}: provider-native function XML leaked to the client`);
  }
}

function parseSse(text) {
  const frames = [];
  for (const frame of text.split('\n\n')) {
    const line = frame.split('\n').find(l => l.startsWith('data: '));
    if (!line) continue;
    const payload = line.slice(6);
    if (payload === '[DONE]') continue;
    try { frames.push(JSON.parse(payload)); } catch {}
  }
  return frames;
}

function collectStreamToolCalls(text) {
  return parseSse(text).flatMap(f => (f.choices || [])
    .flatMap(choice => choice.delta?.tool_calls || []));
}

async function runNonStream() {
  const res = await post(requestBody(false));
  if (res.status !== 200) throw new Error(`non-stream HTTP ${res.status}: ${res.text.slice(0, 500)}`);
  assertNoNativeXml(res.text, 'non-stream');
  const json = JSON.parse(res.text);
  const calls = json.choices?.[0]?.message?.tool_calls || [];
  if (!calls.length) throw new Error(`non-stream produced no tool_calls: ${res.text.slice(0, 500)}`);
  return { toolCalls: calls.length, names: calls.map(c => c.function?.name || c.name || '') };
}

async function runStream() {
  const res = await post(requestBody(true));
  if (res.status !== 200) throw new Error(`stream HTTP ${res.status}: ${res.text.slice(0, 500)}`);
  assertNoNativeXml(res.text, 'stream');
  const calls = collectStreamToolCalls(res.text);
  if (!calls.length) throw new Error(`stream produced no tool_calls: ${res.text.slice(0, 500)}`);
  return { toolCalls: calls.length, names: calls.map(c => c.function?.name || c.name || '') };
}

const summary = {
  baseUrl,
  model,
  marker,
  nonStream: await runNonStream(),
};
if (process.env.NATIVE_BRIDGE_SMOKE_STREAM !== '0') {
  summary.stream = await runStream();
}

console.log(JSON.stringify({ ok: true, ...summary }, null, 2));
