/**
 * DEVIN_CONNECT → OpenAI ChatCompletion adapter.
 *
 * Bridges the structured stream from devin-connect.js (content / reasoning /
 * finish / usage events) into the two OpenAI-compatible response shapes the rest
 * of the proxy already emits:
 *
 *   - non-stream: a single `chat.completion` object
 *   - stream:     a sequence of `chat.completion.chunk` SSE frames + [DONE]
 *
 * The output shape is matched field-for-field to the Cascade path in
 * handlers/chat.js (role-priming chunk, reasoning_content before content, a
 * finish_reason:'stop' chunk, then a usage-only chunk) so a client can't tell
 * which backend served the request. Reasoning maps to `reasoning_content`,
 * which OpenAI-style clients hide by default — see handlers/chat.js:737.
 *
 * Pure translation: no network of its own, it only consumes streamChat().
 */

import { randomUUID } from 'crypto';
import { streamChat as realStreamChat } from './devin-connect.js';

// streamChat is injectable so the adapter can be unit-tested without touching
// the network — mirrors the __set…ForTest convention in windsurf-api.js.
let streamChatImpl = realStreamChat;
export function __setStreamChatForTest(fn) {
  streamChatImpl = typeof fn === 'function' ? fn : realStreamChat;
}

const OBJECT_COMPLETION = 'chat.completion';
const OBJECT_CHUNK = 'chat.completion.chunk';

function newId() {
  return `chatcmpl-${randomUUID().replace(/-/g, '')}`;
}

function nowSeconds() {
  return Math.floor(Date.now() / 1000);
}

/**
 * Collect a full DEVIN_CONNECT completion and shape it as a non-streaming
 * `chat.completion` object.
 *
 * @param {object} params  forwarded to streamChat (messages, model, token, …)
 * @param {object} [opts]
 * @param {string} [opts.id]            response id (default chatcmpl-…)
 * @param {number} [opts.created]       unix seconds (default now)
 * @param {string} [opts.displayModel]  model name echoed back to the client
 * @returns {Promise<{status:number, body:object}>}
 */
export async function toChatCompletion(params, { id = newId(), created = nowSeconds(), displayModel } = {}) {
  const model = displayModel || params.model;
  let content = '';
  let reasoning = '';
  let finishReason = 'stop';
  let usage = null;

  for await (const ev of streamChatImpl(params)) {
    if (ev.type === 'content') content += ev.text;
    else if (ev.type === 'reasoning') reasoning += ev.text;
    else if (ev.type === 'finish') {
      if (ev.reason) finishReason = ev.reason;
      if (ev.usage) usage = ev.usage;
    }
  }

  // OpenAI convention: content is a string (may be empty), never undefined.
  const message = { role: 'assistant', content: content || '' };
  if (reasoning) message.reasoning_content = reasoning;

  const body = {
    id,
    object: OBJECT_COMPLETION,
    created,
    model,
    choices: [{ index: 0, message, finish_reason: finishReason }],
  };
  if (usage) body.usage = usage;
  return { status: 200, body };
}

/**
 * Stream a DEVIN_CONNECT completion as OpenAI `chat.completion.chunk` SSE
 * frames. `send` is the SSE writer used in handlers/chat.js — a function taking
 * a JS object that it JSON-encodes onto the `data:` line. This helper does NOT
 * write `data: [DONE]` or close the response; the caller owns the socket
 * lifecycle (heartbeat, unregister, res.end) exactly as the Cascade path does.
 *
 * Emission order mirrors the Cascade stream:
 *   1. role-priming chunk (delta {role, content:''})
 *   2. reasoning_content deltas as they arrive
 *   3. content deltas as they arrive
 *   4. finish chunk (delta {}, finish_reason)
 *   5. usage-only chunk (choices [], usage) when usage is known
 *
 * @returns {Promise<{content:string, reasoning:string, finish_reason:string, usage:object|null}>}
 *          the assembled result, so callers can cache it after streaming.
 */
export async function streamChatCompletion(params, send, { id = newId(), created = nowSeconds(), displayModel } = {}) {
  const model = displayModel || params.model;
  const base = { id, object: OBJECT_CHUNK, created, model };

  // 1. Prime the stream with the assistant role so clients open the message
  //    even before any token arrives.
  send({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });

  let content = '';
  let reasoning = '';
  let finishReason = 'stop';
  let usage = null;

  for await (const ev of streamChatImpl(params)) {
    if (ev.type === 'reasoning') {
      reasoning += ev.text;
      send({ ...base, choices: [{ index: 0, delta: { reasoning_content: ev.text }, finish_reason: null }] });
    } else if (ev.type === 'content') {
      content += ev.text;
      send({ ...base, choices: [{ index: 0, delta: { content: ev.text }, finish_reason: null }] });
    } else if (ev.type === 'finish') {
      if (ev.reason) finishReason = ev.reason;
      if (ev.usage) usage = ev.usage;
    }
  }

  // 4. Terminal finish chunk.
  send({ ...base, choices: [{ index: 0, delta: {}, finish_reason: finishReason }] });

  // 5. Usage-only chunk (OpenAI streams usage in a trailing choices:[] frame).
  if (usage) {
    send({ ...base, choices: [], usage });
  }

  return { content, reasoning, finish_reason: finishReason, usage };
}

export const __testing = { newId, nowSeconds, OBJECT_COMPLETION, OBJECT_CHUNK };
