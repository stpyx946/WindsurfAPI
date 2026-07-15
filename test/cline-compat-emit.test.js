import { afterEach, describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  toChatCompletion,
  streamChatCompletion,
  __setStreamChatForTest,
} from '../src/devin-connect-openai.js';
import { getClineCompatStats, resetClineCompatStats } from '../src/handlers/cline-compat.js';

afterEach(() => __setStreamChatForTest(null));
beforeEach(() => resetClineCompatStats());

function fakeStream(events) {
  return async function* () { for (const ev of events) yield ev; };
}

// A parameterless tool call: the model emits an EMPTY arguments string. This is
// exactly what Claude does for tools with no parameters, and what
// @ai-sdk/openai-compatible (Cline) silently drops (vercel/ai#6687).
// Native tool calls arrive on the finish event as ev.toolCalls, each
// {id, name, arguments} where arguments is the raw JSON string off the wire.
// Empty string: legacy `|| '{}'` already coalesced this (falsy), so compat adds
// nothing NEW here — the client saw '{}' either way, no repair is counted.
const EMPTY_ARG_TOOLCALL = [
  { type: 'finish', reason: 'tool_calls', usage: { prompt_tokens: 8, completion_tokens: 2, total_tokens: 10 },
    toolCalls: [{ id: 'call_abc', name: 'list_files', arguments: '' }] },
];

// Whitespace / malformed JSON is TRUTHY, so legacy `|| '{}'` passes it straight
// through and @ai-sdk drops the tool call. This is the case only the compat
// layer rescues — and the one that must be COUNTED as a repair.
const WHITESPACE_ARG_TOOLCALL = [
  { type: 'finish', reason: 'tool_calls', usage: null,
    toolCalls: [{ id: 'call_ws', name: 'list_files', arguments: '   ' }] },
];
const MALFORMED_ARG_TOOLCALL = [
  { type: 'finish', reason: 'tool_calls', usage: null,
    toolCalls: [{ id: 'call_bad', name: 'do_thing', arguments: '{"a":' }] },
];

describe('Cline compat emit — non-stream (DEVIN_CONNECT path)', () => {
  it('WITHOUT compat: empty arguments stay "" (byte-identical legacy behavior)', async () => {
    __setStreamChatForTest(fakeStream(EMPTY_ARG_TOOLCALL));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] });
    const tc = body.choices[0].message.tool_calls[0];
    // Legacy coalesced `argumentsJson || tc.arguments || '{}'`: '' is falsy so
    // it already became '{}' historically. Assert the historical value holds.
    assert.equal(tc.function.arguments, '{}');
    assert.equal(getClineCompatStats().argRepairs, 0, 'no repair counted when compat off');
  });

  it('WITH compat: empty arguments still "{}" but NOT counted (legacy already coalesced falsy)', async () => {
    __setStreamChatForTest(fakeStream(EMPTY_ARG_TOOLCALL));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] }, { clineCompat: true });
    assert.equal(body.choices[0].message.tool_calls[0].function.arguments, '{}');
    assert.equal(getClineCompatStats().argRepairs, 0, 'empty string was already handled by legacy || {}');
  });

  it('WITH compat: WHITESPACE arguments rescued to "{}" and counted (legacy would leak "   ")', async () => {
    __setStreamChatForTest(fakeStream(WHITESPACE_ARG_TOOLCALL));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] }, { clineCompat: true });
    assert.equal(body.choices[0].message.tool_calls[0].function.arguments, '{}');
    assert.equal(getClineCompatStats().argRepairs, 1, 'whitespace arg is a real repair');
  });

  it('WITHOUT compat: whitespace arguments leak through as-is (proves the gap exists)', async () => {
    __setStreamChatForTest(fakeStream(WHITESPACE_ARG_TOOLCALL));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] });
    // Legacy: '   ' is truthy → passes through unrepaired. @ai-sdk drops it.
    assert.equal(body.choices[0].message.tool_calls[0].function.arguments, '   ');
    assert.equal(getClineCompatStats().argRepairs, 0);
  });

  it('WITH compat: MALFORMED JSON arguments rescued to "{}" and counted', async () => {
    __setStreamChatForTest(fakeStream(MALFORMED_ARG_TOOLCALL));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] }, { clineCompat: true });
    assert.equal(body.choices[0].message.tool_calls[0].function.arguments, '{}');
    assert.equal(getClineCompatStats().argRepairs, 1);
  });

  it('WITH compat: already-valid JSON arguments pass through untouched, no repair', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'finish', reason: 'tool_calls', usage: null,
        toolCalls: [{ id: 'c1', name: 'get_weather', arguments: '{"city":"Paris"}' }] },
    ]));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] }, { clineCompat: true });
    assert.equal(body.choices[0].message.tool_calls[0].function.arguments, '{"city":"Paris"}');
    assert.equal(getClineCompatStats().argRepairs, 0, 'valid JSON is not a repair');
  });
});

describe('Cline compat emit — stream (DEVIN_CONNECT path)', () => {
  function collectStream(events, opts) {
    __setStreamChatForTest(fakeStream(events));
    const frames = [];
    const send = (obj) => frames.push(obj);
    return streamChatCompletion({ model: 'swe-1-6-slow', messages: [] }, send, opts).then(() => frames);
  }

  it('WITH compat: streamed tool_call with whitespace args normalized to "{}" and counted', async () => {
    const frames = await collectStream(WHITESPACE_ARG_TOOLCALL, { clineCompat: true });
    const toolFrame = frames.find(f => f.choices?.[0]?.delta?.tool_calls);
    assert.ok(toolFrame, 'a tool_calls delta frame was emitted');
    assert.equal(toolFrame.choices[0].delta.tool_calls[0].function.arguments, '{}');
    assert.equal(getClineCompatStats().argRepairs, 1);
  });

  it('WITHOUT compat: streamed tool_call args stay legacy "{}"', async () => {
    const frames = await collectStream(EMPTY_ARG_TOOLCALL, { clineCompat: false });
    const toolFrame = frames.find(f => f.choices?.[0]?.delta?.tool_calls);
    assert.ok(toolFrame);
    assert.equal(toolFrame.choices[0].delta.tool_calls[0].function.arguments, '{}');
    assert.equal(getClineCompatStats().argRepairs, 0);
  });
});
