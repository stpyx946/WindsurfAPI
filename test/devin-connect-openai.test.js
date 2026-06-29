import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  toChatCompletion,
  streamChatCompletion,
  __setStreamChatForTest,
} from '../src/devin-connect-openai.js';

afterEach(() => __setStreamChatForTest(null));

// Build a fake streamChat that yields a scripted event sequence.
function fakeStream(events) {
  return async function* () {
    for (const ev of events) yield ev;
  };
}

const SAMPLE = [
  { type: 'reasoning', text: 'let me think ' },
  { type: 'reasoning', text: 'about it.' },
  { type: 'content', text: 'The answer ' },
  { type: 'content', text: 'is 42.' },
  { type: 'finish', reason: 'stop', usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } },
];

describe('toChatCompletion (non-stream)', () => {
  it('assembles a chat.completion with separated content and reasoning', async () => {
    __setStreamChatForTest(fakeStream(SAMPLE));
    const { status, body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] });
    assert.equal(status, 200);
    assert.equal(body.object, 'chat.completion');
    assert.equal(body.model, 'swe-1-6-slow');
    const msg = body.choices[0].message;
    assert.equal(msg.role, 'assistant');
    assert.equal(msg.content, 'The answer is 42.');
    assert.equal(msg.reasoning_content, 'let me think about it.');
    assert.equal(body.choices[0].finish_reason, 'stop');
    assert.deepEqual(body.usage, { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 });
  });

  it('echoes displayModel over the request model when given', async () => {
    __setStreamChatForTest(fakeStream(SAMPLE));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] }, { displayModel: 'claude-sonnet-4-6' });
    assert.equal(body.model, 'claude-sonnet-4-6');
  });

  it('emits content:"" (never undefined) when the model returns no answer text', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'reasoning', text: 'hmm' },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { body } = await toChatCompletion({ model: 'm', messages: [] });
    assert.equal(body.choices[0].message.content, '');
    assert.equal(body.choices[0].message.reasoning_content, 'hmm');
  });

  it('omits usage when the upstream gave none', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: 'hi' },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { body } = await toChatCompletion({ model: 'm', messages: [] });
    assert.equal('usage' in body, false);
  });

  it('uses a stable id/created when supplied', async () => {
    __setStreamChatForTest(fakeStream(SAMPLE));
    const { body } = await toChatCompletion({ model: 'm', messages: [] }, { id: 'chatcmpl-fixed', created: 123 });
    assert.equal(body.id, 'chatcmpl-fixed');
    assert.equal(body.created, 123);
  });

  it('retries a transient failure then succeeds (no token duplication)', async () => {
    let calls = 0;
    __setStreamChatForTest(async function* () {
      calls++;
      if (calls === 1) {
        // fail AFTER yielding a partial — the retry must discard it cleanly.
        yield { type: 'content', text: 'PARTIAL' };
        throw Object.assign(new Error('reset'), { code: 'ECONNRESET' });
      }
      yield { type: 'content', text: 'clean answer' };
      yield { type: 'finish', reason: 'stop', usage: null };
    });
    const { body } = await toChatCompletion({ model: 'm', messages: [] }, { retryBaseMs: 1 });
    assert.equal(calls, 2);
    assert.equal(body.choices[0].message.content, 'clean answer'); // no leading PARTIAL
  });

  it('does not retry a terminal MODEL_BLOCKED error', async () => {
    let calls = 0;
    __setStreamChatForTest(async function* () {
      calls++;
      throw Object.assign(new Error('/upgrade required'), { code: 'MODEL_BLOCKED' });
    });
    await assert.rejects(
      toChatCompletion({ model: 'm', messages: [] }, { retryBaseMs: 1 }),
      /upgrade/,
    );
    assert.equal(calls, 1); // one attempt, no retry
  });

  it('gives up after maxRetries on a persistent transient error', async () => {
    let calls = 0;
    __setStreamChatForTest(async function* () {
      calls++;
      throw Object.assign(new Error('down'), { code: 'ETIMEDOUT' });
    });
    await assert.rejects(
      toChatCompletion({ model: 'm', messages: [] }, { maxRetries: 2, retryBaseMs: 1 }),
      /down/,
    );
    assert.equal(calls, 3); // initial + 2 retries
  });
});

describe('streamChatCompletion (SSE)', () => {
  function collectSend() {
    const frames = [];
    return { send: (d) => frames.push(d), frames };
  }

  it('emits role-prime, reasoning, content, finish, and usage chunks in order', async () => {
    __setStreamChatForTest(fakeStream(SAMPLE));
    const { send, frames } = collectSend();
    const result = await streamChatCompletion({ model: 'swe-1-6-slow', messages: [] }, send, { id: 'x', created: 1 });

    // 1. role-prime
    assert.deepEqual(frames[0].choices[0].delta, { role: 'assistant', content: '' });
    // every chunk carries the chat.completion.chunk envelope
    for (const f of frames) {
      assert.equal(f.object, 'chat.completion.chunk');
      assert.equal(f.id, 'x');
      assert.equal(f.created, 1);
    }
    // reasoning chunks precede content chunks
    const kinds = frames.map(f => {
      const d = f.choices[0]?.delta;
      if (d?.reasoning_content != null) return 'reasoning';
      if (d?.content && f.choices[0].finish_reason == null && !('role' in d)) return 'content';
      if (f.choices[0]?.finish_reason) return 'finish';
      if (f.choices.length === 0) return 'usage';
      return 'role';
    });
    assert.equal(kinds[0], 'role');
    const firstReasoning = kinds.indexOf('reasoning');
    const firstContent = kinds.indexOf('content');
    assert.ok(firstReasoning > 0 && firstReasoning < firstContent, 'reasoning before content');
    assert.equal(kinds.at(-2), 'finish');
    assert.equal(kinds.at(-1), 'usage');

    // finish chunk shape
    const finish = frames.find(f => f.choices[0]?.finish_reason === 'stop');
    assert.deepEqual(finish.choices[0].delta, {});
    // usage chunk shape
    const usageFrame = frames.find(f => f.choices.length === 0);
    assert.deepEqual(usageFrame.usage, { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 });

    // returns the assembled result for caching
    assert.equal(result.content, 'The answer is 42.');
    assert.equal(result.reasoning, 'let me think about it.');
    assert.equal(result.finish_reason, 'stop');
  });

  it('does not emit a usage frame when upstream had no usage', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: 'hi' },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { send, frames } = collectSend();
    await streamChatCompletion({ model: 'm', messages: [] }, send);
    assert.equal(frames.some(f => f.choices.length === 0), false);
  });

  it('streams each content delta as its own chunk (verbatim, not coalesced)', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: 'a' },
      { type: 'content', text: 'b' },
      { type: 'content', text: 'c' },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { send, frames } = collectSend();
    await streamChatCompletion({ model: 'm', messages: [] }, send);
    const contentDeltas = frames
      .map(f => f.choices[0]?.delta?.content)
      .filter(c => c != null && c !== '');
    assert.deepEqual(contentDeltas, ['a', 'b', 'c']);
  });
});

// Tool emulation: the connect models have no native function-calling, so the
// adapter parses <tool_call>{...}</tool_call> markup out of the answer (the
// same machinery the Cascade path uses) and surfaces OpenAI tool_calls.
describe('toChatCompletion tool emulation', () => {
  const TOOL_ANSWER = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>';

  it('extracts a tool_call and sets finish_reason=tool_calls, content=null', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: TOOL_ANSWER },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { body } = await toChatCompletion({ model: 'swe-1-6-slow', messages: [] }, { emulateTools: true });
    const msg = body.choices[0].message;
    assert.equal(body.choices[0].finish_reason, 'tool_calls');
    assert.equal(msg.content, null);
    assert.equal(msg.tool_calls.length, 1);
    assert.equal(msg.tool_calls[0].type, 'function');
    assert.equal(msg.tool_calls[0].function.name, 'get_weather');
    assert.deepEqual(JSON.parse(msg.tool_calls[0].function.arguments), { city: 'Paris' });
    assert.ok(msg.tool_calls[0].id, 'has an id');
  });

  it('leaves a plain answer untouched when emulateTools is on but no markup present', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: 'just a normal answer' },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { body } = await toChatCompletion({ model: 'm', messages: [] }, { emulateTools: true });
    assert.equal(body.choices[0].finish_reason, 'stop');
    assert.equal(body.choices[0].message.content, 'just a normal answer');
    assert.equal('tool_calls' in body.choices[0].message, false);
  });

  it('does NOT parse tool markup when emulateTools is off (passes through as text)', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: TOOL_ANSWER },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { body } = await toChatCompletion({ model: 'm', messages: [] }); // no emulateTools
    assert.equal(body.choices[0].finish_reason, 'stop');
    assert.equal(body.choices[0].message.content, TOOL_ANSWER);
  });
});

describe('streamChatCompletion tool emulation', () => {
  function collectSend() {
    const frames = [];
    return { send: (d) => frames.push(d), frames };
  }

  it('emits a tool_calls delta and finishes with finish_reason=tool_calls', async () => {
    // Split the markup across deltas to exercise the streaming parser's
    // cross-chunk buffering.
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: '<tool_call>{"name": "search", ' },
      { type: 'content', text: '"arguments": {"q": "cats"}}</tool_call>' },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { send, frames } = collectSend();
    const result = await streamChatCompletion({ model: 'swe-1-6-slow', messages: [] }, send, { emulateTools: true });

    const toolFrame = frames.find(f => f.choices[0]?.delta?.tool_calls);
    assert.ok(toolFrame, 'a tool_calls delta was emitted');
    const tc = toolFrame.choices[0].delta.tool_calls[0];
    assert.equal(tc.index, 0);
    assert.equal(tc.type, 'function');
    assert.equal(tc.function.name, 'search');
    assert.deepEqual(JSON.parse(tc.function.arguments), { q: 'cats' });

    const finish = frames.find(f => f.choices[0]?.finish_reason);
    assert.equal(finish.choices[0].finish_reason, 'tool_calls');
    assert.equal(result.finish_reason, 'tool_calls');
    assert.equal(result.toolCalls.length, 1);

    // The tool markup must NOT leak into content deltas.
    const leaked = frames.map(f => f.choices[0]?.delta?.content || '').join('');
    assert.equal(leaked.includes('<tool_call>'), false, 'markup leaked to content');
  });

  it('streams normal text untouched when emulateTools is on but no tool markup', async () => {
    __setStreamChatForTest(fakeStream([
      { type: 'content', text: 'hello ' },
      { type: 'content', text: 'world' },
      { type: 'finish', reason: 'stop', usage: null },
    ]));
    const { send, frames } = collectSend();
    const result = await streamChatCompletion({ model: 'm', messages: [] }, send, { emulateTools: true });
    const text = frames.map(f => f.choices[0]?.delta?.content || '').join('');
    assert.equal(text, 'hello world');
    assert.equal(result.finish_reason, 'stop');
    assert.equal(result.toolCalls.length, 0);
  });
});
