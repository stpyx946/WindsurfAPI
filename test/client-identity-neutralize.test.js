// Devin's upstream backend hard-rejects (529 overloaded_error / internal error)
// any request whose system prompt announces "You are Claude Code, Anthropic's
// official CLI for Claude." — a client fingerprint / anti-competitor gate,
// CONFIRMED 2026-07-08 by ablation (one-word flip toggles 529↔200, all models).
// neutralizeClientIdentity rewrites only that self-ID line so the request serves.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { neutralizeClientIdentity } from '../src/handlers/messages.js';

afterEach(() => { delete process.env.WINDSURFAPI_NEUTRALIZE_CLIENT_ID; });

describe('neutralizeClientIdentity', () => {
  it('rewrites the exact Claude Code self-identification', () => {
    const out = neutralizeClientIdentity("You are Claude Code, Anthropic's official CLI for Claude.");
    assert.ok(!/Claude Code/.test(out), 'no Claude Code');
    assert.ok(!/Anthropic/.test(out), 'no Anthropic');
    assert.match(out, /AI coding assistant/);
  });

  it('handles a curly/straight apostrophe and trailing text', () => {
    const src = "You are Claude Code, Anthropic’s official CLI for Claude. You help with tasks.";
    const out = neutralizeClientIdentity(src);
    assert.ok(!/Claude Code/.test(out));
    assert.match(out, /You help with tasks\./, 'user instruction preserved');
  });

  it('neutralizes the phrase even without the leading "You are"', () => {
    const out = neutralizeClientIdentity("Note: Claude Code, Anthropic's official CLI for Claude, is running.");
    assert.ok(!/official CLI for Claude/.test(out));
  });

  it('leaves unrelated mentions of the words alone (only the ID phrase is rewritten)', () => {
    // Bare mentions are safe (ablation: standalone "Anthropic"/"Claude Code" → 200),
    // so we must NOT scrub general text — only the exact self-ID phrasing.
    const src = 'Use the Anthropic SDK. The Claude Code style guide applies.';
    const out = neutralizeClientIdentity(src);
    assert.equal(out, src, 'unrelated text untouched');
  });

  it('is a no-op on empty/undefined', () => {
    assert.equal(neutralizeClientIdentity(''), '');
    assert.equal(neutralizeClientIdentity(undefined), undefined);
  });

  it('can be disabled via WINDSURFAPI_NEUTRALIZE_CLIENT_ID=0', () => {
    process.env.WINDSURFAPI_NEUTRALIZE_CLIENT_ID = '0';
    const src = "You are Claude Code, Anthropic's official CLI for Claude.";
    assert.equal(neutralizeClientIdentity(src), src, 'opt-out leaves it verbatim');
  });
});
