## v2.0.105 - native bridge real-smoke guardrails

This release keeps production behavior unchanged and tightens the native bridge
field-test workflow.

### Smoke testing

- `scripts/native-bridge-smoke.mjs` now applies a per-request timeout via
  `NATIVE_BRIDGE_SMOKE_TIMEOUT_MS` (default 120s).
- Stream smoke can stop as soon as the expected `tool_calls` delta is observed,
  which separates "tool call reached the client" from "Cascade kept executing
  the built-in IDE tool until the HTTP request stalled".
- `NATIVE_BRIDGE_SMOKE_NON_STREAM=0` can be used for stream-only canaries when
  testing native Cascade tools that may continue running upstream.

### Field note

Real VPS smoke on `claude-4.5-haiku` showed the gray gate enabling correctly
(`native bridge ON` with `run_command` allowlist), but the upstream built-in
`run_command` path stalled past 240s. Treat the current bridge as experimental:
the next protocol step is to identify or implement a proposal-only native path
that emits tool calls without letting Cascade execute local IDE tools remotely.
