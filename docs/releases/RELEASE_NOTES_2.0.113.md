## v2.0.113 - LSP restart safety and native trace persistence

This release closes the next LSP concurrency gap and makes native bridge protocol smoke runs easier to audit on Docker deployments.

### LSP scheduling

- LS process exit handling is now process-owned: a late exit from an old restarted LS can no longer delete a newer same-key pool entry.
- Intentional shutdown tracking is per process instead of per key, avoiding false "intentional" classification if a replacement LS exits unexpectedly.
- Scheduled probes now take a lightweight maintenance reservation and skip accounts/LS instances that are already serving production traffic.
- Production routing avoids accounts whose LS is under maintenance, including sticky and strict-reuse paths.
- `LS_MAX_INSTANCES=1` now disables default LS prewarm even when `LS_PREWARM_DEFAULT` is unset, preserving the only slot for lazy demand instead of pinning it to non-evictable default.

### Native bridge trace

- Docker deployments now default `WINDSURFAPI_PROTO_TRACE_DIR` to `/data/proto-trace`, so enabled protobuf traces survive container recreates.
- `native-bridge-smoke.mjs` supports `NATIVE_BRIDGE_SMOKE_NO_EXIT_ON_FAILURE=1`, allowing smoke scripts to print protobuf summaries even when a tool scenario fails.

### Verification

- `node --test test/*.test.js` -> 994/994 passing.
