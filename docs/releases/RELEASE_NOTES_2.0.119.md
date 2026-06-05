## v2.0.119 - native bridge protocol lab + LS-budgeted smoke

This release keeps native bridge production behavior opt-in, but makes the
next protocol pass safer and more measurable:

- `scripts/native-bridge-smoke.mjs` now refuses to run by default when the LS
  pool is busy, pending, under maintenance, or blocked by the memory guard.
  Set `NATIVE_BRIDGE_SMOKE_LS_BUDGET=0` only for an explicitly isolated test.
- Added `WINDSURFAPI_NATIVE_TOOL_BRIDGE_CONFIG_RAW` for matrix-only protobuf
  subconfig injection, e.g. `read_file:<hex>;grep_v2:base64:<payload>`.
  This is default-off and does not enable any native tool by itself.
- Proto trace now records compact child-field summaries for non-oneof
  `CortexTrajectoryStep` message fields, so future matrices can distinguish
  actual native bodies from planner-response wrappers.
- Added `docs/native-bridge-protocol-notes.md` with confirmed
  `CascadeToolConfig`, `FindToolConfig`, and `ViewFileToolConfig` fields plus
  unresolved `GrepV2ToolConfig` / full Glob gaps.

Validation:

- `node --test test/*.test.js` passes locally.

