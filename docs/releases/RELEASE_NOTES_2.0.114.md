## v2.0.114 - Native bridge list_directory reverse mapping

This release tightens the native bridge experiment based on live protobuf traces from the Docker deployment.

### Native bridge

- `Glob` reverse lookup now follows allowlist overrides such as `find:list_dir`, so a Cascade `list_directory` trajectory step can be surfaced back to the caller as a `Glob` tool call.
- `list_dir` and `list_directory` are included in the default native bridge tool scope for explicit gray-gate experiments.
- `Glob` argument reconstruction now accepts `directory_path_uri` from `list_directory` steps and emits a valid broad pattern instead of dropping the tool call.

### Verification

- `node --test test/native-tool-routing.test.js` -> 15/15 passing.
- `node --test test/*.test.js` -> 995/995 passing.
