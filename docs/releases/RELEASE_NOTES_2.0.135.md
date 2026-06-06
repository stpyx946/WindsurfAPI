# v2.0.135

- Added a lab-only WebFetch permission POC. When
  `WINDSURFAPI_NATIVE_TOOL_BRIDGE_WEBFETCH_AUTO_APPROVE=1` and
  `WINDSURFAPI_NATIVE_TOOL_BRIDGE_WEBFETCH_AUTO_APPROVE_ORIGINS` explicitly
  matches the requested origin or URL, the Cascade polling loop responds to
  `requested_interaction.read_url_content` through
  `HandleCascadeUserInteraction` with `ALLOW_ONCE`.
- Added protobuf builders/parsers for the official interaction path:
  `GetCascadeTrajectoryResponse.trajectory.trajectory_id`,
  `CortexTrajectoryStep.requested_interaction`, and
  `HandleCascadeUserInteractionRequest`.
- Added a redacted proto trace summary for
  `HandleCascadeUserInteraction` requests so canaries can prove which
  trajectory/step/action was approved without logging raw URLs or IDs.
- Pending WebFetch permission steps are no longer surfaced as completed native
  tool calls. The proxy waits for a real `web_document` or legacy result, which
  prevents an empty pending step from deduping away the later completed fetch.
- WebFetch remains outside the default native bridge allowlist. This release
  only gives protocol canaries a safe way to test whether LS can continue from
  the official permission prompt to a completed `read_url_content.web_document`
  step.

Verification:

- `node --check src/client.js`
- `node --check src/windsurf.js`
- `node --check src/proto-trace.js`
- `node --test test/proto-trace.test.js`
- `node --test test/v2070-issue-fixes.test.js`
- `node --test test/client-panel-retry.test.js`
- `npm test`
