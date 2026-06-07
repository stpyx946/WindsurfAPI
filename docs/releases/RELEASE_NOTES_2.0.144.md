## v2.0.144 - WebFetch completed document handling

This release fixes the lab-gated native WebFetch path when the language server
has already executed `read_url_content` and returned a real `web_document`.

- Completed `read_url_content.web_document` steps are no longer surfaced to
  OpenAI clients as dead tool-call proposals. The proxy now preserves Cascade's
  final assistant text with `finish_reason="stop"`.
- If Cascade returns a completed WebFetch document but no final text, the proxy
  falls back to the fetched document body as assistant content.
- Pending WebFetch permission steps without a document keep the existing
  proposal/wait behavior, and Bash/Read/Grep native proposal semantics are
  unchanged.
- Proto trace classification now checks `completed_web_document` before
  `pending_permission`, so steps that contain both `web_document` and a
  requested-interaction echo are classified as completed.
