## v2.0.102 - update.sh LS install path alignment

This release integrates the useful part of PR #192 by @andya1lan without
pulling in the stale branch history.

### Upgrade path

- `update.sh` now routes LS binary updates through `install-ls.sh`, so one-click
  updates use the same source chain as the dashboard LS updater:
  WindsurfAPI release -> Windsurf desktop LS release -> Exafunction fallback.
- `update.sh` no longer hardcodes the old
  `language_server_linux_x64` WindsurfAPI release URL.
- LS replacement now keeps the `install-ls.sh` atomic rename behavior, avoiding
  the file-busy class of live-update failures.
- If the LS download chain is temporarily unreachable but an existing LS binary
  is present, `update.sh` keeps that binary and continues the code update /
  restart. It only aborts when no usable LS binary exists.

### macOS compatibility

- `.env` parsing in `update.sh` no longer depends on GNU `grep -P`, which is not
  available in macOS BSD grep.
- The parser accepts `LS_BINARY_PATH=...`, `export LS_BINARY_PATH=...`, optional
  spaces around `=`, quoted values, and trailing comments on unquoted values.

### Credits

- Thanks to @andya1lan for PR #192 and the focused regression test direction.
