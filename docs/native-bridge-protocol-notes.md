# Native Bridge Protocol Notes

Status: reverse-engineering notes for the opt-in native bridge. Nothing here
is a default production enablement decision.

## Confirmed Tool Config Fields

`CascadeToolConfig`:

- `find` = field `5` (`FindToolConfig`)
- `run_command` = field `8` (`RunCommandToolConfig`)
- `view_file` = field `10` (`ViewFileToolConfig`)
- `list_dir` = field `19` (`ListDirToolConfig`)
- `tool_allowlist` = repeated field `32`
- `grep_v2` = field `33` (`GrepV2ToolConfig`)

Confirmed from LS binary protobuf struct tags and runtime trace.

`FindToolConfig`:

- `max_find_results` = field `1`
- `fd_path` = field `2`
- `enterprise_config` = field `7`

`ViewFileToolConfig`:

- `max_tokens_per_outline` = field `1`
- `max_doc_lines_fraction` = field `2` (`fixed32`)
- `allow_doc_outline` = field `4` (`optional bool`)
- `use_line_numbers_for_raw` = field `5` (`optional bool`)
- `use_prompt_prefix` = field `6` (`optional bool`)
- `allow_view_gitignore` = field `7` (`optional bool`)
- `split_outline_tool` = field `8` (`optional bool`)
- `max_total_outline_bytes` = field `9`
- `show_full_file_bytes` = field `10` (`optional bool`)
- `max_bytes_per_outline_item` = field `11`
- `enterprise_config` = field `12`
- `show_triggered_memories` = field `13` (`optional bool`)
- `max_lines_per_view` = field `14` (`optional bool/int-style oneof in Go tag`)
- `use_view_file_v2` = field `15` (`optional bool`)

`GrepV2ToolConfig`:

- Methods confirm `enterprise_config` and `allow_access_gitignore`.
- Binary tags show several `allow_access_gitignore` fields across related
  grep/view-code configs. The exact GrepV2 field number still needs an
  isolated descriptor dump or raw-config matrix confirmation before hardcoding.

`ListDirToolConfig`:

- Method confirms `enterprise_config`.
- No safe non-empty field is hardcoded yet.

## Runtime Step Caveat

`CortexTrajectoryStep.type` is not a reliable body-field number. Some traces
show `type=14` with payload on `field=19`, and `type=15` with `field=20`
planner response data. Keep parsing based on actual oneof/message fields and
trace unknown message-field children before promoting a new mapping.

## Experiment Hooks

`WINDSURFAPI_NATIVE_TOOL_BRIDGE_CONFIG_RAW` can inject exact protobuf bytes
for native tool subconfigs:

```text
read_file:<hex>;grep_v2:base64:<base64>;find:<hex>;list_dir:<hex>
```

The hook is default-off and exists only for matrix testing. Smoke must still
require native source plus argument validation; a raw subconfig that merely
causes natural-language or degraded `pattern:"*"` output is not a success.

## Next Matrix

- `Read/read_file`: test `ViewFileToolConfig` with `use_view_file_v2=true`
  (`field 15 = true`) plus, separately, `use_line_numbers_for_raw=true`
  (`field 5 = true`) and `use_prompt_prefix=true` (`field 6 = true`).
- `Grep/grep_v2`: test likely `allow_access_gitignore=true` candidates only
  after isolating the field number. Do not promote from method names alone.
- `Glob/find`: test `FindToolConfig.max_find_results` and `fd_path` only as
  diagnostics; full Glob requires returned arguments to preserve caller
  `pattern`, not just a `list_directory` fallback with `pattern:"*"`.
