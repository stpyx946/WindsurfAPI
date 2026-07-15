#!/usr/bin/env bash
# Cline connectivity smoke — 30-second self-check for a WindsurfAPI gateway
# against the OpenAI-Compatible contract Cline uses. Prints PASS/FAIL per probe.
#
#   BASE_URL="https://<gateway>/v1" API_KEY="<key>" MODEL="swe-1-6-slow" \
#     bash scripts/cline-smoke.sh
#
# NOTE: BASE_URL must include the /v1 suffix (same as the Cline setting).
# Requires: curl. Exit 0 = all probes passed, 1 = a probe failed.
set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:3003/v1}"
API_KEY="${API_KEY:-test}"
MODEL="${MODEL:-swe-1-6-slow}"
BASE_URL="${BASE_URL%/}"

pass=0; fail=0
ok()   { echo -e "  \033[32mPASS\033[0m  $1"; pass=$((pass+1)); }
bad()  { echo -e "  \033[31mFAIL\033[0m  $1"; fail=$((fail+1)); }

auth=(-H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json")

echo "Cline smoke → ${BASE_URL} (model=${MODEL})"

# 1. GET /v1/models
models="$(curl -s --max-time 30 "${auth[@]}" "${BASE_URL}/models")"
if echo "$models" | grep -q '"object"[[:space:]]*:[[:space:]]*"list"'; then
  n="$(echo "$models" | grep -o '"id"' | wc -l | tr -d ' ')"
  ok "GET /models (${n} entries)"
else
  bad "GET /models — no {object:list} (got: $(echo "$models" | head -c 160))"
fi

# 2. Non-stream chat
chat="$(curl -s --max-time 60 "${auth[@]}" "${BASE_URL}/chat/completions" -d "$(cat <<EOF
{"model":"${MODEL}","messages":[{"role":"user","content":"Say hi in one word."}],"stream":false}
EOF
)")"
if echo "$chat" | grep -q '"chat.completion"'; then
  ok "POST /chat/completions non-stream"
else
  bad "POST /chat/completions non-stream (got: $(echo "$chat" | head -c 200))"
fi

# 3. Streaming chat + tools → tool_calls + [DONE]
stream="$(curl -s -N --max-time 60 "${auth[@]}" "${BASE_URL}/chat/completions" -d "$(cat <<EOF
{"model":"${MODEL}","messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],"stream":true,"tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
EOF
)")"
if echo "$stream" | grep -q 'chat.completion.chunk' && echo "$stream" | grep -q '\[DONE\]'; then
  if echo "$stream" | grep -q 'tool_calls'; then
    ok "POST /chat/completions stream + tool_calls + [DONE]"
  else
    ok "POST /chat/completions stream + [DONE] (model chose not to call tool)"
  fi
else
  bad "POST /chat/completions stream (no chunk/[DONE]; got: $(echo "$stream" | head -c 200))"
fi

echo ""
echo "${pass} passed, ${fail} failed"
[ "$fail" -eq 0 ] || exit 1
