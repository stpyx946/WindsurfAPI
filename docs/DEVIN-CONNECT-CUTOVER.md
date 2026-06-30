# DEVIN_CONNECT Production Cutover Runbook

**Created:** 2026-06-30 (eve of Cascade retirement, 2026-07-01)
**Audience:** operator deploying WindsurfAPI to homecloud after Cascade is gone.
**Status:** deploy-ready. Code is verified locally (1492 tests green) but NOT yet
pushed/deployed — production still runs old code that 503s on Devin routes.

---

## 0. The one thing that matters most

There are **two** Devin switches and they are not interchangeable:

| Switch | Routes to | Needs a binary? | Use on homecloud? |
|---|---|---|---|
| **`DEVIN_CONNECT=1`** | Devin **cloud** GetChatMessage over pure HTTP, riding the account pool | **No** | **YES — this one** |
| `DEVIN_ONLY=1` | the **local `devin` CLI** subprocess | Yes (`devin` binary) | No — homecloud has no binary → 503 per request |

Homecloud has neither the `language_server` binary nor the `devin` CLI, so the
cutover switch is **`DEVIN_CONNECT=1`**. Setting `DEVIN_ONLY=1` here would make
every request 503.

---

## 1. Minimal cutover config

On homecloud, in the service `.env` (workdir `/home/dwgx_user/WindsurfAPI`,
loopback `127.0.0.1:3003` per memory `homecloud-deploy`):

```sh
DEVIN_CONNECT=1
API_KEY=sk-dwgxnbnb        # existing downstream proxy key — unchanged
# accounts.json already holds the free session token(s) — nothing else required
```

That is the whole minimum. The pool supplies tokens; free-tier accounts resolve
only `swe-1-6-slow`. Any other model name degrades to that free selector.

## 2. Recommended hardening adds (optional, all default-off)

```sh
DEVIN_CONNECT_LIVENESS_PROBE=1            # zero-billable dead-token detection sweep
DEVIN_CONNECT_AUTO_RELOGIN=1              # self-heal a dead token via Auth1 re-login
DEVIN_CONNECT_CRED_KEY=<32+ char secret>  # REQUIRED for auto-relogin to do anything
```

Auto-relogin only works for accounts that have a **stored password**. An account
added by raw token (the current pooled free account, id `70da7667`) has none, so
it can fail over to other pool members but cannot self-heal its own token. To
enable self-heal, re-add the account through the email/password login path with
`DEVIN_CONNECT_CRED_KEY` set — the password is then auto-stored encrypted
(`src/auth.js:514-515`). See §5.

## 3. Deploy steps (operator runs on homecloud)

```sh
cd /home/dwgx_user/WindsurfAPI
git pull                                  # pulls the DEVIN_CONNECT commits
# edit .env: add DEVIN_CONNECT=1 (+ optional hardening from §2)
sudo systemctl restart windsurfapi
sudo systemctl status windsurfapi --no-pager   # confirm active (running)
```

No `npm install` needed unless `package.json` changed (it didn't for this work).

## 4. Post-deploy verification

```sh
# zero-billable preflight first (no model calls):
API_KEY=sk-dwgxnbnb BASE_URL=http://127.0.0.1:3003 \
  CONNECT_SMOKE_REAL_CALLS=0 npm run smoke:devin-connect

# then one real free-model call to confirm the chat path yields tokens:
API_KEY=sk-dwgxnbnb BASE_URL=http://127.0.0.1:3003 \
  npm run smoke:devin-connect
```

A green run proves: pool token in use, GetChatMessage reachable, free selector
resolves, and (if hardening enabled) credential store + recovery-config sane.

Quick manual sanity check:

```sh
curl -s http://127.0.0.1:3003/v1/chat/completions \
  -H "Authorization: Bearer sk-dwgxnbnb" -H "Content-Type: application/json" \
  -d '{"model":"swe-1-6-slow","messages":[{"role":"user","content":"ping"}]}' \
  | head -c 400
```

## 5. Enabling true self-heal for the free account (optional)

The pooled free account is token-only today. To give it self-heal:

1. Set `DEVIN_CONNECT_CRED_KEY` in `.env` and restart.
2. Remove the token-only account, re-add via the dashboard's email/password
   login (or the login API) — this stores the password encrypted.
3. Verify recovery: `npm run smoke:devin-connect` Stage 0c reports the
   credential store enabled and recovery-config sane.

## 6. Rollback

```sh
# fastest: drop the switch and restart (reverts to prior backend selection)
# edit .env: comment out DEVIN_CONNECT=1
sudo systemctl restart windsurfapi
# or revert the code:
git log --oneline -5 && git checkout <prior-sha> && sudo systemctl restart windsurfapi
```

`DEVIN_CONNECT` is a pure env flag with no migration — toggling it off is an
instant, total rollback.

## 7. What's NOT covered (known limits)

- **Paid models** (claude-*/gpt-*/gemini-*) need a paid entitlement on a pooled
  account — unverified, blocked on a paid token. Free deploy serves
  `swe-1-6-slow` only.
- **Vision/images**: gated off (`DEVIN_CONNECT_IMAGE_TAG=0`); leave unset.
- Remaining P1/P2 hardening (hung-stream absolute deadline, quota-vs-tier
  classification, streaming transient-5xx replay, observability counters) are
  filed as backlog — not blockers for a free-tier cutover.
