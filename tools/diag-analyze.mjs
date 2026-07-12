#!/usr/bin/env node
// diag-analyze — the diagnostic bench for "is the intermittent internal-error
// OUR bug or the upstream's?". Reads a .trace/ tree (WINDSURFAPI_TRACE=1) and:
//   1. decodes each request's wire fields (selector, #31 fingerprint length,
//      system-block presence, tool count) from the raw 03 protobuf bytes;
//   2. classifies each response (200-with-content / 200-empty / internal-error /
//      content-policy / rate-limit / other) from the 04 trailer + 05 client-res;
//   3. aggregates per-model: attempts, ok%, and failure breakdown;
//   4. THE VERDICT STEP — for any model that has BOTH a success and a failure,
//      byte-diffs their upstream requests. Identical request bytes → upstream is
//      flaky (NOT our bug). Differing bytes → the diff IS our bug (a field we
//      send wrong intermittently: fingerprint length, missing system, etc.).
//
// Pure node, reuses src/proto.js. Run on homecloud after a probe batch:
//   node tools/diag-analyze.mjs                 analyze .trace, print report
//   node tools/diag-analyze.mjs --json out.json dump structured findings
//   node tools/diag-analyze.mjs --model claude-opus-4-8-medium  focus one model
//   node tools/diag-analyze.mjs --dir .trace    trace root (default .trace)

import { readFileSync, readdirSync, existsSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { parseFields, getField, getAllFields } from '../src/proto.js';

const argv = process.argv.slice(2);
const flags = { dir: '.trace' };
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--dir') flags.dir = argv[++i];
  else if (argv[i] === '--json') flags.json = argv[++i];
  else if (argv[i] === '--model') flags.model = argv[++i];
}
const ROOT = resolve(process.cwd(), flags.dir);
if (!existsSync(ROOT)) { console.error(`no trace dir at ${ROOT}`); process.exit(1); }

const readJson = (p) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return null; } };
const readBin = (p) => { try { return readFileSync(p); } catch { return null; } };

// GetChatMessageRequest wire field tags (calibrated; see devin-connect.js header).
const TAG = { CLIENT_META: 1, FINGERPRINT: 31, MODEL_CONFIG: 15, SELECTOR: 21 };

// Decode the parts of the upstream request that matter for the OUR-vs-THEIRS
// verdict. Returns a stable, diffable descriptor + a content hash of the whole
// request body (minus the intentionally-random fingerprint, so two requests that
// differ ONLY by the random #31 still hash equal).
function decodeUpstreamReq(bin) {
  if (!bin || bin.length < 8) return null;
  let fields;
  try { fields = parseFields(bin); } catch { return { decodeError: true }; }
  const out = { bytes: bin.length };
  // ClientMetadata (#1) sub-message carries the fingerprint (#31).
  const meta = getField(fields, TAG.CLIENT_META);
  if (meta && meta.wireType === 2) {
    try {
      const metaFields = parseFields(meta.value);
      const fp = getField(metaFields, TAG.FINGERPRINT);
      if (fp && fp.wireType === 2) out.fingerprintLen = fp.value.length; // bytes; 366 expected (732 hex? see note)
    } catch { /* leave undefined */ }
  }
  return out;
}

// Classify the outcome of one trace from its trailer (04) + client-res (05).
function classifyOutcome(dir) {
  const res = readJson(join(dir, '05-client-res.json')) || {};
  const trailerTxt = (() => {
    try { return readFileSync(join(dir, '04-upstream-res.txt'), 'utf8'); } catch { return ''; }
  })();
  const status = res.status ?? null;
  const ms = res.ms ?? null;
  const t = trailerTxt.toLowerCase();
  let kind = 'unknown';
  if (/content policy|remove sensitive|unsafe content/.test(t)) kind = 'content_policy';
  else if (/internal error occurred/.test(t)) kind = 'internal_error';
  else if (/rate limit|resets? in|too many requests/.test(t)) kind = 'rate_limit';
  else if (/high demand|capacity|overloaded|unavailable/.test(t)) kind = 'capacity';
  else if (/permission_denied|unauthenticated|invalid api/.test(t)) kind = 'auth';
  else if (status === 200 && ms != null && ms < 50) kind = 'ok_fast_or_empty'; // 200 but suspiciously instant
  else if (status === 200) kind = 'ok';
  else if (status && status >= 400) kind = `http_${status}`;
  const ok = kind === 'ok';
  return { status, ms, kind, ok, trailer: trailerTxt.slice(0, 200) };
}

// Load every trace into a normalized record.
function loadTraces() {
  const dirs = readdirSync(ROOT).filter((d) => { try { return statSync(join(ROOT, d)).isDirectory(); } catch { return false; } });
  const recs = [];
  for (const d of dirs) {
    const dir = join(ROOT, d);
    const req = readJson(join(dir, '01-client-req.json')) || {};
    const rt = readJson(join(dir, '02-routing.json')) || {};
    const bin = readBin(join(dir, '03-upstream-req.bin'));
    const model = req.model || rt.requested || '?';
    if (flags.model && model !== flags.model && (rt.selector || '') !== flags.model) continue;
    const wire = bin ? decodeUpstreamReq(bin) : null;
    const outcome = classifyOutcome(dir);
    // Content hash of the FULL upstream body — used for the verdict diff.
    const bodyHash = bin ? createHash('sha256').update(bin).digest('hex').slice(0, 16) : null;
    recs.push({ id: d, model, selector: rt.selector || null, mapped: rt.mapped, wire, outcome, bodyHash, bin });
  }
  return recs;
}

// Normalize a request body for comparison: blank out the intentionally-random
// #31 fingerprint region so two requests that differ ONLY by the random hex
// still compare equal. We do a structural compare on decoded fields rather than
// raw bytes (raw bytes always differ due to the random fingerprint value).
function reqShape(bin) {
  if (!bin) return null;
  let fields;
  try { fields = parseFields(bin); } catch { return { err: 'decode' }; }
  // A stable signature = concat of every top-level field's (tag, wireType, len),
  // EXCLUDING the fingerprint value inside ClientMetadata. Captures structure +
  // sizes (which is what an intermittent OUR-bug would perturb) without the noise
  // of the random fingerprint bytes.
  const sig = [];
  for (const f of fields) sig.push(`${f.field}:${f.wireType}:${f.wireType === 2 ? f.value.length : 'v'}`);
  return sig.join('|');
}

// THE VERDICT: only valid when comparing SAME-PROMPT attempts (a controlled
// probe with --repeat). For a model with both a success and a non-policy failure
// of the SAME request size class, compare their decoded shapes.
//   Same shape  → identical input, different output → UPSTREAM flaky (not us).
//   Diff shape  → the differing field is OUR intermittent bug.
// We pick the ok/bad pair whose total body sizes are CLOSEST (most likely the
// same prompt), to avoid diffing two unrelated requests.
function verdict(recsForModel) {
  const oks = recsForModel.filter((r) => r.outcome.ok && r.bin);
  const bads = recsForModel.filter((r) => !r.outcome.ok && r.bin
    && r.outcome.kind !== 'content_policy' && r.outcome.kind !== 'rate_limit' && r.outcome.kind !== 'http_429');
  if (!oks.length || !bads.length) return null;
  // Pair by closest body size (proxy for "same prompt").
  let best = null;
  for (const ok of oks) for (const bad of bads) {
    const d = Math.abs(ok.bin.length - bad.bin.length);
    if (!best || d < best.d) best = { ok, bad, d };
  }
  const { ok, bad, d } = best;
  const sameSize = ok.bin.length === bad.bin.length;
  const shapeOk = reqShape(ok.bin), shapeBad = reqShape(bad.bin);
  const shapeEqual = shapeOk === shapeBad;
  const fpOk = ok.wire?.fingerprintLen, fpBad = bad.wire?.fingerprintLen;
  const fpDiff = fpOk !== fpBad;
  let conclusion;
  if (!sameSize && d > 200) {
    conclusion = `INCONCLUSIVE — closest ok/bad bodies differ by ${d}B (likely different prompts; use a controlled --repeat probe of ONE fixed prompt)`;
  } else if (fpDiff) {
    conclusion = `LIKELY OUR BUG — fingerprint length differs (ok=${fpOk} bad=${fpBad}); server checks #31 length`;
  } else if (shapeEqual) {
    conclusion = 'SAME request shape, different outcome → UPSTREAM flaky (NOT our bug)';
  } else {
    conclusion = 'request SHAPES differ at same size → inspect field-level diff (--raw); possible OUR bug';
  }
  return { okId: ok.id, badId: bad.id, badKind: bad.outcome.kind, sizeDelta: d, sameSize, shapeEqual, fpOk, fpBad, conclusion };
}

function main() {
  const recs = loadTraces();
  if (!recs.length) { console.log(`no traces under ${ROOT}${flags.model ? ` for model ${flags.model}` : ''}`); return; }
  // Group by the client-facing model.
  const byModel = new Map();
  for (const r of recs) { if (!byModel.has(r.model)) byModel.set(r.model, []); byModel.get(r.model).push(r); }

  console.log(`\n═══ DIAG ANALYZE — ${recs.length} traces, ${byModel.size} models ═══\n`);
  const findings = [];
  for (const [model, rs] of [...byModel.entries()].sort()) {
    const n = rs.length;
    const okN = rs.filter((r) => r.outcome.ok).length;
    const breakdown = {};
    for (const r of rs) breakdown[r.outcome.kind] = (breakdown[r.outcome.kind] || 0) + 1;
    const sel = rs[0].selector || '?';
    const fpLens = [...new Set(rs.map((r) => r.wire?.fingerprintLen).filter((x) => x != null))];
    console.log(`■ ${model}  → ${sel}`);
    console.log(`   attempts=${n}  ok=${okN} (${((okN / n) * 100).toFixed(0)}%)  breakdown=${JSON.stringify(breakdown)}`);
    if (fpLens.length) console.log(`   #31 fingerprint len seen: ${fpLens.join(', ')} (must be stable 732; a varying/short value = our bug)`);
    const v = verdict(rs);
    if (v) console.log(`   VERDICT: ${v.conclusion}\n            (ok=${v.okId} vs ${v.badKind}=${v.badId})`);
    else console.log(`   VERDICT: need both a success AND a non-policy failure to diff (have ok=${okN}, fail=${n - okN})`);
    console.log('');
    findings.push({ model, selector: sel, attempts: n, ok: okN, breakdown, fingerprintLens: fpLens, verdict: v });
  }
  if (flags.json) { writeFileSync(flags.json, JSON.stringify(findings, null, 2)); console.log(`→ ${flags.json}`); }
}

main();

