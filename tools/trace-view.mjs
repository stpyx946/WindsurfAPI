#!/usr/bin/env node
// trace-view — inspect a full-chain request trace produced by src/trace.js
// (WINDSURFAPI_TRACE=1). Reads a <traceId>/ dir and prints the stitched chain:
// client request → routing → Devin wire bytes (size + protobuf field peek) →
// client response, plus the ordered timeline. Pure node, zero deps.
//
// Usage:
//   node tools/trace-view.mjs                 list traces (newest first)
//   node tools/trace-view.mjs <traceId>       show one trace end to end
//   node tools/trace-view.mjs <traceId> --raw hexdump the upstream req/res bytes
//   flags: --dir <traceRoot> (default .trace or $WINDSURFAPI_TRACE_DIR)

import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const argv = process.argv.slice(2);
const flags = {};
const pos = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--raw') flags.raw = true;
  else if (argv[i] === '--dir') flags.dir = argv[++i];
  else pos.push(argv[i]);
}
const ROOT = flags.dir || process.env.WINDSURFAPI_TRACE_DIR || resolve(process.cwd(), '.trace');

if (!existsSync(ROOT)) {
  console.error(`no trace dir at ${ROOT} (set WINDSURFAPI_TRACE=1 and make a request, or pass --dir)`);
  process.exit(1);
}

const readJson = (p) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return null; } };
const fmtBytes = (n) => n < 1024 ? `${n}B` : `${(n / 1024).toFixed(1)}KB`;

if (!pos.length) {
  // List traces newest-first with a one-line summary.
  const dirs = readdirSync(ROOT).filter((d) => { try { return statSync(join(ROOT, d)).isDirectory(); } catch { return false; } });
  const rows = dirs.map((d) => {
    const req = readJson(join(ROOT, d, '01-client-req.json'));
    const rt = readJson(join(ROOT, d, '02-routing.json'));
    const res = readJson(join(ROOT, d, '05-client-res.json'));
    const mt = (() => { try { return statSync(join(ROOT, d)).mtimeMs; } catch { return 0; } })();
    return { d, mt, model: req?.model, selector: rt?.selector, status: res?.status, ms: res?.ms };
  }).sort((a, b) => b.mt - a.mt);
  console.log(`traces in ${ROOT}:\n`);
  for (const r of rows) {
    console.log(`  ${r.d}  ${r.model || '?'} → ${r.selector || '?'}  status=${r.status ?? '?'}  ${r.ms != null ? r.ms + 'ms' : ''}`);
  }
  console.log(`\n(${rows.length} traces) — node tools/trace-view.mjs <traceId> to open one`);
  process.exit(0);
}

const id = pos[0];
const dir = join(ROOT, id);
if (!existsSync(dir)) { console.error(`no trace ${id} under ${ROOT}`); process.exit(1); }

const line = (s = '') => console.log(s);
line(`\n═══ TRACE ${id} ═══`);

const req = readJson(join(dir, '01-client-req.json'));
if (req) {
  line(`\n▶ 01 CLIENT REQUEST  [${req.protocol}]`);
  line(`   model: ${req.model}   stream: ${req.stream}   messages: ${req.messageCount}   tools: ${req.toolCount}`);
  if (req.callerKey) line(`   callerKey: ${req.callerKey}`);
}
const rt = readJson(join(dir, '02-routing.json'));
if (rt) {
  line(`\n▶ 02 ROUTING`);
  line(`   requested: ${rt.requestedModel}  →  selector: ${rt.selector}   (mapped=${rt.mapped})`);
  line(`   backend: ${rt.backend}   account: ${rt.account || '?'} (${rt.accountTier || '?'})`);
  line(`   nativeToolCall: ${rt.nativeToolCall}   tools: ${rt.toolCount}`);
}
for (const [leg, label] of [['03-upstream-req', '03 UPSTREAM REQUEST → Devin'], ['04-upstream-res', '04 UPSTREAM RESPONSE ← Devin']]) {
  const bin = join(dir, `${leg}.bin`);
  if (existsSync(bin)) {
    const buf = readFileSync(bin);
    line(`\n▶ ${label}   ${fmtBytes(buf.length)}`);
    const txt = join(dir, `${leg}.txt`);
    if (existsSync(txt)) line(`   ${readFileSync(txt, 'utf8').trim().slice(0, 200)}`);
    if (flags.raw) line(hexdump(buf.slice(0, 512)));
  }
}
const res = readJson(join(dir, '05-client-res.json'));
if (res) {
  line(`\n▶ 05 CLIENT RESPONSE`);
  line(`   status: ${res.status}   stream: ${res.stream}   ${res.ms}ms   cached: ${res.cached}`);
}
const tl = join(dir, 'timeline.jsonl');
if (existsSync(tl)) {
  line(`\n▶ TIMELINE`);
  const evts = readFileSync(tl, 'utf8').trim().split('\n').map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  const t0 = evts[0]?.t || 0;
  for (const e of evts) line(`   +${String(e.t - t0).padStart(5)}ms  ${e.leg}`);
}
line('');

function hexdump(buf) {
  const lines = [];
  for (let i = 0; i < buf.length; i += 16) {
    const slice = buf.slice(i, i + 16);
    const hex = [...slice].map((b) => b.toString(16).padStart(2, '0')).join(' ');
    const ascii = [...slice].map((b) => (b >= 32 && b < 127) ? String.fromCharCode(b) : '.').join('');
    lines.push(`   ${i.toString(16).padStart(4, '0')}  ${hex.padEnd(47)}  ${ascii}`);
  }
  return lines.join('\n');
}
