import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { deflateSync } from 'node:zlib';
import { tryExtractPdf } from '../src/pdf.js';

describe('PDF extraction safety limits', () => {
  it('falls back when a compressed stream expands beyond the per-stream limit', () => {
    const inflated = Buffer.alloc(6 * 1024 * 1024, 0x20);
    const compressed = deflateSync(inflated);
    const pdf = Buffer.concat([
      Buffer.from('%PDF-1.4\n1 0 obj\n<< /Length ' + compressed.length + ' /Filter /FlateDecode >>\nstream\n', 'latin1'),
      compressed,
      Buffer.from('\nendstream\nendobj\n%%EOF', 'latin1'),
    ]);
    const result = tryExtractPdf(pdf.toString('base64'));
    assert.equal(result.text, 'PDF 内容无法提取');
  });

  // Build an uncompressed PDF whose content stream is `streamBody`.
  function rawPdf(streamBody) {
    const body = Buffer.from(streamBody, 'latin1');
    return Buffer.concat([
      Buffer.from('%PDF-1.4\n1 0 obj\n<< /Length ' + body.length + ' >>\nstream\n', 'latin1'),
      body,
      Buffer.from('\nendstream\nendobj\n%%EOF', 'latin1'),
    ]).toString('base64');
  }

  it('does not catastrophically backtrack on an unclosed [ in a TJ block (ReDoS)', () => {
    // Old /\[((?:[^[\]]*|\([^)]*\))*)\]\s*TJ/ hung for tens of seconds on this.
    const evil = 'BT [' + 'a'.repeat(60) + ' ET';
    const t0 = Date.now();
    const result = tryExtractPdf(rawPdf(evil));
    const ms = Date.now() - t0;
    assert.ok(ms < 2000, `extraction took ${ms}ms — possible ReDoS regression`);
    assert.ok(result !== null);
  });

  it('still extracts text from a normal [...] TJ array', () => {
    // kern -250 < -100 inserts a space between the two strings.
    const result = tryExtractPdf(rawPdf('BT [(Hello) -250 (World)] TJ ET'));
    assert.equal(result.text, 'Hello World');
  });
});

