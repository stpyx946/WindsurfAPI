import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolveConnectSelector, FREE_TIER_SELECTOR, __testing } from '../src/devin-connect-models.js';

// Offline catalog drift guard. The committed snapshot (test/fixtures/
// devin-catalog-snapshot.json) is the last known live GetCliModelConfigs
// catalog. These tests fail when the hand-maintained SELECTOR_MAP drifts away
// from what upstream actually serves, prompting a map + snapshot refresh.
//
// No network: the snapshot is the source of truth. The LIVE diff lives in the
// gated smoke (scripts/devin-connect-smoke.mjs / DEVIN_CONNECT_DRIFT_REAL=1),
// keeping `npm test` fully offline. See memory devin-catalog-userstatus-wire-2026-06-30.

const snapshot = JSON.parse(
  readFileSync(new URL('./fixtures/devin-catalog-snapshot.json', import.meta.url), 'utf8'),
);
const catalog = snapshot.models;
const catalogSelectors = new Set(catalog.map((m) => m.selector));

// Selectors that legitimately resolve to a target NOT in the catalog. Keep this
// list tiny and justified — it's the escape hatch for internal/synthetic names.
const NON_CATALOG_TARGETS = new Set([
  'subagent-default', // internal subagent routing alias, not a public model
]);

describe('catalog drift: snapshot integrity', () => {
  it('snapshot count matches its declared _count', () => {
    assert.equal(catalog.length, snapshot._count);
  });

  it('every snapshot selector is unique', () => {
    assert.equal(catalogSelectors.size, catalog.length, 'duplicate selector in snapshot');
  });
});

describe('catalog drift: every live selector resolves', () => {
  for (const { selector } of catalog) {
    it(`resolves selector "${selector}" verbatim`, () => {
      const { selector: resolved, mapped } = resolveConnectSelector(selector);
      assert.equal(mapped, true, `${selector} degraded to free default (not mapped)`);
      assert.equal(resolved, selector, `${selector} should resolve to itself`);
    });
  }
});

describe('catalog drift: every advertised alias resolves to a real selector', () => {
  // Aliases are non-unique FAMILY names — gpt-5.2 spans LOW/MEDIUM/NONE/HIGH/
  // XHIGH, gemini-3.0-flash spans MINIMAL..HIGH, claude-opus-4.5 spans base+
  // thinking. So an alias must resolve (mapped) to SOME real catalog selector
  // of the same family — not necessarily the exact row it was advertised on.
  const seen = new Set();
  for (const { alias } of catalog) {
    if (!alias || seen.has(alias)) continue;
    seen.add(alias);
    it(`alias "${alias}" resolves to a real catalog selector`, () => {
      const { selector: resolved, mapped } = resolveConnectSelector(alias);
      assert.equal(mapped, true, `catalog alias ${alias} degraded to free default`);
      assert.ok(
        catalogSelectors.has(resolved),
        `catalog alias ${alias} resolved to ${resolved}, which is not in the catalog`,
      );
    });
  }
});

describe('catalog drift: hand-curated map targets exist in the catalog', () => {
  // Every value the SELECTOR_MAP points at should be a real catalog selector
  // (or an allowlisted internal name). Catches the "stale default-variant"
  // class of drift: e.g. claude-opus-4-8 -> claude-opus-4-8-medium silently
  // pointing at a selector upstream renamed away.
  for (const [name, target] of __testing.SELECTOR_MAP.entries()) {
    it(`map target "${target}" (from "${name}") is a real catalog selector`, () => {
      if (NON_CATALOG_TARGETS.has(target)) return;
      assert.ok(
        catalogSelectors.has(target),
        `SELECTOR_MAP["${name}"] = "${target}" is not in the catalog snapshot — `
        + 'either upstream renamed it (update the map) or the snapshot is stale '
        + '(refresh test/fixtures/devin-catalog-snapshot.json).',
      );
    });
  }
});

describe('catalog drift: regression — gpt-5.5 bare alias', () => {
  // The catalog advertises bare `gpt-5.5`; it must NOT degrade to the free
  // selector (the bug fixed alongside this test).
  it('bare gpt-5.5 resolves to gpt-5-5-low, not the free default', () => {
    const { selector, mapped } = resolveConnectSelector('gpt-5.5');
    assert.equal(mapped, true);
    assert.equal(selector, 'gpt-5-5-low');
    assert.notEqual(selector, FREE_TIER_SELECTOR);
  });

  it('dash form gpt-5-5 resolves too', () => {
    assert.equal(resolveConnectSelector('gpt-5-5').selector, 'gpt-5-5-low');
  });
});
