import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const dockerfile = readFileSync('Dockerfile', 'utf8');
const dockerignore = readFileSync('.dockerignore', 'utf8');

describe('Docker image script packaging', () => {
  it('enables Devin Connect for direct image runs by default', () => {
    assert.match(dockerfile, /^\s+DEVIN_CONNECT=1 \\$/m);
  });

  it('copies npm smoke/probe scripts that are expected to run inside the container', () => {
    for (const name of ['smoke:native-bridge', 'smoke:lsp-matrix', 'probe:web-search']) {
      const command = pkg.scripts?.[name] || '';
      const script = command.match(/\bnode\s+(\S+)/)?.[1];
      assert.ok(script, `${name} should run a node script`);
      assert.match(dockerfile, new RegExp(`COPY\\s+${script.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`));
      assert.match(dockerignore, new RegExp(`!${script.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`));
    }
  });
});
