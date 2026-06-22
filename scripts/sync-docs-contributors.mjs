import { copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const source = join(root, 'src/dashboard/data/contributors.json');
const target = join(root, 'docs/dashboard/data/contributors.json');

JSON.parse(readFileSync(source, 'utf8'));
mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);

console.log(`Synced ${source} -> ${target}`);
