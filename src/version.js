import { execSync } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

export const VERSION = (() => {
  try {
    return JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).version;
  } catch { return '1.0.0'; }
})();

export const BRAND = 'WindsurfAPI bydwgx1337';

function firstEnv(names) {
  for (const name of names) {
    const value = String(process.env[name] || '').trim();
    if (value) return value;
  }
  return '';
}

function git(args) {
  if (!existsSync(join(root, '.git'))) return '';
  try {
    return execSync(`git ${args}`, { cwd: root, timeout: 2000 }).toString().trim();
  } catch {
    return '';
  }
}

function shortCommit(s) {
  const value = String(s || '').trim();
  return /^[a-f0-9]{7,40}$/i.test(value) ? value.slice(0, 12) : value;
}

export function getVersionInfo() {
  const commit = shortCommit(firstEnv([
    'WINDSURFAPI_BUILD_COMMIT',
    'BUILD_COMMIT',
    'GIT_COMMIT',
    'SOURCE_COMMIT',
    'VCS_REF',
    'COMMIT_SHA',
  ]) || git('rev-parse --short=12 HEAD'));
  const commitMessage = firstEnv([
    'WINDSURFAPI_BUILD_COMMIT_MESSAGE',
    'BUILD_COMMIT_MESSAGE',
  ]) || git('log -1 --pretty=format:%s');
  const commitDate = firstEnv([
    'WINDSURFAPI_BUILD_COMMIT_DATE',
    'BUILD_COMMIT_DATE',
  ]) || git('log -1 --pretty=format:%cI');
  const branch = firstEnv([
    'WINDSURFAPI_BUILD_BRANCH',
    'BUILD_BRANCH',
    'GIT_BRANCH',
  ]) || git('rev-parse --abbrev-ref HEAD') || 'unknown';
  const source = commit
    ? (existsSync(join(root, '.git')) ? 'git' : 'build-env')
    : 'package';
  return {
    version: firstEnv(['WINDSURFAPI_BUILD_VERSION', 'BUILD_VERSION']) || VERSION,
    commit,
    commitMessage,
    commitDate,
    branch,
    source,
  };
}
