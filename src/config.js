import { readFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

// When packaged as a single .exe (pkg), __dirname/ROOT point INSIDE the
// read-only snapshot (C:\snapshot\...), so .env can't be read from there and
// DATA_DIR would try to mkdir in the snapshot (fails). The real writable
// location is the folder the .exe sits in. Use that as the base for .env and
// the default data dir so a double-clicked exe keeps its state in a `data/`
// folder next to itself — cleanly isolated from the bundled Linux snapshot.
const IS_PACKAGED = !!process.pkg;
const EXE_DIR = IS_PACKAGED ? dirname(process.execPath) : ROOT;

// Load .env file manually (zero dependencies)
function loadEnv() {
  // Packaged: read the .env sitting next to the .exe, not the snapshot copy.
  const envPath = resolve(EXE_DIR, '.env');
  if (!existsSync(envPath)) return;
  const content = readFileSync(envPath, 'utf-8');
  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    let val = trimmed.slice(eqIdx + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    } else {
      // Strip inline comments for unquoted values: PORT=3003 # port → 3003
      const commentIdx = val.indexOf(' #');
      if (commentIdx !== -1) val = val.slice(0, commentIdx).trim();
    }
    if (!process.env[key]) {
      process.env[key] = val;
    }
  }
}

if (process.env.WINDSURFAPI_SKIP_DOTENV !== '1') {
  loadEnv();
}

// Packaged-exe sane defaults (only when the user hasn't set them). The Windows
// exe has no bundled Language Server (Linux-only) and is meant for local
// single-user use, so: DEVIN_CONNECT=1 (pure-HTTP path, skip the LS entirely),
// HOST=127.0.0.1 (loopback — don't expose an unauthenticated gateway to the
// LAN on first run). Both stay overridable via .env / env.
if (IS_PACKAGED) {
  if (!process.env.DEVIN_CONNECT && !process.env.DEVIN_ONLY) process.env.DEVIN_CONNECT = '1';
  if (!process.env.HOST && !process.env.BIND_HOST) process.env.HOST = '127.0.0.1';
}

// `sharedDataDir` is the cluster-shared root: a single accounts.json lives
// here so add-account writes from any replica are visible to every replica
// after restart. `dataDir` is replica-local under REPLICA_ISOLATE=1 and is
// safe to use for telemetry that does not need cross-replica visibility.
// See issue #67 — when the two were collapsed into one path, every
// docker-compose upgrade orphaned the user's accounts.json under a stale
// `replica-${HOSTNAME}` subdir.
// Base that relative DATA_DIR / default data dir resolve against. Packaged:
// the .exe's folder (writable) — NEVER the snapshot ROOT. Default (DATA_DIR
// unset) for a packaged exe is `<exe-dir>/Windsurf_data` so a double-click
// drops all state (accounts/stats/logs) into one tidy folder beside the
// program, isolated from the bundled Linux snapshot.
const DATA_BASE = IS_PACKAGED ? EXE_DIR : ROOT;
const sharedDataDir = process.env.DATA_DIR
  ? resolve(DATA_BASE, process.env.DATA_DIR)
  : (IS_PACKAGED ? join(EXE_DIR, 'Windsurf_data') : ROOT);
const dataDir = (() => {
  let base = sharedDataDir;
  if (process.env.REPLICA_ISOLATE === '1' && process.env.HOSTNAME) {
    base = join(base, `replica-${process.env.HOSTNAME}`);
  }
  return base;
})();

// First-run detection (packaged only): capture whether the data folder exists
// BEFORE we mkdir it, so index.js can decide to auto-open the dashboard only on
// the very first launch (no Windsurf_data folder yet = fresh deploy).
const isFirstRun = IS_PACKAGED && !existsSync(sharedDataDir);

try {
  mkdirSync(sharedDataDir, { recursive: true });
  mkdirSync(dataDir, { recursive: true });
} catch (err) {
  // Don't swallow this — a non-writable DATA_DIR means accounts.json / stats.json
  // / logs all fail to persist later with a confusing downstream error. Surface
  // the path + code now. (log isn't defined yet at module-init time; use console.)
  console.warn(`[WARN] Could not create data dir "${dataDir}" (${err.code || err.message}). `
    + 'accounts/stats/logs may not persist — set DATA_DIR to a writable path.');
}

export function defaultLsBinaryPath(platform = process.platform, arch = process.arch, home = process.env.HOME) {
  if (platform === 'darwin') {
    const name = arch === 'arm64' ? 'language_server_macos_arm' : 'language_server_macos_x64';
    return `${home}/.windsurf/${name}`;
  }
  const name = arch === 'arm64' ? 'language_server_linux_arm' : 'language_server_linux_x64';
  return `/opt/windsurf/${name}`;
}

export const config = {
  port: parseInt(process.env.PORT || '3003', 10),
  // Bind host. Defaults to all interfaces. Set HOST=127.0.0.1 (or BIND_HOST=)
  // for localhost-only deployments — when bound non-locally, missing API_KEY /
  // DASHBOARD_PASSWORD switches to fail-closed instead of default-allow.
  host: process.env.HOST || process.env.BIND_HOST || '0.0.0.0',
  apiKey: process.env.API_KEY || '',
  dataDir,
  sharedDataDir,

  codeiumAuthToken: process.env.CODEIUM_AUTH_TOKEN || '',
  codeiumApiKey: process.env.CODEIUM_API_KEY || '',
  codeiumEmail: process.env.CODEIUM_EMAIL || '',
  codeiumPassword: process.env.CODEIUM_PASSWORD || '',

  codeiumApiUrl: process.env.CODEIUM_API_URL || 'https://server.self-serve.windsurf.com',
  // Astraflow — OpenAI-compatible aggregation platform by UCloud (200+ models)
  // Global:  https://api-us-ca.umodelverse.ai/v1  — signup: https://astraflow.ucloud-global.com
  // China:   https://api.modelverse.cn/v1          — signup: https://astraflow.ucloud.cn
  astraflowApiKey: process.env.ASTRAFLOW_API_KEY || '',
  astraflowApiKeyCn: process.env.ASTRAFLOW_CN_API_KEY || '',
  astraflowApiUrl: 'https://api-us-ca.umodelverse.ai/v1',
  astraflowApiUrlCn: 'https://api.modelverse.cn/v1',
  defaultModel: process.env.DEFAULT_MODEL || 'claude-4.5-sonnet-thinking',
  maxTokens: parseInt(process.env.MAX_TOKENS || '8192', 10),
  logLevel: process.env.LOG_LEVEL || 'info',

  // Language server
  lsBinaryPath: process.env.LS_BINARY_PATH || defaultLsBinaryPath(),
  lsPort: parseInt(process.env.LS_PORT || '42100', 10),

  // Dashboard
  dashboardPassword: process.env.DASHBOARD_PASSWORD || '',

  // Proxy testing
  allowPrivateProxyHosts: process.env.ALLOW_PRIVATE_PROXY_HOSTS === '1',

  // True when running as a packaged single-exe (pkg). Used to enable the
  // double-click desktop UX: data folder beside the exe, auto-open dashboard.
  isPackaged: IS_PACKAGED,
  // True on the very first packaged launch (Windsurf_data didn't exist yet).
  // index.js opens the dashboard in the browser only on this first deploy.
  isFirstRun,
};

const levels = { debug: 0, info: 1, warn: 2, error: 3 };
const currentLevel = levels[config.logLevel] ?? 1;

export const log = {
  debug: (...args) => currentLevel <= 0 && console.log('[DEBUG]', ...args),
  info: (...args) => currentLevel <= 1 && console.log('[INFO]', ...args),
  warn: (...args) => currentLevel <= 2 && console.warn('[WARN]', ...args),
  error: (...args) => currentLevel <= 3 && console.error('[ERROR]', ...args),
};
