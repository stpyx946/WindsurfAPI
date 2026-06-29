/**
 * Model name → DEVIN_CONNECT selector resolver.
 *
 * GetChatMessageRequest.model_selector (proto field #21) takes a STRING selector,
 * not the Cascade modelUid/enum. The full catalog was frame-captured from a live
 * GetCliModelConfigs response (see .workflow-results/devin-protobuf/
 * model-catalog-CAPTURED.md and memory: devin-getchatmessage-wire-calibrated-
 * 2026-06-30). This maps the OpenAI-style model names a client sends onto the
 * verified upstream selectors.
 *
 * Free-tier accounts only resolve `swe-1-6-slow`; every other selector returns
 * "/upgrade to access this model". That's an account-tier wall, not a protocol
 * gap — the mapping is complete and ready for a paid entitlement.
 */

// Canonical OpenAI-ish name / alias → upstream selector (#21). Both the
// dash-form and enum-form selectors are accepted by the API; we prefer the
// dash-form where the catalog exposed one.
const SELECTOR_MAP = new Map(Object.entries({
  // ── SWE / Cognition (free-tier reachable) ──
  'swe-1-6-slow': 'swe-1-6-slow',
  'swe-1.6-slow': 'swe-1-6-slow',
  'swe-1-6': 'swe-1-6',
  'swe-1.6': 'swe-1-6',
  'swe-1-6-fast': 'swe-1-6-fast',
  'swe-1.6-fast': 'swe-1-6-fast',
  'swe-1-5': 'MODEL_SWE_1_5_SLOW',
  'swe-1.5': 'MODEL_SWE_1_5_SLOW',
  'swe-1-5-fast': 'MODEL_SWE_1_5',
  'swe-1.5-fast': 'MODEL_SWE_1_5',
  'subagent-default': 'subagent-default',

  // ── Anthropic (paid) ──
  'claude-opus-4-8': 'claude-opus-4-8-medium',
  'claude-opus-4.8': 'claude-opus-4-8-medium',
  'claude-opus-4-8-medium': 'claude-opus-4-8-medium',
  'claude-sonnet-4-6': 'claude-sonnet-4-6-thinking',
  'claude-sonnet-4.6': 'claude-sonnet-4-6-thinking',
  'claude-sonnet-4-6-thinking': 'claude-sonnet-4-6-thinking',
  'claude-opus-4-5': 'MODEL_CLAUDE_4_5_OPUS',
  'claude-opus-4.5': 'MODEL_CLAUDE_4_5_OPUS',
  'claude-opus-4-5-thinking': 'MODEL_CLAUDE_4_5_OPUS_THINKING',
  'claude-sonnet-4-5': 'MODEL_PRIVATE_2',
  'claude-sonnet-4.5': 'MODEL_PRIVATE_2',
  'claude-sonnet-4-5-thinking': 'MODEL_PRIVATE_3',
  'claude-haiku-4-5': 'MODEL_PRIVATE_11',
  'claude-haiku-4.5': 'MODEL_PRIVATE_11',

  // ── OpenAI (paid) ──
  'gpt-5-5-low': 'gpt-5-5-low',
  'gpt-5.5-low': 'gpt-5-5-low',
  'gpt-5-2': 'MODEL_GPT_5_2_NONE',
  'gpt-5.2': 'MODEL_GPT_5_2_NONE',
  'gpt-5-2-low': 'MODEL_GPT_5_2_LOW',
  'gpt-5-2-medium': 'MODEL_GPT_5_2_MEDIUM',
  'gpt-5-2-high': 'MODEL_GPT_5_2_HIGH',
  'gpt-5-2-xhigh': 'MODEL_GPT_5_2_XHIGH',

  // ── Google (paid) ──
  'gemini-3-flash': 'MODEL_GOOGLE_GEMINI_3_0_FLASH_MEDIUM',
  'gemini-3-flash-minimal': 'MODEL_GOOGLE_GEMINI_3_0_FLASH_MINIMAL',
  'gemini-3-flash-low': 'MODEL_GOOGLE_GEMINI_3_0_FLASH_LOW',
  'gemini-3-flash-medium': 'MODEL_GOOGLE_GEMINI_3_0_FLASH_MEDIUM',
  'gemini-3-flash-high': 'MODEL_GOOGLE_GEMINI_3_0_FLASH_HIGH',

  // ── Others (paid) ──
  'glm-5-2': 'glm-5-2',
  'glm-5.2': 'glm-5-2',
  'kimi-k2-7': 'kimi-k2-7',
}));

// The only selector a free-tier account can actually run. Used as the safe
// default when DEVIN_CONNECT is enabled but the requested model isn't mapped.
export const FREE_TIER_SELECTOR = 'swe-1-6-slow';

/**
 * Resolve a client-supplied model name to an upstream DEVIN_CONNECT selector.
 * Normalizes case and dot/dash variations. Returns the free-tier default for
 * unknown names so an enabled DEVIN_CONNECT deploy never hard-fails on an
 * unmapped alias — it degrades to the one selector that always works.
 *
 * @param {string} model
 * @returns {{ selector: string, mapped: boolean }}
 */
export function resolveConnectSelector(model) {
  const raw = String(model || '').trim();
  if (!raw) return { selector: FREE_TIER_SELECTOR, mapped: false };

  // Direct hit (covers both dash-form and enum-form selectors passed verbatim).
  if (SELECTOR_MAP.has(raw)) return { selector: SELECTOR_MAP.get(raw), mapped: true };

  // Normalize: lowercase, collapse dots to dashes, strip a leading provider
  // prefix some clients prepend (e.g. "anthropic/claude-...").
  const norm = raw.toLowerCase().replace(/^[a-z]+\//, '').replace(/\./g, '-');
  if (SELECTOR_MAP.has(norm)) return { selector: SELECTOR_MAP.get(norm), mapped: true };

  // Enum-form selectors are accepted as-is by the API; pass them through.
  if (/^MODEL_[A-Z0-9_]+$/.test(raw)) return { selector: raw, mapped: true };

  return { selector: FREE_TIER_SELECTOR, mapped: false };
}

export const __testing = { SELECTOR_MAP };
