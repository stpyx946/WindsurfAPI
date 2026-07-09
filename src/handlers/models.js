import { listModels } from '../models.js';
import { resolveConnectSelector, __testing } from '../devin-connect-models.js';
import { getBackendSwitch } from '../runtime-config.js';

// GET /v1/models. On a DEVIN_CONNECT deployment (the production transport) only
// expose models that actually resolve to a real catalog selector — otherwise
// /v1/models advertises ~90 models the account can't reach (they'd 400 at chat).
// The MODELS table stays full for the Cascade transport; this is a per-transport
// view, not a catalog edit. Non-connect deployments see the full list unchanged.
export function handleModels(env = process.env) {
  let data = listModels();
  if (getBackendSwitch('devinConnect', env)) {
    data = data.filter((m) => {
      const { selector, mapped } = resolveConnectSelector(m._windsurf_id);
      return mapped && __testing.CATALOG_SELECTORS.has(selector);
    });
  }
  return { object: 'list', data };
}
