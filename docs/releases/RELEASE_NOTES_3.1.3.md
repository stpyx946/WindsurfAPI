# v3.1.3 — Harden the WebGL pool fire (context leaks fixed)

2026-07-10 (UTC+9)

A follow-up to v3.1.2. An adversarial code review of the new WebGL pool fire
found three ways it could leak or thrash WebGL contexts; all three are fixed.
No user-visible behaviour change beyond "the fire no longer wastes GPU / breaks
under a full-pool rate-limit event".

## Fixed

- **Unbounded fire contexts under pool-wide saturation.** A row got its own
  WebGL2 context whenever the account was `saturated`, with no upper bound. A
  pool-wide rate-limit event (the exact state this dashboard exists to show)
  marks many accounts saturated at once, so a 20-account deployment would spawn
  20+ contexts — past the browser's ~16-context cap, which silently drops the
  oldest and thrashes the GPU. Fires are now capped to the **3 hottest** `.hot`
  rows (by intensity, `_FIRE_MAX`); the rest keep the emoji/CSS highlight.
- **Orphaned fires on a transient empty poll.** When a `/connect-metrics` poll
  came back empty (`cm===null`, a common transient blip), `_renderPoolHealth`
  rebuilt the panel to the empty state and returned **without disposing** the
  running fires — their RAF loops kept rendering the full 3-pass pipeline against
  detached canvases until a later non-empty poll. Empty-state now disposes first.
- **Fires running forever after leaving the overview.** `display:none` does not
  pause `requestAnimationFrame` (only a hidden tab does), so navigating to
  another panel left every fire burning its 3-pass GL in the background. `navigate()`
  now disposes all fires (re-mounted on return to overview).

All three funnel through a new `_teardownFires()` that force-loses each WebGL
context. Verified via CDP: 5 saturated accounts → 3 fires; leaving overview and
a `cm===null` poll → 0 fires (contexts reclaimed).

## Notes

- Verified: full test suite green (2494), i18n check green, dashboard syntax green.
