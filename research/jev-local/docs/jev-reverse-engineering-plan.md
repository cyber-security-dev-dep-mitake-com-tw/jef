# Plan A: Reverse engineering the Jev pages

Goal: move every row in `jev-findings.md` from Inferred/Unknown to Observed where that is possible from outside, and produce a verified component and data model for both pages. Scope is the two public pages on vercel.com. Nothing here touches authenticated or private endpoints.

## Phase 0: Scope and ethics (done once)

- Targets: the two URLs, their public chunks, and the public `ai-gateway.vercel.sh/v1/models` API.
- Rules: passive inspection and static analysis only. No credential use, no bypass of bot protection, no scraping of Vercel's article text for republication (copyright). Structure and behaviour are the deliverable, not the prose.

## Phase 1: Freeze a reproducible snapshot

1. Save both HTML responses, response headers, and all 82 chunks + CSS with their URLs and SHA-256 digests (REA already recorded script digests in the two `analyze_web_bundle` records).
2. Record the deployment id (`dpl_DxDhzq3YgVThygCZAD45sA4vKZDH`) so later diffs can be attributed to a redeploy.
3. Export the REA session with `export_evidence_bundle` so findings stay citable.

Deliverable: `snapshot/<date>/` with a manifest.

## Phase 2: Routing and flags

1. Confirm the precompute rewrite: fetch the public URL with and without a `rsc: 1` header and compare `x-matched-path`, `vary`, and the Flight router tree.
2. Enumerate the `[experimentCode]` and `[aiGatewayFlagsCode]` spaces: request the page with different `Sec-CH-Prefers-Color-Scheme`, locales, and with/without the Vercel cookies and record which codes come back. Decode each code (HS256 JWS, 9-byte packed payload) and build a table of code → flag vector. The signing key is not needed to read the payload.
3. Map sibling segments (`labs`, `providers`, `marketing-overview`, `category`, `feed`) by requesting them directly and recording which are real pages, redirects, or 404s.
4. Use `inspect_web_event_listeners` on the tab bar to find the handler that switches `overview`/`providers` and confirm it is `nuqs` query state versus route navigation.

Deliverable: `routing.md` with the app-directory tree and a flags table.

## Phase 3: Client bundle to component tree

1. Run `export_web_scripts` on the two inspect_web_page captures (re-run inspect with `include_script_sources: true`) and `analyze_javascript_application` on the exported tree with `detail: "summary"`.
2. Walk the Turbopack module graph from the page entry chunk: identify the model page components (header, tab bar, setup panel with code snippet, providers tables, AI chat drawer) and the article renderer (Contentful rich-text → React mapping, heading-anchor generator, table component, FAQ block, related cards).
3. For each component record props seen in the Flight payload (for example the `$L157` setup panel props: `modelId, modelName, modelType, typescriptSnippet, pythonSnippet, showAgentTabs, apiSurfaces`).
4. Fingerprint shared packages and pin versions: Next 16.5 canary, React 19.3 canary, `ai@2.13.1`, Tailwind v4, Geist, Radix, SWR, nuqs, zod, sonner, Statsig, flags.
5. Optionally run `recover_javascript_sources` (Wakaru, Linux only) on the largest page chunks to get readable module boundaries.

Deliverable: `components.md` listing each component, its props contract, and the chunk it lives in.

## Phase 4: Data model

1. Model page: diff the `backingModels[]` object against `GET /v1/models` for `typesafe-ai/jev` and classify every field as public API, derived (price per 1M = per-token × 1e6), or internal (performance metrics, residency, BYOK flags).
2. Article page: reconstruct the Contentful content type from the rich-text tree and surrounding props (`title, description, slug, sysId, author(s), tags, category, publish date, related resources, inline CTA`). Note every `nodeType` and `marks` value used so the renderer can be rebuilt.
3. Record the JSON-LD schemas (`SoftwareApplication`, `BreadcrumbList`, `TechArticle`) and the OG image route (`/api/model-og`).
4. Confirm which data is Payload-sourced by checking `x-powered-by` on neighbouring routes (`/i`, `/i/category/*`, `/ai-gateway`) and by locating Payload REST/GraphQL paths in the chunks (`/api/...` under the Payload admin prefix).

Deliverable: `data-model.md` with TypeScript types for `GatewayModel`, `BackingModel`, `Article`.

## Phase 5: Runtime behaviour

1. Use `capture_browser_scenario` (launch mode, Google Chrome) with declared actions: load the model page, click the Providers tab, open the AI chat drawer, switch colour scheme. Capture `network` events and `dom` after each step.
2. Confirm which fetch candidates actually fire (`/api/geolocation`, Statsig stream, microfrontends client config, speed insights) and when.
3. Compare the two captures with `compare_web_captures` to isolate page-specific requests from shared-shell requests.

Deliverable: `runtime.md` with a request timeline per interaction.

## Phase 6: Deployment model

1. Document the microfrontends topology from `/.well-known/vercel/microfrontends/client-config` (default app + 7 child apps and their path rules).
2. Verify ISR behaviour: hit the page every 60 s for 10 minutes and watch `age`, `x-vercel-cache` (HIT → STALE → HIT) and `x-nextjs-stale-time`.
3. Verify the edge/origin split from `x-vercel-id` across regions (use a second vantage point or a VPN).

Deliverable: `deployment.md`.

## Phase 7: Verification and ledger

- Re-run Phase 1 after any observed redeploy and diff with `compare_web_captures` / `compare_bundles`.
- Keep the finding ledger: each claim → evidence id → observed/inferred/unknown. Close the plan only when every Phase 2 to 6 deliverable has at least one observed citation.

Effort estimate: Phases 1 to 4 about one working day; Phase 5 and 6 half a day; Phase 7 ongoing.
