# How the Jev pages are built, structured and deployed

Targets analysed on 2026-10-15 with REA 6.5.0 (passive CDP inspection + static bundle analysis, no page JavaScript executed) plus raw HTTP fetches.

- https://vercel.com/ai-gateway/models/jev  (model page)
- https://vercel.com/i/what-is-jev          (article page)

REA evidence IDs

| Operation | Page | Evidence ID |
|---|---|---|
| list_browser_targets | both | ev_d559b01c9d3dfa1c3c0a506bbe6f7b9f5edef694a625c6adfde2032cb702e5c2 |
| analyze_web_bundle | model | ev_40f5555dfcfca8a0805d6fb84329db97f916a7c411d1647e568706c741b45be3 |
| analyze_web_bundle | article | ev_c33e2eb6be7605324853ec14fe0c64fcc6963742a3e5b2bccc5cac1e17a668c0 |
| inspect_web_page | model | ev_63a687400c084470e1f6ce83dcfe72bc0e5b19c3d737a5300bc88d612c5ac32c |
| inspect_web_page | article | ev_3a61f8a877a88c5521aad46c1c49d66301cf3cf5314a2ec4cb2048702def8467 |

Legend: **Observed** = seen directly in headers, HTML, Flight payload or chunk bytes. **Inferred** = a bounded conclusion from observations. **Unknown** = not recoverable from the outside.

## 1. Framework and build

| Fact | Status | Evidence |
|---|---|---|
| Next.js App Router with React Server Components | Observed | `self.__next_f.push` Flight rows, `vary: rsc, next-router-state-tree, next-router-prefetch, next-router-segment-prefetch`; REA vendor fingerprint "Next.js" high confidence |
| Next.js 16.5.0-canary.9 | Observed | version string `next@16.5.0-canary.9` in 4 chunks |
| React 19.3 (canary build 2026-10-07) and 19.2.10 | Observed | `react@19.3.0`, `react@19.2.10`, `version:"19.3.0-canary-b618bbb4-20261007"` in chunks |
| Turbopack production bundler | Observed | `__turbopack` runtime in 80 of 82 chunks; short hashed chunk names (`0-rnn0que1ra3.js`) under `_next/static/immutable/chunks/` |
| Tailwind CSS v4 | Inferred | `--tw-*` custom properties, v4 syntax in class names: container queries `@lg:col-span-8`, `z-(--header-zindex)`, `in-[#...]` |
| Geist design system + shadcn-style primitives | Observed | 35 chunks reference `geist`; `data-slot="geist-icon"`, `data-testid="geistcn/skip-nav-content"`; `clsx`/`tailwind-merge`/`cva` in 11 chunks; Radix in 8 chunks |
| Source maps declared but not served | Observed | 79/84 `.map` URLs declared, all unavailable (REA `source_maps.status: unavailable`) |

## 2. Routing structure (from the Flight router tree)

Model page segment path:

```
(site) / (default) / precomputed / [experimentCode] / ai-gateway / (with-mini-nav) / [aiGatewayFlagsCode] / models / [model] / __PAGE__
```

Sibling segments next to `[model]`: `labs`, `page`, `providers`. Sibling segments next to `[aiGatewayFlagsCode]`: `components`, `lib`, `marketing-overview`, `models`.

Article page segment path:

```
(site) / (default) / precomputed / [experimentCode] / i / [slug] / __PAGE__
```

Sibling segments next to `[slug]`: `ai-gateway-hackathon`, `category`, `components`, `feed`, `page`.

Inferred app directory shape:

```
app/
  (site)/
    (default)/
      precomputed/
        [experimentCode]/
          ai-gateway/
            (with-mini-nav)/
              [aiGatewayFlagsCode]/
                models/[model]/page.tsx
                models/[model]/providers/...
                models/[model]/labs/...
                marketing-overview/
              components/  lib/
          i/
            [slug]/page.tsx
            category/  feed/  ai-gateway-hackathon/
```

The public URL has no `precomputed/...` prefix. The server header shows the internal match:

- `x-matched-path: /precomputed/[experimentCode]/ai-gateway/[aiGatewayFlagsCode]/models/[model]`
- `x-matched-path: /precomputed/exp~-empty-/i/what-is-jev`

This is the Vercel Flags SDK **precompute** pattern: middleware evaluates flags, encodes the result as a code, and rewrites `/ai-gateway/models/jev` to `/precomputed/<experimentCode>/ai-gateway/<flagsCode>/models/jev` so each flag combination is a separately cached static page. Observed `experimentCode = exp~-empty-` (no experiments active for an anonymous visitor). Observed `aiGatewayFlagsCode` is an HS256 JWS whose 9-byte payload is a bit-packed flag vector (decoded header `{"alg":"HS256"}`), which matches the `flags` package's signed precompute serialization. Flight payload also carries `flags` / `flagValues` props (12 and 11 mentions).

## 3. Deployment and hosting

| Fact | Status | Evidence |
|---|---|---|
| Hosted on Vercel | Observed | `server: Vercel`, `x-vercel-id: hkg1::iad1::…` (edge in HKG, origin region iad1), `x-vercel-cache: HIT` / `STALE` |
| Vercel Microfrontends, app id `vercel-marketing` | Observed | asset prefix `/vc-ap-vercel-marketing/_next/static/…` on every chunk; client fetches `/.well-known/vercel/microfrontends/client-config`; that config lists ~7 sibling apps routed by path (docs, academy, geist/design, landers, remote components, etc.) |
| ISR / static prerender with 300 s stale time | Observed | `x-nextjs-prerender: 1`, `x-nextjs-stale-time: 300`, HTML `cache-control: public, max-age=0, must-revalidate` + `x-vercel-cache: HIT` |
| Immutable asset caching | Observed | chunks: `cache-control: public,max-age=31536000,immutable` |
| Deployment id | Observed | `dpl_DxDhzq3YgVThygCZAD45sA4vKZDH` in HTML |
| Payload CMS mounted in the same Next.js app | Observed | `x-powered-by: Next.js, Payload` on both pages; 18 chunks mention `payload` |
| Contentful is the content source of the `/i/` article | Observed | Flight payload contains a Contentful rich-text `document` tree (272 `nodeType` entries: paragraph, heading-2/3, table, hyperlink, list) and a `blogInlineCta` content type; images from `images.ctfassets.net` |
| Images via `next/image` optimizer | Observed | route candidates `/_next/image` and `/vc-ap-vercel-marketing/_next/image` |
| Provider logos on Vercel Blob | Observed | 111 refs to `7nyt0uhk7sse4zvn.public.blob.vercel-storage.com/docs-assets/static/docs/ai-gateway/logos/*.png` |
| Client-hint driven theming | Observed | `accept-ch` / `critical-ch: Sec-CH-Prefers-Color-Scheme` |
| HSTS preload | Observed | `strict-transport-security: max-age=63072000; includeSubDomains; preload` |
| Bot protection declared | Observed | Flight payload lists protected POST routes with `checkLevel: deepAnalysis` (`/api/ai-chat`, `/api/feedback`, `/functions/sales-email`, …), consistent with Vercel BotID |

## 4. Runtime integrations seen in the bundle

| Library / service | Chunks | Role (inferred) |
|---|---|---|
| `ai@2.13.1` (Vercel AI SDK) | 11 | code snippet rendering, `experimental_decide` example, AI chat sidebar |
| `flags` (Flags SDK) | 4 | precompute + `flagValues` hydration |
| Statsig | 4 | experimentation, `/api/stream/internal/statsig` endpoint |
| `@vercel/analytics`, `@vercel/speed-insights` | 2 / 3 | telemetry |
| SWR | 7 | client data fetching |
| `nuqs` | 3 | URL query state (tabs) |
| `zod` | 3 | schema validation |
| `sonner` | 9 | toasts |
| Radix UI | 8 | primitives |
| Prism code block | 2 | `prism-code language-ts` on model page |
| CodeSandbox define API | 2 | "open in sandbox" from article code blocks |

Client endpoints discovered statically (REA `call:fetch`): `/api/feedback`, `/api/geolocation`, `/api/force-draft-mode-marketing?dryRun=1`, `/api/force-deploy-freeze-toggle`, `/api/allow-blog-remote-components`, `/api/stream/internal/statsig`, `/.well-known/vercel/microfrontends/client-config`. These are candidates, not proof of execution.

Browser storage keys observed: localStorage `utmValues`, `ai-chat-enabled`; sessionStorage `srid`, `uval`, `lpid`, `sid`; IndexedDB `ChatMessagesDB`. No service worker, no workers.

## 5. Page anatomy

### Model page `/ai-gateway/models/jev`

- Sticky `#marketing-header`, AI Gateway sub-nav (`nav[aria-label="AI Gateway"]`), tab bar with `availableTabs: ["overview","providers"]` (so `labs` exists as a segment but is hidden for this model).
- `article#overview`: h1 "Jev", description, price badge ("From $0.04 per 1M"), CTA buttons ("Get API key" deep link, "Read the docs"), a model-setup panel with a TypeScript snippet (`experimental_decide` from `ai`), `pythonSnippet: null`, `showAgentTabs: false`.
- `h2#providers` with three tables: provider list, pricing, and performance (keys `servedSpeed`, `throughput`, `averageTokensPerSecond`, `p50/p95TokensPerSecond`, `latency`, `p50/p95TimeToFirstTokenMs`, `cacheHitRatePct`).
- Server data object `backingModels[]` per provider with fields: `name, chef, primaryModel, provider, type ("evaluation"), description, contextSize (32000), maxOutputTokens (0), releaseDate (2026-09-15), inputCost ("0.042"), outputCost ("0"), playgroundUrl, websiteUrl, modelUrl, pricingUrl, zeroDataRetention, disallowPromptTraining, hipaaCompliant, availableToFreeTier, isByokConnectionTestModel, providerResidencyRegions`. Providers observed: `typesafe-ai`, `digitalocean`.
- JSON-LD: `SoftwareApplication` (+ `Offer`, `Organization`) and `BreadcrumbList`.
- Public data source that matches these fields: `GET https://ai-gateway.vercel.sh/v1/models` returns `typesafe-ai/jev` with `context_window: 32000`, `max_tokens: 0`, `type: "evaluation"`, `pricing.input: "0.000000042"` (per token) → $0.042 per 1M. Performance metrics and provider residency are **not** in the public API (Unknown source, inferred internal gateway telemetry).
- Right-hand sticky `aside` = AI chat drawer (`ai-chat-enabled`, `ChatMessagesDB`).

### Article page `/i/what-is-jev`

- Same header/footer shell; a 12-column container-query grid.
- `header` (h1 + lede), `aside` with `time datetime="2026-09-19T00:00+10:00"`, author, category, share.
- `article` with 9 `h2` sections (ids are slugified headings with "Copy link to heading" anchors), two tables, FAQ `h3` group, "Related resources".
- Rendered from a Contentful rich-text document via a renderer component (`$L146` with `content.nodeType: "document"`).
- "More Decision models articles" section with three `article[aria-label]` cards; "Ready to deploy?" CTA section.
- JSON-LD `TechArticle`.
- Record identity `sysId: "jev-182cf60f0ae057d035724661"`, `slug: "what-is-jev"`.

## 6. Unknowns (not recoverable externally)

- Server code: `generateStaticParams`, data loaders, the Payload collections and the Contentful space model. Only the rendered output shape is visible.
- Which parts come from Payload versus Contentful. Both are present; the article body is Contentful rich text, the Payload header may come from the shared marketing shell or other collections.
- Middleware source (flags evaluation, Statsig experiment assignment, microfrontends routing). Only its effects are visible.
- Performance/throughput metrics source for the providers table.
- The Jev model itself (TypeSafe AI). The page only describes it. It cannot be reproduced; it can only be called through the gateway.
- Exact `.map` contents (declared, 404 at runtime).
