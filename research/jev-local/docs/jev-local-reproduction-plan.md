# Plan B: Reproducing the Jev pages locally (code + environment)

What is reproducible: the two pages' architecture, routing, data flow, deployment pattern, and look. What is not: the Jev model (it belongs to TypeSafe AI and is only callable through the gateway), Vercel's internal performance metrics, their Contentful/Payload content, and their article text (copyright; write your own content or stub it).

## 1. Environment

| Item | Choice | Why |
|---|---|---|
| Node | 24.x (already installed: v24.20.0) | matches a 2026 Next canary |
| Package manager | pnpm | workspace needed for microfrontends |
| Framework | `next@canary` (16.5.x), `react@canary` (19.3) | matches observed versions; fall back to `next@16` stable if canary breaks |
| Bundler | Turbopack (`next build --turbopack`, default in 16) | matches observed chunk layout |
| Styling | Tailwind v4 + Geist fonts (`geist` npm package) + shadcn/ui primitives (Radix) | matches observed class syntax and `data-slot` markers |
| Flags | `flags` (Flags SDK) with `precompute` | matches `/precomputed/[code]` rewrite |
| Experiments | Statsig (optional) via `@flags-sdk/statsig` | observed in bundle; can be stubbed |
| CMS | Payload 3 inside the Next app (SQLite adapter locally) for the model/site records, Contentful (free space) or a local JSON fixture for the article rich text | matches `x-powered-by: Next.js, Payload` and Contentful rich text |
| AI SDK | `ai@^2.13` | for the snippet and the optional chat drawer |
| Deploy | Vercel CLI (`vercel dev` locally, `vercel deploy`), optional `@vercel/microfrontends` with two apps | matches hosting pattern |

## 2. Repository layout

```
jev-repro/
  pnpm-workspace.yaml
  microfrontends.json            # optional: marketing (default) + docs child app
  apps/marketing/
    next.config.ts               # images.remotePatterns for blob + ctfassets; assetPrefix via microfrontends
    middleware.ts                # flags precompute rewrite
    flags.ts                     # flag definitions + precompute groups
    payload.config.ts            # Payload 3 with sqlite adapter
    app/
      (site)/
        layout.tsx               # header, footer, AI-chat drawer slot, theme via client hints
        (default)/
          precomputed/[experimentCode]/
            ai-gateway/(with-mini-nav)/[aiGatewayFlagsCode]/
              layout.tsx         # AI Gateway sub-nav
              models/[model]/
                page.tsx         # overview + providers tabs (nuqs query state)
                providers/page.tsx
            i/
              [slug]/page.tsx    # article
              category/[slug]/page.tsx
              feed/route.ts      # RSS
      api/model-og/route.tsx     # OG image (ImageResponse)
      api/geolocation/route.ts
      api/feedback/route.ts
    components/
      geist/                     # button, tabs, table, code-block (prism/shiki), skip-nav
      model/                     # ModelHeader, SetupPanel, ProvidersTable, PricingTable, PerfTable
      article/                   # RichText renderer (Contentful → React), Heading anchors, FAQ, RelatedCards
      chat/                      # optional AI chat drawer (IndexedDB ChatMessagesDB)
    lib/
      gateway.ts                 # fetch + normalise https://ai-gateway.vercel.sh/v1/models
      pricing.ts                 # per-token → per-1M formatting ("From $0.04 per 1M")
      jsonld.ts                  # SoftwareApplication, BreadcrumbList, TechArticle
      contentful.ts | fixtures/  # rich-text documents
  fixtures/
    models.json                  # snapshot of /v1/models (public)
    articles/what-is-jev.json    # your own rich-text document in Contentful shape
```

## 3. Build steps

### Step 1: Scaffold

```bash
pnpm dlx create-next-app@canary apps/marketing --ts --tailwind --app --turbopack --src-dir=false
cd apps/marketing && pnpm add ai flags swr nuqs zod sonner geist @radix-ui/react-tabs class-variance-authority clsx tailwind-merge
pnpm add payload @payloadcms/next @payloadcms/db-sqlite @payloadcms/richtext-lexical
pnpm add -D @vercel/microfrontends
```

### Step 2: Flags precompute routing

- Define flags in `flags.ts` (`themeFlag`, `showLabsTab`, `aiChatEnabled`, …) and two precompute groups: `experimentFlags` and `aiGatewayFlags`.
- `middleware.ts`: for `/ai-gateway/:path*` and `/i/:path*`, call `precompute()` on each group and `NextResponse.rewrite()` to `/precomputed/${expCode}/ai-gateway/${gwCode}/...` or `/precomputed/${expCode}/i/...`. Use `exp~-empty-`-style neutral code when no experiment applies.
- Pages read the codes from params and call `deserialize` to get `flagValues`; pass them to client components as props (matches the `flagValues` prop seen in the Flight payload).
- `generateStaticParams` returns the known code combinations so pages are prerendered (ISR `revalidate = 300` to match `x-nextjs-stale-time: 300`).

### Step 3: Model page

- `lib/gateway.ts`: fetch `https://ai-gateway.vercel.sh/v1/models` at build/revalidate time (cache 300 s), find `id === params.model` (map `jev` → `typesafe-ai/jev` with a slug table), and map to `BackingModel` (`contextSize ← context_window`, `maxOutputTokens ← max_tokens`, `inputCost ← pricing.input × 1e6`, `releaseDate ← released`, `zeroDataRetention ← zdr`, `disallowPromptTraining ← no_training`).
- Providers: the public API lists one entry per model; add a static `providers.json` for extra backing providers (for example `digitalocean`) and leave performance columns as "n/a" unless you collect your own measurements.
- Components: `ModelHeader` (h1, description, "From $X per 1M" badge, CTA buttons), `SetupPanel` (Prism code block with the `experimental_decide` TypeScript snippet, `showAgentTabs=false`, no Python tab), `Tabs` (`overview`, `providers`, hidden `labs`) driven by `nuqs`, three tables.
- Metadata: `generateMetadata` for title/description, JSON-LD `SoftwareApplication` + `Offer` + `BreadcrumbList`, OG image from `api/model-og`.

### Step 4: Article page

- Content: either a free Contentful space with a `blogPost` type (title, slug, excerpt, body rich text, author, category, tags, publishedAt, related, inline CTA) or the JSON fixture in Contentful rich-text shape.
- `RichText` renderer: map `document, paragraph, heading-2, heading-3, unordered-list, list-item, table, table-row, table-header-cell, table-cell, hyperlink, text (+marks)` to React; generate heading ids by slugifying text and render "Copy link to heading" anchors.
- Layout: 12-column grid with container queries, `header` (h1 + lede), `aside` (time, author, category, share), `article`, FAQ block, Related resources, "More articles" cards, CTA section.
- Metadata: JSON-LD `TechArticle`, canonical URL, OG image.

### Step 5: Shell and design

- `(site)/layout.tsx`: sticky `#marketing-header` with nav panels, skip-nav, footer "Vercel Directory"-style column nav, right-hand drawer slot.
- Theme: read `Sec-CH-Prefers-Color-Scheme` (set `Accept-CH` / `Critical-CH` headers in `next.config.ts` `headers()`), default to system.
- Tailwind v4 config with Geist tokens (`--ds-*` variables, `text-heading-32`, `text-copy-16`, `text-label-14` utilities).
- Instrumentation: `@vercel/analytics`, `@vercel/speed-insights` (no-ops locally).

### Step 6: Payload CMS

- `payload.config.ts` with SQLite adapter, collections `models` (slug, display name, overrides, hidden tabs), `providers`, `pages`. Mount admin at `/admin` via `@payloadcms/next`. This yields the same `x-powered-by: Next.js, Payload` header.

### Step 7: Local run and checks

```bash
pnpm dev --turbopack
```

Checks:

- `curl -I http://localhost:3000/ai-gateway/models/jev` shows `x-powered-by: Next.js, Payload` and the page renders the same section order as the findings.
- The router tree in the Flight payload contains `precomputed / [experimentCode] / ai-gateway / (with-mini-nav) / [aiGatewayFlagsCode] / models / [model]`.
- REA parity: point `analyze_web_bundle` at the local page (headless Chrome, CDP 9222) and compare route/endpoint candidates and vendor fingerprints with the originals using `compare_web_captures`.

### Step 8: Deploy pattern (optional)

- `vercel link && vercel deploy` for a single app; expect `x-vercel-cache`, `x-nextjs-prerender: 1`, immutable chunk caching.
- Microfrontends: add `microfrontends.json` with `marketing` as default and a second app (for example `docs`) routed on `/docs/:path*`; chunks then get the `/vc-ap-<app>/_next/...` prefix and the client fetches `/.well-known/vercel/microfrontends/client-config`, as observed.

## 4. Acceptance criteria

1. Both routes render with the same landmark structure (header, sub-nav, article/aside grid, footer, drawer).
2. Model data comes from the public gateway API and formats price as "From $0.04 per 1M".
3. Flags precompute rewrite works and is visible in the router tree; ISR revalidate is 300 s.
4. Article renders from a rich-text document through the custom renderer with anchored headings, tables, FAQ and related cards.
5. JSON-LD and OG image routes exist.
6. Local REA bundle analysis shows the same vendor set (Next, React, Turbopack, flags, swr, nuqs, zod, sonner, Radix, Geist, Payload).

## 5. Out of scope

- The Jev model weights or inference. Calling it requires a Vercel AI Gateway API key and TypeSafe AI's `decide` API; use the gateway for a live demo only.
- Vercel's internal performance telemetry, Statsig experiment definitions, and bot-protection configuration.
- Republishing Vercel's article text or provider logos.
