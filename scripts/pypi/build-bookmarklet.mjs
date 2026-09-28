/**
 * Build the bookmarklet from the readable script.
 *
 * esbuild rather than a regex: collapsing newlines by hand turns every `//`
 * comment into a swallowed line, which produced a bookmarklet that parsed as
 * "Unexpected end of input" -- and only when clicked, not when generated.
 *
 *     node scripts/pypi/build-bookmarklet.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "fill-trusted-publisher.js");
const target = join(here, "bookmarklet.txt");

const minified = execFileSync(
  "npx",
  ["--yes", "esbuild", "--minify", "--target=es2020", source],
  { encoding: "utf8" },
).trim();

// Verify before writing: a broken bookmarklet fails silently on click.
try {
  new Function(minified);
} catch (error) {
  console.error("minified output does not parse:", error.message);
  process.exit(1);
}

writeFileSync(target, `javascript:${encodeURIComponent(minified)}\n`, "utf8");
console.log(
  `wrote ${target}\n  ${readFileSync(source, "utf8").length} chars source` +
    ` → ${minified.length} minified → ${encodeURIComponent(minified).length} encoded`,
);
