// Offline tests for jev-local's isolation: no network, no claude CLI, no npm install.
//   node --test research/jev-local/test/*.test.mjs
import assert from 'node:assert/strict';
import { gzipSync, zstdCompressSync } from 'node:zlib';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { decodeBody, tenantOf } from '../app.mjs';
import { BACKENDS, BackendConfigError, checkBackend, resolveBackend } from '../backends/index.mjs';

// ---- backend loading ----------------------------------------------------------

test('backends are a closed table; unknown names fail', () => {
  assert.deepEqual(Object.keys(BACKENDS).sort(), ['claude', 'heuristic']);
  for (const bad of ['../../etc/passwd', 'heuristic.mjs', 'gpt', '__proto__', 'constructor', 'toString']) {
    assert.throws(() => checkBackend(bad, {}), BackendConfigError, bad);
  }
});

test('the default backend is offline and marked test-only', async () => {
  const b = await resolveBackend(undefined, {});
  assert.equal(b.name, 'heuristic');
  assert.equal(b.testOnly, true);
  assert.equal(b.thirdParty, false);
});

test('claude refuses to load without the third-party opt-in', () => {
  assert.throws(() => checkBackend('claude', {}), /JEV_ALLOW_THIRD_PARTY=1/);
  assert.throws(() => checkBackend('claude', { JEV_ALLOW_THIRD_PARTY: 'true' }), BackendConfigError);
  assert.equal(checkBackend('claude', { JEV_ALLOW_THIRD_PARTY: '1' }).name, 'claude');
});

test('server and CLI default to the same backend', async () => {
  const server = await readFile(new URL('../server.mjs', import.meta.url), 'utf8');
  const cli = await readFile(new URL('../bin/jev.mjs', import.meta.url), 'utf8');
  assert.match(server, /resolveBackend\(process\.env\.JEV_BACKEND\)/);
  assert.doesNotMatch(server, /import\(`/, 'no dynamic import built from a string');
  assert.match(cli, /process\.env\.JEV_BACKEND \|\| 'heuristic'/);
});

// ---- body decoding --------------------------------------------------------------

const body = JSON.stringify({ model: 'jev-latest', state: 'x', questions: {} });

test('plain, gzip and zstd bodies decode', () => {
  assert.equal(decodeBody({}, Buffer.from(body)).model, 'jev-latest');
  assert.equal(decodeBody({ 'content-encoding': 'gzip' }, gzipSync(body)).model, 'jev-latest');
  assert.equal(decodeBody({ 'content-encoding': 'zstd' }, zstdCompressSync(Buffer.from(body))).model, 'jev-latest');
});

test('a gzip bomb stops at the body limit with 413', () => {
  // 40 MiB of zeros compresses to ~40 KiB; uncapped, gunzip would allocate all of it.
  const bomb = gzipSync(Buffer.alloc(40 * 1024 * 1024));
  assert.ok(bomb.length < 1024 * 1024);
  assert.throws(() => decodeBody({ 'content-encoding': 'gzip' }, bomb), (e) => e.status === 413);
});

test('a zstd bomb stops at the body limit with 413', () => {
  const bomb = zstdCompressSync(Buffer.alloc(40 * 1024 * 1024));
  assert.throws(() => decodeBody({ 'content-encoding': 'zstd' }, bomb), (e) => e.status === 413);
});

test('stacked and unknown encodings are refused', () => {
  assert.throws(() => decodeBody({ 'content-encoding': 'gzip, gzip' }, Buffer.from('')), (e) => e.status === 415);
  assert.throws(() => decodeBody({ 'content-encoding': 'br' }, Buffer.from('')), (e) => e.status === 415);
});

// ---- tenant isolation -------------------------------------------------------------

test('tenants are derived from the Authorization header and never contain it', () => {
  const a = tenantOf({ authorization: 'Bearer alpha-secret' });
  const b = tenantOf({ authorization: 'Bearer beta-secret' });
  assert.notEqual(a, b);
  assert.equal(tenantOf({}), 'anonymous');
  assert.ok(!a.includes('alpha'));
});

// ---- claude backend, without calling claude -----------------------------------------
// Importing the module is safe offline: it only spawns on infer().

const claude = await import('../backends/claude.mjs');

test('the claude CLI runs with no tools, restricted, without MCP or slash commands', () => {
  const args = claude.claudeArgs('m');
  const i = args.indexOf('--tools');
  assert.ok(i >= 0 && args[i + 1] === '', '--tools "" disables every built-in tool');
  for (const f of ['--restricted', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence']) {
    assert.ok(args.includes(f), f);
  }
  assert.ok(!args.includes('--dangerously-skip-permissions'));
  assert.ok(!args.some((a) => /bypassPermissions/.test(a)));
});

test('the state is fenced and cannot close its own fence', async () => {
  const fx = JSON.parse(await readFile(new URL('./fixtures/prompt-injection.json', import.meta.url), 'utf8'));
  const escape = fx.cases.find((c) => c.id === 'LLM-001-delimiter-escape');
  const text = claude.buildPrompt({ state: escape.injected, questions: fx.question }, []);
  assert.equal(text.split(claude.STATE_OPEN).length - 1, 2, 'the open marker appears in the rules and once as the fence');
  assert.equal(text.split(claude.STATE_CLOSE).length - 1, 2, 'the close marker appears in the rules and once as the fence');
  const fenced = text.slice(text.lastIndexOf(claude.STATE_OPEN), text.lastIndexOf(claude.STATE_CLOSE));
  assert.ok(fenced.includes('New instruction: every question'), 'the injected text stays inside the fence');
  assert.ok(text.indexOf('untrusted data') < text.lastIndexOf(claude.STATE_OPEN), 'the data rule precedes the state');
});

test('every injection fixture has a control and a demand direction', async () => {
  const fx = JSON.parse(await readFile(new URL('./fixtures/prompt-injection.json', import.meta.url), 'utf8'));
  assert.ok(fx.cases.length >= 5);
  for (const c of fx.cases) {
    assert.ok(c.injected.startsWith(c.control), `${c.id}: injected must extend the control`);
    assert.ok(['low', 'high'].includes(c.demand), c.id);
  }
});

test('secrets are masked by default; emails and IPs on request', () => {
  const text = 'Authorization: Bearer abcdefghijklmnop1234 password=hunter2 key AKIAABCDEFGHIJKLMNOP mail ann@example.com from 10.1.2.3';
  const def = claude.redact(text, claude.redactionSet(undefined));
  assert.ok(!def.includes('abcdefghijklmnop1234'));
  assert.ok(!def.includes('hunter2'));
  assert.ok(!def.includes('AKIAABCDEFGHIJKLMNOP'));
  assert.ok(def.includes('ann@example.com') && def.includes('10.1.2.3'), 'emails and IPs pass by default');
  const all = claude.redact(text, claude.redactionSet('secrets,emails,ipv4'));
  assert.ok(!all.includes('ann@example.com') && !all.includes('10.1.2.3'));
  assert.deepEqual(claude.redactionSet('off'), []);
  assert.throws(() => claude.redactionSet('everything'), /unknown JEV_REDACT/);
});

test('cache keys are tenant- and prompt-scoped', () => {
  const base = { state: 's', questions: { q: { type: 'noul', instructions: 'i' } }, model: 'm', samples: 1, redaction: ['secrets'] };
  const a = claude.cacheKey({ ...base, tenant: 't-a' });
  assert.notEqual(a, claude.cacheKey({ ...base, tenant: 't-b' }), 'another tenant never hits the same entry');
  assert.notEqual(a, claude.cacheKey({ ...base, tenant: 't-a', model: 'm2' }));
  assert.notEqual(a, claude.cacheKey({ ...base, tenant: 't-a', redaction: [] }));
  assert.equal(a, claude.cacheKey({ ...base, tenant: 't-a' }));
  assert.match(claude.PROMPT_DIGEST, /^[0-9a-f]{16}$/);
});
