// backends/index.mjs — the only place a backend name becomes code.
//
// The server used to do `await import(`./backends/${process.env.JEV_BACKEND}.mjs`)`.
// That turns an environment string into a module path: `JEV_BACKEND=../../x`
// loads whatever sits there. Backends are now a closed table, an unknown name
// fails at startup, and the one backend that sends state off this machine
// refuses to load until the operator says so.

export const BACKENDS = Object.freeze({
  heuristic: {
    load: () => import('./heuristic.mjs'),
    // Lexical overlap and cue words. Deterministic, offline, and semantically
    // meaningless: fine for contract tests, never for a decision.
    testOnly: true,
    thirdParty: false,
  },
  claude: {
    load: () => import('./claude.mjs'),
    testOnly: false,
    // Every uncached request sends the state to Anthropic through the local
    // `claude` CLI. That is data egress, and it is opt-in.
    thirdParty: true,
  },
});

export const DEFAULT_BACKEND = 'heuristic';

export class BackendConfigError extends Error {}

/** Validate a backend name and the egress opt-in. Pure: no import, no I/O. */
export function checkBackend(name, env = process.env) {
  const key = name || DEFAULT_BACKEND;
  if (!Object.hasOwn(BACKENDS, key)) {
    throw new BackendConfigError(
      `unknown JEV_BACKEND ${JSON.stringify(key)}; expected one of: ${Object.keys(BACKENDS).join(', ')}`,
    );
  }
  const spec = BACKENDS[key];
  if (spec.thirdParty && env.JEV_ALLOW_THIRD_PARTY !== '1') {
    throw new BackendConfigError(
      `backend ${key} sends request state to a third party (Anthropic, via the claude CLI). ` +
        'Set JEV_ALLOW_THIRD_PARTY=1 (or pass --allow-third-party) to confirm that the data you ' +
        'send is cleared for that. Do not send production telemetry without that clearance.',
    );
  }
  return { name: key, ...spec };
}

/** checkBackend, then import the module. */
export async function resolveBackend(name, env = process.env) {
  const spec = checkBackend(name, env);
  const module = await spec.load();
  if (typeof module.infer !== 'function') throw new BackendConfigError(`backend ${spec.name} exports no infer()`);
  return { name: spec.name, testOnly: spec.testOnly, thirdParty: spec.thirdParty, infer: module.infer };
}
