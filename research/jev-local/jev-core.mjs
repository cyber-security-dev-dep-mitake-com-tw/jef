// jev-core.mjs — behaviourally compatible core of TypeSafe AI's "Jev" System One
// decision model, reconstructed from public contracts:
//   - TypeSafe HTTP API docs (POST /v1/systemone, question/answer types, limits)
//   - TypeSafe confidence formulas (docs.typesafe.ai/confidence)
//   - @ai-sdk/gateway GatewayDecisionModel (POST {baseURL}/decision-model, spec v4)
//   - ai (AI SDK) validateDecisionInput / validateDecisionAnswers
// Nothing here is TypeSafe's weights. The inference step is pluggable (see backends/).

export const MODEL_IDS = {
  'jev-latest': 'jev-1.13.0',
  'jev-preview': 'jev-1.14.0',
  'jev-1.13.0': 'jev-1.13.0',
  'jev-1.14.0': 'jev-1.14.0',
  'typesafe-ai/jev': 'jev-1.13.0',   // Vercel AI Gateway id
  'jev': 'jev-1.13.0',
};

export const LIMITS = {
  perRequestTokens: 64_000,
  perQuestionTokens: 32_000,
  longContextTokens: 262_000,
  maxChoiceOptions: 255,
  minScoreLevels: 2,
  maxScoreLevels: 10,
  maxBodyBytes: 32 * 1024 * 1024,
  maxNesting: 64,
};

export const ROUNDING = { probabilityDecimals: 4, scoreDecimals: 4 };

// ---- helpers ---------------------------------------------------------------

export class JevError extends Error {
  constructor(status, code, message, field) {
    super(message); this.status = status; this.code = code; this.field = field;
  }
  toJSON() { return { error: { code: this.code, message: this.message, ...(this.field ? { field: this.field } : {}) } }; }
}

export function approxTokens(value) {
  const s = typeof value === 'string' ? value : JSON.stringify(value ?? '');
  return Math.ceil(s.length / 4);
}

export function depthOf(v, d = 0) {
  if (v === null || typeof v !== 'object') return d;
  let m = d + 1;
  for (const x of Array.isArray(v) ? v : Object.values(v)) m = Math.max(m, depthOf(x, d + 1));
  return m;
}

function assertFiniteNumbers(v, path) {
  if (typeof v === 'number' && !Number.isFinite(v)) throw new JevError(400, 'invalid_number', `Non-finite number at ${path}`, path);
  if (v && typeof v === 'object') for (const [k, x] of Object.entries(v)) assertFiniteNumbers(x, `${path}.${k}`);
}

const isInput = (v) => typeof v === 'string' || (v !== null && typeof v === 'object');

export function round(x, decimals) { const f = 10 ** decimals; return Math.round(x * f) / f; }

// ---- validation (TypeSafe rules + AI SDK rules) ----------------------------

/** Validate a native TypeSafe request: { state, model, questions }. Returns resolved model id. */
export function validateNativeRequest(body, { longContext = false } = {}) {
  if (!body || typeof body !== 'object') throw new JevError(422, 'invalid_body', 'Body must be a JSON object');
  const { state, model, questions } = body;
  if (state === undefined) throw new JevError(422, 'missing_field', 'state is required', 'state');
  if (!(typeof state === 'string' || (state !== null && typeof state === 'object'))) throw new JevError(422, 'invalid_field', 'state must be a string, object, or array', 'state');
  if (typeof model !== 'string') throw new JevError(422, 'missing_field', 'model is required', 'model');
  const resolved = MODEL_IDS[model];
  if (!resolved) throw new JevError(422, 'unknown_model', `Unknown model "${model}"`, 'model');
  validateQuestions(questions, 'native');
  for (const [k, v] of Object.entries({ state, questions })) {
    if (depthOf(v) > LIMITS.maxNesting) throw new JevError(400, 'too_deep', `${k} exceeds ${LIMITS.maxNesting} nesting levels`, k);
    assertFiniteNumbers(v, k);
  }
  checkContext(state, questions, longContext);
  return resolved;
}

/** Validate a gateway v4 request: { stateParts, questions }. */
export function validateGatewayRequest(body, { longContext = false } = {}) {
  if (!body || typeof body !== 'object') throw new JevError(422, 'invalid_body', 'Body must be a JSON object');
  const { stateParts, questions } = body;
  if (!Array.isArray(stateParts)) throw new JevError(422, 'invalid_field', 'stateParts must be an array', 'stateParts');
  for (const p of stateParts) {
    if (!p || typeof p !== 'object') throw new JevError(422, 'invalid_field', 'bad state part', 'stateParts');
    if (p.type === 'text' && typeof p.text !== 'string') throw new JevError(422, 'invalid_field', 'text part needs text', 'stateParts');
    if (p.type === 'json' && p.value === undefined) throw new JevError(422, 'invalid_field', 'json part needs value', 'stateParts');
    if (p.type === 'file') throw new JevError(422, 'unsupported', 'Jev accepts text only; file parts are not supported', 'stateParts');
    if (!['text', 'json', 'file'].includes(p.type)) throw new JevError(422, 'invalid_field', `unknown state part type ${p.type}`, 'stateParts');
  }
  validateQuestions(questions, 'gateway');
  const state = gatewayStateToNative(stateParts);
  checkContext(state, questions, longContext);
  return state;
}

export function gatewayStateToNative(stateParts) {
  if (stateParts.length === 1 && stateParts[0].type === 'text') return stateParts[0].text;
  if (stateParts.length === 1 && stateParts[0].type === 'json') return stateParts[0].value;
  return stateParts.map((p) => (p.type === 'text' ? p.text : p.value));
}

function validateQuestions(questions, dialect) {
  if (!questions || typeof questions !== 'object' || Array.isArray(questions) || Object.keys(questions).length === 0)
    throw new JevError(422, 'invalid_field', 'questions must be a non-empty map', 'questions');
  const boolType = dialect === 'gateway' ? 'boolean' : 'noul';
  for (const [id, q] of Object.entries(questions)) {
    const path = `questions.${id}`;
    if (!q || typeof q !== 'object') throw new JevError(422, 'invalid_field', `${path} must be an object`, path);
    if (!isInput(q.instructions)) throw new JevError(422, 'invalid_field', `${path}.instructions must be a string, object, or array`, path);
    switch (q.type) {
      case 'choice': {
        const c = q.criteria;
        if (!c || typeof c !== 'object' || Array.isArray(c) || Object.keys(c).length === 0) throw new JevError(422, 'invalid_field', `${path}.criteria must be a non-empty option map`, path);
        if (Object.keys(c).length > LIMITS.maxChoiceOptions) throw new JevError(422, 'invalid_field', `${path}.criteria has more than ${LIMITS.maxChoiceOptions} options`, path);
        for (const v of Object.values(c)) if (v !== null && !isInput(v)) throw new JevError(422, 'invalid_field', `${path}.criteria values must be string, object, array, or null`, path);
        break;
      }
      case 'score': {
        const c = q.criteria;
        if (!Array.isArray(c) || c.length < LIMITS.minScoreLevels) throw new JevError(422, 'invalid_field', `${path}.criteria must be an ordered array of at least ${LIMITS.minScoreLevels} levels`, path);
        if (c.length > LIMITS.maxScoreLevels) throw new JevError(422, 'invalid_field', `${path}.criteria accepts at most ${LIMITS.maxScoreLevels} levels`, path);
        for (const v of c) if (v !== null && !isInput(v)) throw new JevError(422, 'invalid_field', `${path}.criteria levels must be string, object, or array`, path);
        break;
      }
      case boolType: {
        const c = q.criteria;
        if (c !== undefined) {
          if (!c || typeof c !== 'object' || Array.isArray(c) || Object.keys(c).some((k) => k !== 'true' && k !== 'false'))
            throw new JevError(422, 'invalid_field', `${path}.criteria may only describe true and false`, path);
        }
        break;
      }
      default:
        throw new JevError(422, 'invalid_field', `${path}.type must be choice, score, or ${boolType}`, path);
    }
  }
}

function checkContext(state, questions, longContext) {
  const perReq = longContext ? LIMITS.longContextTokens : LIMITS.perRequestTokens;
  const perQ = longContext ? LIMITS.longContextTokens : LIMITS.perQuestionTokens;
  const stateTok = approxTokens(state);
  let total = stateTok, longest = 0;
  for (const q of Object.values(questions)) { const t = approxTokens(q); total += t; longest = Math.max(longest, t); }
  if (total > perReq) throw new JevError(400, 'max_tokens_exceeded', `Request is ~${total} tokens; per-request context is ${perReq}`);
  if (stateTok + longest > perQ) throw new JevError(400, 'max_tokens_exceeded', `State plus longest question is ~${stateTok + longest} tokens; per-question context is ${perQ}`);
}

// ---- confidence (docs.typesafe.ai/confidence) ------------------------------

export function choiceConfidence(probs) {
  const n = probs.length; if (n < 2) return 1;
  return (Math.max(...probs) - 1 / n) / (1 - 1 / n);
}
/** Score confidence per the docs' demo: 1 - (prob-weighted mean distance from the peak level) / evenSpread, clamped to [0,1]. */
export function scoreConfidence(probs) {
  const n = probs.length; if (n < 2) return 1;
  const peak = probs.indexOf(Math.max(...probs));
  const spread = probs.reduce((s, p, i) => s + p * Math.abs(i - peak), 0);
  const mid = (n - 1) / 2;
  const evenSpread = probs.map((_, i) => Math.abs(i - mid) / n).reduce((a, b) => a + b, 0);
  return Math.max(0, Math.min(1, 1 - spread / evenSpread));
}
export const noulConfidence = (p) => Math.abs(2 * p - 1);

// ---- answer construction ---------------------------------------------------

/** Normalise a raw distribution (array of non-negative weights) to probabilities rounded so they still sum to 1. */
export function normalise(weights, decimals = ROUNDING.probabilityDecimals) {
  const w = weights.map((x) => (Number.isFinite(x) && x > 0 ? x : 0));
  const sum = w.reduce((a, b) => a + b, 0) || 1;
  const p = w.map((x) => round(x / sum, decimals));
  // fix rounding drift on the largest entry so the distribution sums to exactly 1
  const drift = round(1 - p.reduce((a, b) => a + b, 0), decimals);
  const i = p.indexOf(Math.max(...p));
  p[i] = round(p[i] + drift, decimals);
  return p;
}

/** Build the native TypeSafe answer for a question from a raw weight vector (or P(yes) for noul). */
export function buildNativeAnswer(question, raw) {
  switch (question.type) {
    case 'noul': case 'boolean': {
      const p = round(Math.max(0, Math.min(1, raw)), ROUNDING.probabilityDecimals);
      return question.type === 'noul' ? { type: 'noul', noul: p } : { type: 'boolean', probability: p };
    }
    case 'choice': {
      const keys = Object.keys(question.criteria);
      const p = normalise(raw);
      const probabilities = Object.fromEntries(keys.map((k, i) => [k, p[i]]));
      const choice = keys[p.indexOf(Math.max(...p))];
      return { type: 'choice', choice, probabilities, confidence: round(choiceConfidence(p), ROUNDING.probabilityDecimals) };
    }
    case 'score': {
      const p = normalise(raw);
      const score = round(p.reduce((s, x, i) => s + i * x, 0), ROUNDING.scoreDecimals);
      const legend = Object.fromEntries(question.criteria.map((d, i) => [String(i), typeof d === 'string' ? d : JSON.stringify(d)]));
      const probabilities = Object.fromEntries(p.map((x, i) => [String(i), x]));
      return { type: 'score', score, legend, probabilities, confidence: round(scoreConfidence(p), ROUNDING.probabilityDecimals) };
    }
  }
}

/** Convert a native answer to the gateway v4 answer shape (noul→boolean, drop legend/confidence). */
export function toGatewayAnswer(a) {
  switch (a.type) {
    case 'noul': return { type: 'boolean', probability: a.noul };
    case 'boolean': return a;
    case 'choice': return { type: 'choice', choice: a.choice, probabilities: a.probabilities };
    case 'score': return { type: 'score', score: a.score, probabilities: a.probabilities };
    default: return { type: 'refusal' };
  }
}

/** Convert gateway questions (type "boolean") to native ("noul") for the backend. */
export function gatewayQuestionsToNative(questions) {
  return Object.fromEntries(Object.entries(questions).map(([id, q]) => [id, q.type === 'boolean' ? { ...q, type: 'noul' } : q]));
}

export function usageFor(state, questions) {
  let input = approxTokens(state);
  for (const q of Object.values(questions)) input += approxTokens(q);
  // Jev bills input only; output tokens are a small fixed cost per question in the public examples (~18-34)
  return { input_tokens: input, output_tokens: 16 + 6 * Object.keys(questions).length };
}
