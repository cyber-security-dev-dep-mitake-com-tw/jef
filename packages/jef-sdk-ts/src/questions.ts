import type { BooleanQuestion, ChoiceQuestion, NoulQuestion, ScoreQuestion } from './types.js';

/**
 * A choice question. Key order in the object literal is the option order the
 * model sees, and the return type keeps the literal keys so
 * `answer.probabilities` is typed with your option names rather than `string`.
 */
export function choice<const C extends Record<string, string | null>>(
  instructions: string,
  criteria: C,
): ChoiceQuestion & { criteria: C } {
  const keys = Object.keys(criteria);
  if (keys.length < 2) {
    throw new Error(`a choice question needs at least two options, got ${keys.length}`);
  }
  return { type: 'choice', instructions, criteria };
}

/**
 * A score question. Levels are ordinal and must be lowest-first.
 *
 * The answer is the expectation over this ordering, so passing them in reverse
 * inverts the scale silently instead of failing — hence the explicit note here
 * rather than a runtime check that cannot tell the two apart.
 */
export function score<const L extends readonly string[]>(
  instructions: string,
  levels: L,
): ScoreQuestion & { criteria: L } {
  if (levels.length < 2) {
    throw new Error(`a score question needs at least two ordered levels, got ${levels.length}`);
  }
  if (new Set(levels).size !== levels.length) {
    throw new Error('score levels must be distinct');
  }
  return { type: 'score', instructions, criteria: levels };
}

/** Yes/no, in TypeSafe's native spelling. The answer comes back as `noul`. */
export function noul(
  instructions: string,
  criteria?: { true?: string; false?: string },
): NoulQuestion {
  return criteria && (criteria.true !== undefined || criteria.false !== undefined)
    ? { type: 'noul', instructions, criteria }
    : { type: 'noul', instructions };
}

/**
 * The same primitive in the Vercel AI SDK's spelling, so code ported from
 * `experimental_evaluate` reads unchanged. The answer comes back as
 * `probability` rather than `noul`.
 */
export function boolean(
  instructions: string,
  criteria?: { true?: string; false?: string },
): BooleanQuestion {
  return criteria && (criteria.true !== undefined || criteria.false !== undefined)
    ? { type: 'boolean', instructions, criteria }
    : { type: 'boolean', instructions };
}
