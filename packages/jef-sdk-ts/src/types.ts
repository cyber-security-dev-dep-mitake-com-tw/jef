/**
 * The `/v1/systemone` wire contract.
 *
 * Two spellings exist in the wild for the yes/no primitive: TypeSafe's native
 * API calls it `noul` and returns `{ noul }`, while Vercel's AI SDK calls it
 * `boolean` and returns `{ probability }`. JEF accepts both and answers in
 * whichever was asked, so this module models both rather than picking one and
 * making the other a special case at every call site.
 */

/** Pick exactly one option. Key order is the order the model sees. */
export interface ChoiceQuestion {
  type: 'choice';
  instructions: string;
  /** Option key to description. `null` means "no description". */
  criteria: Record<string, string | null>;
}

/**
 * Rate against ordered levels, lowest first.
 *
 * Order is load-bearing: the answer is the expectation over this ordering, so
 * reversing the levels inverts the scale silently rather than erroring.
 */
export interface ScoreQuestion {
  type: 'score';
  instructions: string;
  /** Readonly so `['低','中','高'] as const` is accepted and stays narrow. */
  criteria: readonly string[];
}

/** Yes/no, in TypeSafe's native spelling. */
export interface NoulQuestion {
  type: 'noul';
  instructions: string;
  criteria?: { true?: string; false?: string };
}

/** The same primitive, in the Vercel AI SDK's spelling. */
export interface BooleanQuestion {
  type: 'boolean';
  instructions: string;
  criteria?: { true?: string; false?: string };
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion | BooleanQuestion;

/** A state is one document, never a batch — an array is a single shared state. */
export type State = string | Record<string, unknown> | unknown[];

export interface ChoiceAnswer {
  type: 'choice';
  choice: string;
  probabilities: Record<string, number>;
  /**
   * How peaked the distribution is: `(n * peak - 1) / (n - 1)`.
   *
   * Not the probability of being correct. For that, run a scene — its gates
   * read `p_correct`, which is the calibrated mapping of this number onto
   * observed correctness.
   */
  confidence: number;
}

export interface ScoreAnswer {
  type: 'score';
  /** Continuous, and may land between levels. */
  score: number;
  legend: string[];
  probabilities: Record<string, number>;
  confidence: number;
}

export interface NoulAnswer {
  type: 'noul';
  noul: number;
  confidence: number;
}

export interface BooleanAnswer {
  type: 'boolean';
  probability: number;
  confidence: number;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer | BooleanAnswer;

export interface Usage {
  inputTokens: number;
  /** Structurally zero. A System One model generates nothing. */
  outputTokens: number;
  totalTokens: number;
}

export interface EvaluateResult<Q extends Record<string, Question> = Record<string, Question>> {
  model: string;
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
  usage: Usage;
  warnings?: string[];
}

/** Maps a question type to the answer type it produces. */
export type AnswerFor<Q extends Question> = Q extends ChoiceQuestion
  ? ChoiceAnswer
  : Q extends ScoreQuestion
    ? ScoreAnswer
    : Q extends NoulQuestion
      ? NoulAnswer
      : Q extends BooleanQuestion
        ? BooleanAnswer
        : Answer;

// --------------------------------------------------------------------------- //
// Scenes
// --------------------------------------------------------------------------- //

export interface QuestionTrace {
  id: string;
  kind: 'choice' | 'score' | 'noul';
  instructions: string;
  allowed_answers: string[];
  answer: Answer;
  confidence: number;
  /**
   * Calibrated probability the answer is correct.
   *
   * `null` means no correctness map was fitted for this bucket — which is not
   * the same as a low probability, and a gate must be able to tell them apart.
   */
  p_correct: number | null;
  /** More than one entry means the model could not separate the candidates. */
  prediction_set: string[];
}

export interface GateTrace {
  index: number;
  condition: string | null;
  is_else: boolean;
  fired: boolean;
  then: string | null;
  because: string | null;
  error?: string | null;
}

export interface LayerTrace {
  id: string;
  description: string | null;
  questions: QuestionTrace[];
  gates: GateTrace[];
  outcome: string;
}

export interface SceneTrace {
  scene: string;
  version: number;
  model: string;
  verdict: 'decided' | 'fallthrough';
  action: string | null;
  human_review: boolean;
  action_params: Record<string, unknown>;
  layers: LayerTrace[];
  /** Layers never reached. Distinguishes "not asked" from "inconclusive". */
  layers_skipped: string[];
  state_encodes: number;
  questions_asked: number;
  calibrated: boolean;
  warnings?: string[];
}

export interface SceneSummary {
  id: string;
  version: number;
  description: string | null;
  layers: number;
  questions: number;
}

export interface ModelInfo {
  id: string;
  object: string;
  backbone: string;
  head: string;
  calibrated: boolean;
  question_types: string[];
  runtime?: string;
}

export interface Health {
  status: string;
  model: string;
  calibrated: boolean;
  /** True when the server is running the semantically meaningless test stub. */
  test_backbone: boolean;
  runtime?: string;
  version?: string;
}

export interface Limits {
  max_questions_per_request: number;
  max_state_chars?: number;
  max_state_bytes?: number;
}
