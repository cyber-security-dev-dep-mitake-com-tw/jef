import type {
  IExecuteFunctions,
  IDataObject,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  IHttpRequestOptions,
} from 'n8n-workflow';
// NodeConnectionTypes is the value export; NodeConnectionType is type-only in
// current n8n-workflow, so importing it as a value fails to compile.
import { NodeApiError, NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';

/** A state is text, or an object/array sent as structure. */
function parseState(raw: string): unknown {
  const text = (raw ?? '').trim();
  if (text.startsWith('{') || text.startsWith('[')) {
    try {
      return JSON.parse(text);
    } catch {
      // Genuinely text that happens to start with a brace.
      return text;
    }
  }
  return text;
}

/** `key=description` per line, preserving order — order is the option index. */
function parseOptions(raw: string): Record<string, string | null> {
  const criteria: Record<string, string | null> = {};
  for (const line of (raw ?? '').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const index = trimmed.indexOf('=');
    const key = (index === -1 ? trimmed : trimmed.slice(0, index)).trim();
    const description = index === -1 ? '' : trimmed.slice(index + 1).trim();
    if (key) criteria[key] = description || null;
  }
  return criteria;
}

function parseLevels(raw: string): string[] {
  return (raw ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

export class Jef implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'JEF',
    name: 'jef',
    icon: 'file:jef.svg',
    group: ['transform'],
    version: 1,
    subtitle: '={{ $parameter["operation"] }}',
    description:
      'Typed decisions with calibrated confidence. Route, score and judge without generating text.',
    defaults: { name: 'JEF' },
    inputs: [NodeConnectionTypes.Main],
    outputs: [NodeConnectionTypes.Main],
    credentials: [{ name: 'jefApi', required: true }],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        default: 'runScene',
        options: [
          {
            name: 'Run Scene',
            value: 'runScene',
            description:
              'Run a whole playbook and return its decision trace. The state is read once for every layer.',
            action: 'Run a scene',
          },
          {
            name: 'Ask Choice',
            value: 'askChoice',
            description: 'Pick one of the options you supply',
            action: 'Ask a choice question',
          },
          {
            name: 'Ask Score',
            value: 'askScore',
            description: 'Rate against ordered levels; the score may land between them',
            action: 'Ask a score question',
          },
          {
            name: 'Ask Yes/No',
            value: 'askYesNo',
            description: 'Probability that a statement holds',
            action: 'Ask a yes no question',
          },
          {
            name: 'Ask Many',
            value: 'askMany',
            description:
              'Several questions in one call. They run in parallel against the same state, so extra questions are nearly free.',
            action: 'Ask several questions',
          },
          {
            name: 'Health',
            value: 'health',
            description:
              'Server status. Gate on this: an uncalibrated server produces thresholds that do not mean what they look like.',
            action: 'Check server health',
          },
        ],
      },

      {
        displayName: 'Scene',
        name: 'scene',
        type: 'string',
        default: 'incident-triage',
        required: true,
        displayOptions: { show: { operation: ['runScene'] } },
        description: 'Scene id as registered on the server',
      },

      {
        displayName: 'State',
        name: 'state',
        type: 'string',
        typeOptions: { rows: 6 },
        default: '',
        required: true,
        displayOptions: { hide: { operation: ['health'] } },
        description:
          'The evidence to judge. Plain text, or JSON for an object or array. An array is one shared state, never a batch.',
      },

      {
        displayName: 'Question',
        name: 'instructions',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['askChoice', 'askScore', 'askYesNo'] } },
        placeholder: '這則告警應由哪一個團隊處理？',
      },

      {
        displayName: 'Options',
        name: 'options',
        type: 'string',
        typeOptions: { rows: 5 },
        default: 'soc=一般資安監控事件\nappsec=應用程式漏洞與程式碼相關\ninfra=基礎設施、網路、主機層',
        required: true,
        displayOptions: { show: { operation: ['askChoice'] } },
        description:
          'One key=description per line. The description is what the model reads, so write it properly.',
      },

      {
        displayName: 'Levels (Lowest First)',
        name: 'levels',
        type: 'string',
        typeOptions: { rows: 5 },
        default: '資訊\n低\n中\n高\n危急',
        required: true,
        displayOptions: { show: { operation: ['askScore'] } },
        description:
          'One level per line, lowest first. Order is load-bearing: the answer is the expectation over it, so reversing the list inverts the scale silently.',
      },

      {
        displayName: 'If True',
        name: 'ifTrue',
        type: 'string',
        default: '',
        displayOptions: { show: { operation: ['askYesNo'] } },
        description: 'Optional description of what a yes means',
      },
      {
        displayName: 'If False',
        name: 'ifFalse',
        type: 'string',
        default: '',
        displayOptions: { show: { operation: ['askYesNo'] } },
        description: 'Optional description of what a no means',
      },

      {
        displayName: 'Questions (JSON)',
        name: 'questions',
        type: 'json',
        default:
          '{\n  "urgent": {"type": "noul", "instructions": "是否需要立即處理？"},\n  "team": {"type": "choice", "instructions": "誰處理？", "criteria": {"soc": "監控", "infra": "網路"}}\n}',
        required: true,
        displayOptions: { show: { operation: ['askMany'] } },
        description: 'A JSON object of question id to question definition',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const credentials = await this.getCredentials('jefApi');
    const baseUrl = String(credentials.baseUrl ?? '').replace(/\/+$/, '');
    const out: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      const operation = String(this.getNodeParameter('operation', i));

      try {
        let path: string;
        let body: IDataObject | undefined;
        let method: 'GET' | 'POST' = 'POST';

        if (operation === 'health') {
          path = '/healthz';
          method = 'GET';
        } else {
          const state = parseState(String(this.getNodeParameter('state', i)));

          if (operation === 'runScene') {
            const scene = String(this.getNodeParameter('scene', i));
            path = `/v1/scenes/${encodeURIComponent(scene)}:evaluate`;
            body = { state } as IDataObject;
          } else {
            path = '/v1/systemone';
            let questions: IDataObject;

            if (operation === 'askChoice') {
              const criteria = parseOptions(String(this.getNodeParameter('options', i)));
              if (Object.keys(criteria).length < 2) {
                throw new NodeOperationError(
                  this.getNode(),
                  'A choice needs at least two "key=description" options',
                  { itemIndex: i },
                );
              }
              questions = {
                answer: {
                  type: 'choice',
                  instructions: String(this.getNodeParameter('instructions', i)),
                  criteria,
                },
              };
            } else if (operation === 'askScore') {
              const levels = parseLevels(String(this.getNodeParameter('levels', i)));
              if (levels.length < 2) {
                throw new NodeOperationError(
                  this.getNode(),
                  'A score needs at least two ordered levels, lowest first',
                  { itemIndex: i },
                );
              }
              if (new Set(levels).size !== levels.length) {
                throw new NodeOperationError(this.getNode(), 'Score levels must be distinct', {
                  itemIndex: i,
                });
              }
              questions = {
                answer: {
                  type: 'score',
                  instructions: String(this.getNodeParameter('instructions', i)),
                  criteria: levels,
                },
              };
            } else if (operation === 'askYesNo') {
              const ifTrue = String(this.getNodeParameter('ifTrue', i, ''));
              const ifFalse = String(this.getNodeParameter('ifFalse', i, ''));
              const question: IDataObject = {
                type: 'noul',
                instructions: String(this.getNodeParameter('instructions', i)),
              };
              if (ifTrue || ifFalse) {
                const criteria: IDataObject = {};
                if (ifTrue) criteria.true = ifTrue;
                if (ifFalse) criteria.false = ifFalse;
                question.criteria = criteria;
              }
              questions = { answer: question };
            } else if (operation === 'askMany') {
              // The json parameter type arrives as a string from the editor and as
              // an object when set by an expression, so both are handled.
              const raw: unknown = this.getNodeParameter('questions', i);
              const parsed: unknown = typeof raw === 'string' ? JSON.parse(raw) : raw;
              if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
                throw new NodeOperationError(
                  this.getNode(),
                  'Questions must be a JSON object of id to question',
                  { itemIndex: i },
                );
              }
              questions = parsed as IDataObject;
            } else {
              throw new NodeOperationError(this.getNode(), `Unknown operation: ${operation}`, {
                itemIndex: i,
              });
            }

            body = { state, questions } as IDataObject;
          }
        }

        const options: IHttpRequestOptions = {
          method,
          url: `${baseUrl}${path}`,
          json: true,
          // A cold CPU encode of a long state is not instant, and cutting it off
          // would look like a model failure rather than a timeout.
          timeout: 120_000,
          ...(body ? { body } : {}),
        };

        const response = (await this.helpers.httpRequestWithAuthentication.call(
          this,
          'jefApi',
          options,
        )) as IDataObject;

        out.push({ json: response, pairedItem: { item: i } });
      } catch (error) {
        if (this.continueOnFail()) {
          out.push({
            json: { error: (error as Error).message },
            pairedItem: { item: i },
          });
          continue;
        }
        if (error instanceof NodeOperationError) throw error;
        throw new NodeApiError(this.getNode(), error as never, { itemIndex: i });
      }
    }

    return [out];
  }
}
