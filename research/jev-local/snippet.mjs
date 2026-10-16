// snippet.mjs — verbatim TypeScript snippet from https://vercel.com/ai-gateway/models/jev (as ESM JS)
import { experimental_decide as decide } from 'ai';

const result = await decide({
  model: 'typesafe-ai/jev',
  state: 'The support agent issued a full refund to the customer.',
  questions: {
    refunded: {
      type: 'boolean',
      instructions: 'Was a refund issued?',
    },
  },
})
console.log(JSON.stringify(result.answers), result.response.modelId);
