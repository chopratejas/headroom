import { compress } from "../compress.js";
import type { CompressOptions, CompressResult } from "../types.js";
import { vercelToOpenAI, openAIToVercel } from "../utils/format.js";

/* eslint-disable @typescript-eslint/no-explicit-any */
type VercelMessage = any;

/**
 * Vercel AI SDK LanguageModelV3Middleware that compresses messages
 * before they reach the LLM.
 *
 * @example
 * ```typescript
 * import { headroomMiddleware } from 'headroom-ai/vercel-ai';
 * import { wrapLanguageModel } from 'ai';
 * import { openai } from '@ai-sdk/openai';
 *
 * const model = wrapLanguageModel({
 *   model: openai('gpt-4o'),
 *   middleware: headroomMiddleware(),
 * });
 * ```
 */
export function headroomMiddleware(options: CompressOptions = {}) {
  return {
    transformParams: async ({
      params,
    }: {
      params: any;
      model: any;
      type: string;
    }) => {
      const prompt: VercelMessage[] = params.prompt;
      if (!prompt || prompt.length === 0) return params;

      const model = options.model ?? params.modelId ?? "gpt-4o";

      // Convert Vercel format → OpenAI format
      const openaiMessages = vercelToOpenAI(prompt);

      // Compress via Headroom
      const result = await compress(openaiMessages, { ...options, model });

      if (!result.compressed) return params;

      // Convert back to Vercel format
      const compressedPrompt = openAIToVercel(result.messages);

      return { ...params, prompt: compressedPrompt };
    },
  };
}

/**
 * Standalone: compress Vercel AI SDK ModelMessage[] directly.
 * Returns compressed messages in Vercel format + compression stats.
 */
export async function compressVercelMessages(
  messages: VercelMessage[],
  options: CompressOptions = {},
): Promise<CompressResult & { messages: VercelMessage[] }> {
  const openaiMessages = vercelToOpenAI(messages);
  const result = await compress(openaiMessages, options);
  const vercelMessages = openAIToVercel(result.messages);

  return {
    ...result,
    messages: vercelMessages,
  };
}
