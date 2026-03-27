import { HeadroomClient } from "./client.js";
import type {
  CompressResult,
  CompressOptions,
} from "./types.js";
import { detectFormat, toOpenAI, fromOpenAI } from "./utils/format.js";

/**
 * Compress an array of messages using the Headroom proxy or cloud.
 *
 * Accepts messages in any format: OpenAI, Anthropic, Vercel AI SDK, or Google Gemini.
 * Detects the format automatically, compresses via the proxy, and returns
 * compressed messages in the same format as the input.
 *
 * @example
 * ```typescript
 * import { compress } from 'headroom-ai';
 *
 * // Works with any message format:
 * const result = await compress(openaiMessages, { model: 'gpt-4o' });
 * const result = await compress(anthropicMessages, { model: 'claude-sonnet-4-5-20250929' });
 * const result = await compress(vercelMessages, { model: 'gpt-4o' });
 * const result = await compress(geminiContents, { model: 'gemini-2.0-flash' });
 *
 * // result.messages is in the same format as input
 * ```
 */
export async function compress(
  messages: any[],
  options: CompressOptions = {},
): Promise<CompressResult> {
  const { client: providedClient, model, ...clientOptions } = options;

  // Detect input format
  const inputFormat = detectFormat(messages);

  // Convert to OpenAI format (the proxy's lingua franca)
  const openaiMessages = toOpenAI(messages);

  // Compress via proxy
  const client = providedClient ?? new HeadroomClient(clientOptions);
  const result = await client.compress(openaiMessages, { model });

  // Convert compressed messages back to original format
  const outputMessages = fromOpenAI(result.messages, inputFormat);

  return {
    ...result,
    messages: outputMessages,
  };
}
