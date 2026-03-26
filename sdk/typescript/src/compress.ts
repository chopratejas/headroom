import { HeadroomClient } from "./client.js";
import type {
  OpenAIMessage,
  CompressResult,
  CompressOptions,
} from "./types.js";

/**
 * Compress an array of messages using the Headroom proxy or cloud.
 *
 * This is the main entry point for the SDK. It accepts OpenAI-format messages,
 * sends them to the Headroom compression pipeline, and returns compressed
 * messages with stats.
 *
 * @example
 * ```typescript
 * import { compress } from 'headroom-ai';
 *
 * const result = await compress(messages, { model: 'gpt-4o' });
 * console.log(`Saved ${result.tokensSaved} tokens`);
 * ```
 */
export async function compress(
  messages: OpenAIMessage[],
  options: CompressOptions = {},
): Promise<CompressResult> {
  const { client: providedClient, model, ...clientOptions } = options;

  const client = providedClient ?? new HeadroomClient(clientOptions);

  return client.compress(messages, { model });
}
