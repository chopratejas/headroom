/**
 * Convert between OpenClaw's AgentMessage format and OpenAI message format.
 *
 * AgentMessage uses:
 *   role: "user" | "assistant" | "toolResult"
 *   content: string | ContentBlock[]
 *
 * OpenAI uses:
 *   role: "user" | "assistant" | "system" | "tool"
 *   content: string
 *   tool_calls?: ToolCall[]
 *   tool_call_id?: string
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface OpenAIMessage {
  role: string;
  content: string | null;
  tool_calls?: any[];
  tool_call_id?: string;
  name?: string;
}

/**
 * Convert AgentMessage[] to OpenAI message format for compression.
 */
export function agentToOpenAI(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    const role = msg.role;

    if (role === "system") {
      result.push({
        role: "system",
        content: typeof msg.content === "string" ? msg.content : extractText(msg.content),
      });
      continue;
    }

    if (role === "user") {
      result.push({
        role: "user",
        content: typeof msg.content === "string" ? msg.content : extractText(msg.content),
      });
      continue;
    }

    if (role === "assistant") {
      const content = msg.content;
      if (typeof content === "string") {
        result.push({ role: "assistant", content });
        continue;
      }

      // Content blocks: extract text and tool call blocks.
      // OpenClaw uses `toolCall`; some adapters still emit legacy `tool_use`.
      if (Array.isArray(content)) {
        const textParts: string[] = [];
        const toolCalls: any[] = [];

        for (const block of content) {
          if (typeof block === "string") {
            textParts.push(block);
          } else if (block.type === "text") {
            textParts.push(block.text);
          } else if (block.type === "tool_use" || block.type === "toolCall") {
            const args =
              block.type === "toolCall"
                ? block.arguments
                : block.input;
            toolCalls.push({
              id: block.id,
              type: "function",
              function: {
                name: block.name,
                arguments:
                  typeof args === "string"
                    ? args
                    : JSON.stringify(args ?? {}),
              },
            });
          }
        }

        const openaiMsg: OpenAIMessage = {
          role: "assistant",
          content: textParts.length > 0 ? textParts.join("") : null,
        };
        if (toolCalls.length > 0) {
          openaiMsg.tool_calls = toolCalls;
        }
        result.push(openaiMsg);
      }
      continue;
    }

    if (role === "toolResult" || role === "tool_result") {
      const content =
        typeof msg.content === "string"
          ? msg.content
          : Array.isArray(msg.content)
            ? extractText(msg.content)
            : JSON.stringify(msg.content);

      result.push({
        role: "tool",
        content,
        tool_call_id: msg.tool_use_id ?? msg.toolCallId ?? msg.id ?? "unknown",
      });
      continue;
    }

    // Fallback: pass through as user message
    result.push({
      role: "user",
      content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
    });
  }

  return result;
}

/**
 * Convert compressed OpenAI messages back to AgentMessage format.
 */
export function openAIToAgent(messages: OpenAIMessage[]): any[] {
  const result: any[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      result.push({
        role: "system",
        content: msg.content ?? "",
        timestamp: Date.now(),
      });
      continue;
    }

    if (msg.role === "user") {
      result.push({
        role: "user",
        content: msg.content ?? "",
        timestamp: Date.now(),
      });
      continue;
    }

    if (msg.role === "assistant") {
      const blocks: any[] = [];
      if (msg.content) {
        blocks.push({ type: "text", text: msg.content });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          let input: any;
          try {
            input = JSON.parse(tc.function.arguments);
          } catch {
            input = tc.function.arguments ?? {};
          }
          // Emit OpenClaw-native block shape so downstream transports keep call linkage.
          blocks.push({
            type: "toolCall",
            id: tc.id,
            name: tc.function.name,
            arguments: input,
          });
        }
      }
      // OpenClaw's Pi agent expects content to always be an array for assistant messages
      // (it calls .flatMap() on it). Never flatten to a string.
      result.push({
        role: "assistant",
        content: blocks,
        timestamp: Date.now(),
      });
      continue;
    }

    if (msg.role === "tool") {
      const textContent =
        typeof msg.content === "string"
          ? msg.content
          : msg.content == null
            ? ""
            : JSON.stringify(msg.content);
      const toolCallId = msg.tool_call_id ?? "unknown";
      result.push({
        role: "toolResult",
        // OpenClaw transport layers expect toolResult content blocks, not a raw string.
        content: [{ type: "text", text: textContent }],
        toolCallId,
        tool_use_id: toolCallId,
        timestamp: Date.now(),
      });
      continue;
    }
  }

  return result;
}

/**
 * Extract text from content blocks.
 */
function extractText(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return JSON.stringify(content);

  return content
    .map((block: any) => {
      if (typeof block === "string") return block;
      if (block.type === "text") return block.text;
      if (block.type === "tool_result") {
        return typeof block.content === "string" ? block.content : JSON.stringify(block.content);
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}
