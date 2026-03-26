/**
 * Convert between Vercel AI SDK ModelMessage format and OpenAI message format.
 *
 * Vercel uses typed content parts (TextPart, ToolCallPart, ToolResultPart, etc.)
 * OpenAI uses { role, content, tool_calls, tool_call_id }.
 *
 * We use loose types (any) for Vercel messages to avoid forcing users to install
 * @ai-sdk/provider. The adapter (vercel-ai.ts) provides the typed wrapper.
 */

import type {
  OpenAIMessage,
  AssistantMessage,
  ToolCall,
} from "../types.js";

/* eslint-disable @typescript-eslint/no-explicit-any */
type VercelMessage = any;

/**
 * Convert Vercel AI SDK ModelMessage[] to OpenAI message format.
 *
 * Handles: system, user (text + image), assistant (text + tool-call),
 * tool (tool-result). Skips reasoning and file parts (non-compressible).
 */
export function vercelToOpenAI(messages: VercelMessage[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      result.push({ role: "system", content: msg.content });
      continue;
    }

    if (msg.role === "user") {
      const parts = Array.isArray(msg.content) ? msg.content : [msg.content];
      const textParts = parts.filter((p: any) => p.type === "text");
      const imageParts = parts.filter((p: any) => p.type === "image");

      if (imageParts.length === 0 && textParts.length > 0) {
        // Text-only: flatten to string
        result.push({
          role: "user",
          content: textParts.map((p: any) => p.text).join(""),
        });
      } else {
        // Mixed content: use content parts array
        const openaiParts = parts
          .filter((p: any) => p.type === "text" || p.type === "image")
          .map((p: any) => {
            if (p.type === "text") {
              return { type: "text" as const, text: p.text };
            }
            if (p.type === "image") {
              const url =
                p.image instanceof URL ? p.image.toString() : String(p.image);
              return { type: "image_url" as const, image_url: { url } };
            }
            return { type: "text" as const, text: "" };
          });
        result.push({ role: "user", content: openaiParts });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const parts = Array.isArray(msg.content) ? msg.content : [];
      const textParts = parts.filter((p: any) => p.type === "text");
      const toolCallParts = parts.filter((p: any) => p.type === "tool-call");

      const content =
        textParts.length > 0
          ? textParts.map((p: any) => p.text).join("")
          : null;

      const openaiMsg: AssistantMessage = { role: "assistant", content };

      if (toolCallParts.length > 0) {
        openaiMsg.tool_calls = toolCallParts.map(
          (p: any): ToolCall => ({
            id: p.toolCallId,
            type: "function",
            function: {
              name: p.toolName,
              arguments: JSON.stringify(p.args),
            },
          }),
        );
      }

      result.push(openaiMsg);
      continue;
    }

    if (msg.role === "tool") {
      const parts = Array.isArray(msg.content) ? msg.content : [];
      for (const part of parts) {
        if (part.type === "tool-result") {
          result.push({
            role: "tool",
            content:
              typeof part.result === "string"
                ? part.result
                : JSON.stringify(part.result),
            tool_call_id: part.toolCallId,
          });
        }
      }
      continue;
    }
  }

  return result;
}

/**
 * Convert OpenAI message format back to Vercel AI SDK ModelMessage[].
 */
export function openAIToVercel(messages: OpenAIMessage[]): VercelMessage[] {
  const result: VercelMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      result.push({ role: "system", content: msg.content });
      continue;
    }

    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({
          role: "user",
          content: [{ type: "text", text: msg.content }],
        });
      } else if (Array.isArray(msg.content)) {
        const parts = msg.content.map((p) => {
          if (p.type === "text") return { type: "text", text: p.text };
          if (p.type === "image_url") {
            return { type: "image", image: new URL(p.image_url.url) };
          }
          return { type: "text", text: "" };
        });
        result.push({ role: "user", content: parts });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const parts: any[] = [];
      if (msg.content) {
        parts.push({ type: "text", text: msg.content });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          parts.push({
            type: "tool-call",
            toolCallId: tc.id,
            toolName: tc.function.name,
            args: JSON.parse(tc.function.arguments),
          });
        }
      }
      result.push({ role: "assistant", content: parts });
      continue;
    }

    if (msg.role === "tool") {
      let parsedResult: any;
      try {
        parsedResult = JSON.parse(msg.content);
      } catch {
        parsedResult = msg.content;
      }
      result.push({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: msg.tool_call_id,
            toolName: "unknown",
            result: parsedResult,
          },
        ],
      });
      continue;
    }
  }

  return result;
}
