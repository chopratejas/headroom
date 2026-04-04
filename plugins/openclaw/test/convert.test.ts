import { describe, expect, it } from "vitest";
import { openAIToAgent, type OpenAIMessage } from "../src/convert";

describe("openAIToAgent", () => {
  it("emits toolResult content as blocks so transports can safely filter", () => {
    const messages: OpenAIMessage[] = [
      {
        role: "tool",
        content: "tool output",
        tool_call_id: "call_123",
      },
    ];

    const result = openAIToAgent(messages);
    const toolResult = result[0] as {
      role: string;
      content: Array<{ type: string; text?: string }>;
      toolCallId: string;
      tool_use_id: string;
    };

    expect(toolResult.role).toBe("toolResult");
    expect(Array.isArray(toolResult.content)).toBe(true);
    expect(toolResult.content).toEqual([{ type: "text", text: "tool output" }]);
    expect(toolResult.toolCallId).toBe("call_123");
    expect(toolResult.tool_use_id).toBe("call_123");
  });
});
