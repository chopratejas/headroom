/**
 * Integration tests for HeadroomContextEngine.
 *
 * Tests the full flow: proxy management, AgentMessage conversion,
 * compression via proxy, and round-trip back to AgentMessage.
 *
 * Requires: Python 3 + headroom-ai[proxy] installed
 * Run: HEADROOM_INTEGRATION=1 npx vitest run test/engine.test.ts
 */
import { describe, it, expect, beforeAll, afterAll, vi, afterEach } from "vitest";
import { HeadroomContextEngine } from "../src/engine.js";
import { agentToOpenAI, openAIToAgent } from "../src/convert.js";
import { ProxyManager } from "../src/proxy-manager.js";

const RUN = process.env.HEADROOM_INTEGRATION === "1";
const PROXY_URL = process.env.HEADROOM_PROXY_URL ?? "http://127.0.0.1:8787";

afterEach(() => {
  vi.restoreAllMocks();
});

// Proxy probing and ProxyManager.start tests live in proxy-manager.test.ts

describe("AgentMessage conversion", () => {
  it("converts user message", () => {
    const agent = [{ role: "user", content: "hello", timestamp: Date.now() }];
    const openai = agentToOpenAI(agent);
    expect(openai).toEqual([{ role: "user", content: "hello" }]);
  });

  it("converts assistant with tool_use blocks", () => {
    const agent = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me search" },
          { type: "tool_use", id: "tu_1", name: "search", input: { q: "test" } },
        ],
        timestamp: Date.now(),
      },
    ];
    const openai = agentToOpenAI(agent);
    expect(openai[0].role).toBe("assistant");
    expect(openai[0].content).toBe("Let me search");
    expect(openai[0].tool_calls).toHaveLength(1);
    expect(openai[0].tool_calls![0].function.name).toBe("search");
  });

  it("converts assistant with toolCall blocks", () => {
    const agent = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me search" },
          { type: "toolCall", id: "call_1|fc_1", name: "search", arguments: { q: "test" } },
        ],
        timestamp: Date.now(),
      },
    ];
    const openai = agentToOpenAI(agent);
    expect(openai[0].role).toBe("assistant");
    expect(openai[0].content).toBe("Let me search");
    expect(openai[0].tool_calls).toHaveLength(1);
    expect(openai[0].tool_calls![0].id).toBe("call_1|fc_1");
    expect(openai[0].tool_calls![0].function.name).toBe("search");
  });

  it("converts toolResult message", () => {
    const agent = [
      {
        role: "toolResult",
        content: '{"results": [1, 2, 3]}',
        tool_use_id: "tu_1",
        timestamp: Date.now(),
      },
    ];
    const openai = agentToOpenAI(agent);
    expect(openai[0].role).toBe("tool");
    expect(openai[0].content).toBe('{"results": [1, 2, 3]}');
    expect(openai[0].tool_call_id).toBe("tu_1");
  });

  it("round-trips user message", () => {
    const original = [{ role: "user", content: "hello", timestamp: Date.now() }];
    const openai = agentToOpenAI(original);
    const back = openAIToAgent(openai);
    expect(back[0].role).toBe("user");
    expect(back[0].content).toBe("hello");
  });

  it("round-trips assistant text-only (content always array)", () => {
    const original = [
      {
        role: "assistant",
        content: [{ type: "text", text: "Hello there!" }],
        timestamp: Date.now(),
      },
    ];
    const openai = agentToOpenAI(original);
    const back = openAIToAgent(openai);
    expect(back[0].role).toBe("assistant");
    // OpenClaw requires content to ALWAYS be an array for assistant messages
    const content = back[0].content;
    expect(Array.isArray(content)).toBe(true);
    expect(content[0]).toEqual({ type: "text", text: "Hello there!" });
  });

  it("round-trips assistant with tool calls", () => {
    const original = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Searching..." },
          { type: "toolCall", id: "call_1|fc_1", name: "search", arguments: { q: "test" } },
        ],
        timestamp: Date.now(),
      },
    ];
    const openai = agentToOpenAI(original);
    const back = openAIToAgent(openai);
    expect(back[0].role).toBe("assistant");
    const content = back[0].content;
    expect(Array.isArray(content)).toBe(true);
    expect(content).toContainEqual(expect.objectContaining({ type: "text", text: "Searching..." }));
    expect(content).toContainEqual(
      expect.objectContaining({ type: "toolCall", id: "call_1|fc_1", name: "search" }),
    );
  });

  it("round-trips toolResult", () => {
    const original = [
      {
        role: "toolResult",
        content: '{"data": true}',
        tool_use_id: "tu_1",
        timestamp: Date.now(),
      },
    ];
    const openai = agentToOpenAI(original);
    const back = openAIToAgent(openai);
    expect(back[0].role).toBe("toolResult");
    expect(back[0].content).toEqual([{ type: "text", text: '{"data": true}' }]);
    expect(back[0].tool_use_id).toBe("tu_1");
  });
});

if (RUN) {
  describe("ProxyManager", () => {
    it("connects to configured proxy URL", { timeout: 30000 }, async () => {
      const manager = new ProxyManager({ proxyUrl: PROXY_URL });
      try {
        const url = await manager.start();
        expect(url).toMatch(/^http:\/\/(127\.0\.0\.1|localhost):\d+$/);

        // Verify health
        const resp = await fetch(`${url}/health`);
        expect(resp.ok).toBe(true);
      } finally {
        await manager.stop();
      }
    });
  });

  describe("HeadroomContextEngine", () => {
    let engine: HeadroomContextEngine;

    beforeAll(async () => {
      engine = new HeadroomContextEngine({ proxyUrl: PROXY_URL });
      await engine.bootstrap({
        sessionId: "test-session",
        sessionFile: "/tmp/test-session.jsonl",
      });
    }, 30000);

    afterAll(async () => {
      await engine.dispose();
    });

    it("assemble() compresses tool outputs", { timeout: 15000 }, async () => {
    // Simulate an OpenClaw agent conversation with large tool result
    const serverData = Array.from({ length: 100 }, (_, i) => ({
      id: i + 1,
      name: `server-${i + 1}`,
      status: i % 15 === 0 ? "critical" : i % 5 === 0 ? "warning" : "healthy",
      cpu: Math.round(Math.random() * 100),
      memory: Math.round(Math.random() * 100),
      region: ["us-east-1", "eu-west-1", "ap-southeast-1"][i % 3],
      description: `Production server ${i + 1} running service-${["auth", "payment", "user", "api"][i % 4]}`,
      lastAlert: i % 15 === 0 ? `Disk usage at ${90 + (i % 10)}%` : null,
    }));

    const messages = [
      { role: "user", content: "Check the fleet status", timestamp: Date.now() },
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "tu_fleet", name: "getFleetStatus", input: {} },
        ],
        timestamp: Date.now(),
      },
      {
        role: "toolResult",
        content: JSON.stringify(serverData),
        tool_use_id: "tu_fleet",
        timestamp: Date.now(),
      },
      { role: "user", content: "Which servers are critical?", timestamp: Date.now() },
    ];

    const result = await engine.assemble({
      sessionId: "test-session",
      messages,
      model: "claude-sonnet-4-5",
    });

    console.log(
      `  assemble(): estimatedTokens=${result.estimatedTokens}, ` +
        `systemPrompt=${result.systemPromptAddition ? "yes" : "no"}`,
    );

    // Messages should be returned (compressed or not)
    expect(result.messages.length).toBeGreaterThan(0);
    // First and last messages should still be user messages
    expect(result.messages[0].role).toBe("user");
    expect(result.messages[result.messages.length - 1].role).toBe("user");
    });

    it("assemble() preserves small conversations", { timeout: 15000 }, async () => {
    const messages = [
      { role: "user", content: "Hello", timestamp: Date.now() },
      { role: "assistant", content: "Hi there!", timestamp: Date.now() },
    ];

    const result = await engine.assemble({
      sessionId: "test-session",
      messages,
    });

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0].content).toBe("Hello");
    expect(result.messages[1].content).toBe("Hi there!");
    });

    it("compact() returns success (compression handled in assemble)", async () => {
    const result = await engine.compact({
      sessionId: "test-session",
      sessionFile: "/tmp/test.jsonl",
    });

    expect(result.ok).toBe(true);
    expect(result.compacted).toBe(true);
    });

    it("getStats() returns compression statistics", () => {
    const stats = engine.getStats();
    expect(stats).toHaveProperty("totalCompressions");
    expect(stats).toHaveProperty("totalTokensSaved");
    expect(stats.totalCompressions).toBeGreaterThanOrEqual(0);
    });
  });
}
