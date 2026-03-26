import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  headroomMiddleware,
  compressVercelMessages,
} from "../../src/adapters/vercel-ai.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockCompressResponse(
  messages: any[],
  tokensBefore = 100,
  tokensAfter = 30,
) {
  return new Response(
    JSON.stringify({
      messages,
      tokens_before: tokensBefore,
      tokens_after: tokensAfter,
      tokens_saved: tokensBefore - tokensAfter,
      compression_ratio: tokensAfter / tokensBefore,
      transforms_applied: ["smart_crusher"],
      ccr_hashes: [],
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

describe("headroomMiddleware", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("returns object with transformParams function", () => {
    const mw = headroomMiddleware({ baseUrl: "http://localhost:8787" });
    expect(mw.transformParams).toBeDefined();
    expect(typeof mw.transformParams).toBe("function");
  });

  it("compresses prompt via transformParams", async () => {
    // Proxy returns compressed OpenAI messages
    mockFetch.mockResolvedValueOnce(
      mockCompressResponse([{ role: "user", content: "compressed" }]),
    );

    const mw = headroomMiddleware({ baseUrl: "http://localhost:8787" });

    const params = {
      prompt: [
        { role: "system", content: "Be helpful" },
        {
          role: "user",
          content: [{ type: "text", text: "long message here" }],
        },
      ],
      modelId: "gpt-4o",
    };

    const result = await mw.transformParams!({
      params: params as any,
      model: {} as any,
      type: "generate",
    });

    expect(mockFetch).toHaveBeenCalledOnce();
    expect(result.prompt).toBeDefined();
    expect(result.prompt).not.toEqual(params.prompt); // should be modified
  });

  it("passes through empty prompt unchanged", async () => {
    const mw = headroomMiddleware({ baseUrl: "http://localhost:8787" });

    const params = { prompt: [], modelId: "gpt-4o" };
    const result = await mw.transformParams!({
      params: params as any,
      model: {} as any,
      type: "generate",
    });

    expect(mockFetch).not.toHaveBeenCalled();
    expect(result).toEqual(params);
  });

  it("passes through when compression returns compressed=false", async () => {
    // Simulate fallback (proxy unreachable)
    mockFetch.mockRejectedValueOnce(new TypeError("fetch failed"));

    const mw = headroomMiddleware({
      baseUrl: "http://localhost:8787",
      fallback: true,
    });

    const originalPrompt = [
      {
        role: "user",
        content: [{ type: "text", text: "hello" }],
      },
    ];

    const params = { prompt: originalPrompt, modelId: "gpt-4o" };
    const result = await mw.transformParams!({
      params: params as any,
      model: {} as any,
      type: "generate",
    });

    // Should return original params since compression failed
    expect(result).toEqual(params);
  });

  it("uses model from options over params.modelId", async () => {
    mockFetch.mockResolvedValueOnce(
      mockCompressResponse([{ role: "user", content: "x" }]),
    );

    const mw = headroomMiddleware({
      baseUrl: "http://localhost:8787",
      model: "claude-sonnet-4-5-20250929",
    });

    await mw.transformParams!({
      params: {
        prompt: [
          { role: "user", content: [{ type: "text", text: "hi" }] },
        ],
        modelId: "gpt-4o",
      } as any,
      model: {} as any,
      type: "generate",
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.model).toBe("claude-sonnet-4-5-20250929");
  });
});

describe("compressVercelMessages", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("converts, compresses, and converts back to Vercel format", async () => {
    mockFetch.mockResolvedValueOnce(
      mockCompressResponse([
        { role: "system", content: "Be helpful" },
        { role: "user", content: "hello" },
      ]),
    );

    const vercelMessages = [
      { role: "system", content: "Be helpful" },
      {
        role: "user",
        content: [{ type: "text", text: "hello" }],
      },
    ];

    const result = await compressVercelMessages(vercelMessages, {
      model: "gpt-4o",
      baseUrl: "http://localhost:8787",
    });

    expect(result.compressed).toBe(true);
    expect(result.tokensBefore).toBe(100);
    // Messages should be in Vercel format (user has content parts array)
    expect(result.messages[0]).toEqual({
      role: "system",
      content: "Be helpful",
    });
    expect(result.messages[1].role).toBe("user");
    expect(result.messages[1].content[0].type).toBe("text");
  });
});
