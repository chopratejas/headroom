import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  ensureProxyUrl: vi.fn(async () => "http://127.0.0.1:8787"),
  getProxyUrl: vi.fn(() => "http://127.0.0.1:8787"),
  createHeadroomRetrieveTool: vi.fn(({ proxyUrl }: { proxyUrl: string }) => ({ proxyUrl })),
}));

vi.mock("../src/engine.js", () => ({
  HeadroomContextEngine: class {
    ensureProxyUrl = mocked.ensureProxyUrl;
    getProxyUrl = mocked.getProxyUrl;
  },
}));

vi.mock("../src/tools/headroom-retrieve.js", () => ({
  createHeadroomRetrieveTool: mocked.createHeadroomRetrieveTool,
}));

import headroomPlugin from "../src/plugin/index.js";

afterEach(() => {
  mocked.ensureProxyUrl.mockClear();
  mocked.getProxyUrl.mockClear();
  mocked.createHeadroomRetrieveTool.mockClear();
});

describe("headroomPlugin runtime routing", () => {
  it("routes configured providers in memory without writing config files", async () => {
    const gatewayHandlers = new Map<string, () => Promise<void>>();
    const writeConfigFile = vi.fn();
    const loadConfig = vi.fn(() => ({
      models: {
        providers: {
          anthropic: {
            api: "anthropic-messages",
          },
        },
      },
    }));

    const api: any = {
      config: {
        plugins: {
          entries: {
            headroom: {
              config: {
                gatewayProviderIds: ["openai-codex", "anthropic", "github-copilot"],
              },
            },
          },
        },
        models: {
          providers: {
            anthropic: {
              api: "anthropic-messages",
            },
          },
        },
      },
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
      registerContextEngine: vi.fn(),
      registerTool: vi.fn(),
      on: vi.fn((event: string, handler: () => Promise<void>) => {
        gatewayHandlers.set(event, handler);
      }),
      runtime: {
        config: {
          loadConfig,
          writeConfigFile,
        },
      },
    };

    headroomPlugin(api);
    await Promise.resolve();

    expect(mocked.ensureProxyUrl).toHaveBeenCalledTimes(1);
    expect(writeConfigFile).not.toHaveBeenCalled();
    expect(loadConfig).not.toHaveBeenCalled();
    expect(api.config.models.providers["openai-codex"]).toEqual({
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });
    expect(api.config.models.providers.anthropic).toEqual({
      api: "anthropic-messages",
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });
    expect(api.config.models.providers["github-copilot"]).toEqual({
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });

    const gatewayStart = gatewayHandlers.get("gateway_start");
    expect(gatewayStart).toBeTypeOf("function");
    await gatewayStart?.();
    expect(writeConfigFile).not.toHaveBeenCalled();
    expect(loadConfig).not.toHaveBeenCalled();
  });
});
