import { describe, expect, it } from "vitest";
import {
  applyGatewayProviderBaseUrls,
  applyGatewayProviderBaseUrlsInPlace,
  resolveGatewayProviderIds,
} from "../src/gateway-config.js";

describe("resolveGatewayProviderIds", () => {
  it("routes openai-codex by default", () => {
    expect(resolveGatewayProviderIds(undefined)).toEqual(["openai-codex"]);
  });

  it("allows routing to be disabled", () => {
    expect(resolveGatewayProviderIds({ routeCodexViaProxy: false })).toEqual([]);
  });
});

describe("applyGatewayProviderBaseUrls", () => {
  it("creates an openai-codex provider config when missing", () => {
    const result = applyGatewayProviderBaseUrls({}, "http://127.0.0.1:8787", ["openai-codex"]);

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["openai-codex"]).toEqual({
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });
  });

  it("preserves existing provider config fields", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "openai-codex": {
              api: "openai-codex-responses",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["openai-codex"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["openai-codex"]).toEqual({
      api: "openai-codex-responses",
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });
  });

  it("is a no-op when the provider already points at headroom", () => {
    const cfg = {
      models: {
        providers: {
          "openai-codex": {
            baseUrl: "http://127.0.0.1:8787",
            models: [],
          },
        },
      },
    };

    const result = applyGatewayProviderBaseUrls(cfg, "http://127.0.0.1:8787", ["openai-codex"]);

    expect(result.changed).toBe(false);
    expect(result.config).toEqual(cfg);
  });
});

describe("applyGatewayProviderBaseUrlsInPlace", () => {
  it("updates the live config object in place", () => {
    const cfg: any = { models: { providers: {} } };

    const changed = applyGatewayProviderBaseUrlsInPlace(
      cfg,
      "http://127.0.0.1:8787",
      ["openai-codex"],
    );

    expect(changed).toBe(true);
    expect(cfg.models.providers["openai-codex"]).toEqual({
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });
  });

  it("does not clobber existing provider logic when changing only the base URL", () => {
    const cfg: any = {
      models: {
        providers: {
          "openai-codex": {
            api: "openai-codex-responses",
            envKey: "OPENAI_API_KEY",
            models: ["gpt-5.3-codex"],
          },
        },
      },
    };

    const changed = applyGatewayProviderBaseUrlsInPlace(
      cfg,
      "http://127.0.0.1:8787",
      ["openai-codex"],
    );

    expect(changed).toBe(true);
    expect(cfg.models.providers["openai-codex"]).toEqual({
      api: "openai-codex-responses",
      envKey: "OPENAI_API_KEY",
      baseUrl: "http://127.0.0.1:8787",
      models: ["gpt-5.3-codex"],
    });
  });
});
