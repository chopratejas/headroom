import type {
  OpenAIMessage,
  CompressResult,
  HeadroomClientOptions,
  HeadroomClientInterface,
  ProxyCompressResponse,
  ProxyErrorResponse,
} from "./types.js";
import {
  HeadroomConnectionError,
  HeadroomAuthError,
  HeadroomCompressError,
} from "./types.js";

const DEFAULT_BASE_URL = "http://localhost:8787";
const DEFAULT_TIMEOUT = 30_000;
const DEFAULT_RETRIES = 1;

function getEnv(key: string): string | undefined {
  if (typeof process !== "undefined" && process.env) {
    return process.env[key];
  }
  return undefined;
}

function makeFallbackResult(messages: OpenAIMessage[]): CompressResult {
  return {
    messages,
    tokensBefore: 0,
    tokensAfter: 0,
    tokensSaved: 0,
    compressionRatio: 1.0,
    transformsApplied: [],
    ccrHashes: [],
    compressed: false,
  };
}

export class HeadroomClient implements HeadroomClientInterface {
  private baseUrl: string;
  private apiKey: string | undefined;
  private timeout: number;
  private fallback: boolean;
  private retries: number;

  constructor(options: HeadroomClientOptions = {}) {
    this.baseUrl = (
      options.baseUrl ??
      getEnv("HEADROOM_BASE_URL") ??
      DEFAULT_BASE_URL
    ).replace(/\/+$/, "");
    this.apiKey = options.apiKey ?? getEnv("HEADROOM_API_KEY");
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.fallback = options.fallback ?? true;
    this.retries = options.retries ?? DEFAULT_RETRIES;
  }

  async compress(
    messages: OpenAIMessage[],
    options: { model?: string; tokenBudget?: number } = {},
  ): Promise<CompressResult> {
    const model = options.model ?? "gpt-4o";

    let lastError: unknown;
    const maxAttempts = 1 + this.retries;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        return await this._doCompress(messages, model, options.tokenBudget);
      } catch (error) {
        lastError = error;
        // Don't retry auth errors or client errors
        if (error instanceof HeadroomAuthError) throw error;
        if (
          error instanceof HeadroomCompressError &&
          error.statusCode < 500
        ) {
          throw error;
        }
        // Retry on transient errors (network, 5xx)
      }
    }

    // All attempts exhausted
    if (this.fallback) {
      return makeFallbackResult(messages);
    }
    if (lastError instanceof HeadroomConnectionError) throw lastError;
    if (lastError instanceof HeadroomCompressError) throw lastError;
    throw new HeadroomConnectionError(
      `Failed after ${maxAttempts} attempts: ${lastError}`,
    );
  }

  private async _doCompress(
    messages: OpenAIMessage[],
    model: string,
    tokenBudget?: number,
  ): Promise<CompressResult> {
    const url = `${this.baseUrl}/v1/compress`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const body: Record<string, unknown> = { messages, model };
    if (tokenBudget) {
      body.token_budget = tokenBudget;
    }

    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeout),
      });
    } catch (error) {
      throw new HeadroomConnectionError(
        `Failed to connect to Headroom at ${this.baseUrl}: ${error}`,
      );
    }

    if (!response.ok) {
      let errorBody: ProxyErrorResponse | undefined;
      try {
        errorBody = (await response.json()) as ProxyErrorResponse;
      } catch {
        // ignore parse errors on error responses
      }
      const errorType = errorBody?.error?.type ?? "unknown";
      const errorMessage =
        errorBody?.error?.message ?? `HTTP ${response.status}`;

      if (response.status === 401) {
        throw new HeadroomAuthError(errorMessage);
      }
      throw new HeadroomCompressError(response.status, errorType, errorMessage);
    }

    const data = (await response.json()) as ProxyCompressResponse;

    return {
      messages: data.messages,
      tokensBefore: data.tokens_before,
      tokensAfter: data.tokens_after,
      tokensSaved: data.tokens_saved,
      compressionRatio: data.compression_ratio,
      transformsApplied: data.transforms_applied,
      ccrHashes: data.ccr_hashes,
      compressed: true,
    };
  }
}
