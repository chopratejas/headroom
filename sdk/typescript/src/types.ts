// --- Message types (OpenAI chat format) ---

export interface TextContentPart {
  type: "text";
  text: string;
}

export interface ImageContentPart {
  type: "image_url";
  image_url: { url: string; detail?: "auto" | "low" | "high" };
}

export type ContentPart = TextContentPart | ImageContentPart;

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface SystemMessage {
  role: "system";
  content: string;
}

export interface UserMessage {
  role: "user";
  content: string | ContentPart[];
}

export interface AssistantMessage {
  role: "assistant";
  content: string | null;
  tool_calls?: ToolCall[];
}

export interface ToolMessage {
  role: "tool";
  content: string;
  tool_call_id: string;
}

export type OpenAIMessage =
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | ToolMessage;

// --- Compress API ---

export interface CompressOptions {
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  fallback?: boolean;
  retries?: number;
  client?: HeadroomClientInterface;
}

export interface CompressResult {
  /** Compressed messages in the same format as input. */
  messages: any[];
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  compressionRatio: number;
  transformsApplied: string[];
  ccrHashes: string[];
  compressed: boolean;
}

// --- Client ---

export interface HeadroomClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  fallback?: boolean;
  retries?: number;
}

export interface HeadroomClientInterface {
  compress(
    messages: OpenAIMessage[],
    options?: { model?: string },
  ): Promise<CompressResult>;
}

// --- Errors ---

export class HeadroomError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HeadroomError";
  }
}

export class HeadroomConnectionError extends HeadroomError {
  constructor(message: string) {
    super(message);
    this.name = "HeadroomConnectionError";
  }
}

export class HeadroomAuthError extends HeadroomError {
  constructor(message: string) {
    super(message);
    this.name = "HeadroomAuthError";
  }
}

export class HeadroomCompressError extends HeadroomError {
  statusCode: number;
  errorType: string;

  constructor(statusCode: number, errorType: string, message: string) {
    super(message);
    this.name = "HeadroomCompressError";
    this.statusCode = statusCode;
    this.errorType = errorType;
  }
}

// --- Proxy response (internal) ---

export interface ProxyCompressResponse {
  messages: OpenAIMessage[];
  tokens_before: number;
  tokens_after: number;
  tokens_saved: number;
  compression_ratio: number;
  transforms_applied: string[];
  ccr_hashes: string[];
}

export interface ProxyErrorResponse {
  error: {
    type: string;
    message: string;
  };
}
