export { compress } from "./compress.js";
export { HeadroomClient } from "./client.js";
export { detectFormat, toOpenAI, fromOpenAI } from "./utils/format.js";
export type { MessageFormat } from "./utils/format.js";

export type {
  TextContentPart,
  ImageContentPart,
  ContentPart,
  ToolCall,
  SystemMessage,
  UserMessage,
  AssistantMessage,
  ToolMessage,
  OpenAIMessage,
  CompressOptions,
  CompressResult,
  HeadroomClientOptions,
  HeadroomClientInterface,
} from "./types.js";

export {
  HeadroomError,
  HeadroomConnectionError,
  HeadroomAuthError,
  HeadroomCompressError,
} from "./types.js";
