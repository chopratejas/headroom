/**
 * Headroom OpenClaw Plugin — register ContextEngine + CCR retrieval tool.
 *
 * Usage:
 *   openclaw plugins install @headroom-ai/openclaw
 *
 * Configuration (in ~/.openclaw/config.json or ~/.clawdbot/clawdbot.json):
 *   {
 *     "plugins": {
 *       "slots": { "contextEngine": "headroom" },
 *       "entries": { "headroom": { "enabled": true } }
 *     }
 *   }
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { HeadroomContextEngine } from "../engine.js";
import { createHeadroomRetrieveTool } from "../tools/headroom-retrieve.js";

export default function headroomPlugin(api: any) {
  const config = api.config?.plugins?.entries?.headroom?.config ?? {};
  const logger = api.logger ?? console;

  const engine = new HeadroomContextEngine(config, {
    info: (m: string) => logger.info(m),
    warn: (m: string) => logger.warn(m),
    error: (m: string) => logger.error(m),
    debug: (m: string) => logger.debug?.(m),
  });

  // Register as context engine
  api.registerContextEngine("headroom", () => engine);

  // Register CCR retrieval tool (active once proxy is running)
  api.registerTool((ctx: any) => {
    const proxyUrl = engine.getProxyUrl();
    if (!proxyUrl) return null;
    return createHeadroomRetrieveTool({ proxyUrl });
  });

  logger.info("[headroom] Plugin registered");
}
