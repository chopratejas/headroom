/**
 * Headroom transport shim for OpenCode.
 *
 * Loaded via NODE_OPTIONS="--import ./shim.mjs" at wrap time.
 * Intercepts outbound HTTP/HTTPS/fetch requests and routes provider
 * traffic through the Headroom proxy.  Subagents and child Node
 * processes inherit the shim via patched child_process APIs.
 *
 * Environment variables:
 *   HEADROOM_PROXY_URL        - proxy base URL (default: http://127.0.0.1:8787)
 *   HEADROOM_PROVIDER_HOSTS   - comma-separated hostnames to route through proxy
 *   HEADROOM_SHIM_BYPASS_HOSTS - comma-separated hostnames to never route (avoid loops)
 */

const PROXY_URL = process.env.HEADROOM_PROXY_URL || "http://127.0.0.1:8787";
const PROXY_HOST = new URL(PROXY_URL).hostname;
const PROXY_PORT = parseInt(new URL(PROXY_URL).port, 10);

const PROVIDER_HOSTS = (process.env.HEADROOM_PROVIDER_HOSTS || "")
  .split(",")
  .map((h) => h.trim().toLowerCase())
  .filter(Boolean);

const BYPASS_HOSTS = new Set(
  [
    PROXY_HOST,
    "127.0.0.1",
    "localhost",
    "::1",
    ...(process.env.HEADROOM_SHIM_BYPASS_HOSTS || "")
      .split(",")
      .map((h) => h.trim().toLowerCase())
      .filter(Boolean),
  ].map((h) => h.toLowerCase())
);

function shouldRoute(hostname) {
  if (!hostname) return false;
  const lower = hostname.toLowerCase();
  if (BYPASS_HOSTS.has(lower)) return false;
  return PROVIDER_HOSTS.some((ph) => lower === ph || lower.endsWith("." + ph));
}

function patchHttp() {
  try {
    const http = require("http");
    const https = require("https");

    const origHttpRequest = http.request;
    const origHttpsRequest = https.request;
    const origHttpGet = http.get;
    const origHttpsGet = https.get;

    function rewriteOptions(options) {
      if (typeof options === "string" || options instanceof URL) {
        const url = new URL(options.toString());
        if (shouldRoute(url.hostname)) {
          const proxyUrl = new URL(url.pathname + url.search, PROXY_URL);
          return {
            hostname: PROXY_HOST,
            port: PROXY_PORT,
            path: url.pathname + url.search,
            headers: {
              ...(options.headers || {}),
              "x-headroom-base-url": url.origin,
            },
          };
        }
        return options;
      }

      if (options && typeof options === "object") {
        const hostname = options.hostname || options.host || "";
        if (shouldRoute(hostname)) {
          const origOrigin = `https://${hostname}:${options.port || 443}`;
          const newOpts = { ...options };
          newOpts.hostname = PROXY_HOST;
          newOpts.port = PROXY_PORT;
          newOpts.headers = { ...(newOpts.headers || {}) };
          newOpts.headers["x-headroom-base-url"] = origOrigin;
          return newOpts;
        }
      }

      return options;
    }

    http.request = function (url, options, callback) {
      if (typeof options === "function") {
        callback = options;
        options = {};
      }
      const rewritten = rewriteOptions(url);
      return origHttpRequest.call(this, rewritten, options, callback);
    };

    https.request = function (url, options, callback) {
      if (typeof options === "function") {
        callback = options;
        options = {};
      }
      const rewritten = rewriteOptions(url);
      return origHttpsRequest.call(this, rewritten, options, callback);
    };

    http.get = function (url, options, callback) {
      if (typeof options === "function") {
        callback = options;
        options = {};
      }
      const rewritten = rewriteOptions(url);
      return origHttpGet.call(this, rewritten, options, callback);
    };

    https.get = function (url, options, callback) {
      if (typeof options === "function") {
        callback = options;
        options = {};
      }
      const rewritten = rewriteOptions(url);
      return origHttpsGet.call(this, rewritten, options, callback);
    };
  } catch {
    // http/https not available — skip
  }
}

function patchFetch() {
  if (typeof globalThis.fetch !== "function") return;

  const origFetch = globalThis.fetch;

  globalThis.fetch = function (input, init) {
    let url;
    if (typeof input === "string") {
      url = new URL(input);
    } else if (input instanceof URL) {
      url = input;
    } else if (input && typeof input === "object" && input.url) {
      url = new URL(input.url);
    } else {
      return origFetch.call(this, input, init);
    }

    if (shouldRoute(url.hostname)) {
      const proxyUrl = new URL(url.pathname + url.search, PROXY_URL);
      const newInit = { ...(init || {}) };
      newInit.headers = { ...(newInit.headers || {}) };
      newInit.headers["x-headroom-base-url"] = url.origin;
      return origFetch.call(this, proxyUrl.toString(), newInit);
    }

    return origFetch.call(this, input, init);
  };
}

function patchChildProcess() {
  try {
    const cp = require("child_process");
    const origSpawn = cp.spawn;
    const origFork = cp.fork;

    const shimPath = __filename;

    function ensureNodeOptions(env) {
      const newEnv = { ...env };
      const existing = newEnv.NODE_OPTIONS || "";
      if (!existing.includes(shimPath)) {
        newEnv.NODE_OPTIONS = `${existing} --import ${shimPath}`.trim();
      }
      return newEnv;
    }

    cp.spawn = function (command, args, options) {
      if (options && typeof options === "object" && options.env) {
        options.env = ensureNodeOptions(options.env);
      }
      return origSpawn.call(this, command, args, options);
    };

    cp.fork = function (modulePath, args, options) {
      if (options && typeof options === "object" && options.env) {
        options.env = ensureNodeOptions(options.env);
      } else if (options && typeof options === "object") {
        options.env = ensureNodeOptions(process.env);
      }
      return origFork.call(this, modulePath, args, options);
    };
  } catch {
    // child_process not available — skip
  }
}

patchHttp();
patchFetch();
patchChildProcess();
