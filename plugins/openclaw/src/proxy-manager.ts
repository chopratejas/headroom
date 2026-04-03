/**
 * Manages connectivity to an externally managed Headroom proxy.
 *
 * Security model:
 * - Optional local process execution to auto-start Headroom proxy
 * - No environment variable access
 * - Localhost-only network access (127.0.0.1 / localhost)
 */
import { spawn } from "node:child_process";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export interface ProxyManagerConfig {
  proxyUrl?: string;
  proxyPort?: number;
  pythonPath?: string;
  autoStart?: boolean;
  startupTimeoutMs?: number;
}

export interface ProxyManagerLogger {
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
  debug(message: string): void;
}

export interface ProxyProbeResult {
  reachable: boolean;
  isHeadroom: boolean;
  reason?: string;
}

interface LaunchSpec {
  label: string;
  command: string;
  args: string[];
  checkCommand: string;
  checkArgs: string[];
}

const defaultLogger: ProxyManagerLogger = {
  info: (m) => console.log(`[headroom] ${m}`),
  warn: (m) => console.warn(`[headroom] ${m}`),
  error: (m) => console.error(`[headroom] ${m}`),
  debug: () => {},
};

export class ProxyManager {
  private config: ProxyManagerConfig;
  private logger: ProxyManagerLogger;
  private proxyUrl: string | null = null;

  constructor(config: ProxyManagerConfig = {}, logger?: ProxyManagerLogger) {
    this.config = config;
    this.logger = logger ?? defaultLogger;
  }

  /**
   * Ensure a proxy is available. Returns the normalized URL origin.
   */
  async start(): Promise<string> {
    const port = this.getProxyPort();
    const explicitUrl =
      typeof this.config.proxyUrl === "string" && this.config.proxyUrl.trim().length > 0
        ? normalizeAndValidateProxyUrl(this.config.proxyUrl)
        : null;
    const defaultCandidates = this.getDefaultProxyCandidates(port);
    const candidateUrls = explicitUrl ? [explicitUrl] : [...defaultCandidates];
    const probeByUrl = new Map<string, ProxyProbeResult>();

    for (const url of candidateUrls) {
      const probe = await probeHeadroomProxy(url);
      probeByUrl.set(url, probe);
      if (probe.reachable && probe.isHeadroom) {
        this.proxyUrl = url;
        this.logger.info(`Headroom proxy already running at ${url}`);
        return url;
      }
    }

    if (explicitUrl) {
      const explicitProbe = probeByUrl.get(explicitUrl);
      if (explicitProbe?.reachable && !explicitProbe.isHeadroom) {
        throw new Error(
          `Service reachable at ${explicitUrl}, but it does not appear to be a Headroom proxy (${explicitProbe.reason ?? "unknown service"}).`,
        );
      }
    }

    if (this.config.autoStart !== false) {
      const startupUrl = explicitUrl ?? defaultCandidates[0];
      const startupProbe = probeByUrl.get(startupUrl);
      if (startupProbe?.reachable && !startupProbe.isHeadroom) {
        throw new Error(
          `Cannot auto-start Headroom at ${startupUrl}: port is in use by a non-Headroom service (${startupProbe.reason ?? "unknown service"}).`,
        );
      }

      this.logger.info(
        `No Headroom proxy detected${explicitUrl ? ` at ${startupUrl}` : " on default local endpoints"}; attempting to auto-start...`,
      );
      await this.startHeadroomProxy(startupUrl);

      const startedProbe = await waitForHeadroomProxy(
        startupUrl,
        this.config.startupTimeoutMs ?? 20_000,
      );
      if (startedProbe.reachable && startedProbe.isHeadroom) {
        this.proxyUrl = startupUrl;
        this.logger.info(`Headroom proxy started and reachable at ${startupUrl}`);
        return startupUrl;
      }
      throw new Error(
        `Attempted to start Headroom proxy, but it was not reachable at ${startupUrl} (${startedProbe.reason ?? "unknown"}).`,
      );
    }

    if (explicitUrl) {
      throw new Error(
        `Headroom proxy not reachable at ${explicitUrl}. Ensure the proxy is running first.`,
      );
    }

    throw new Error(
      `Headroom proxy not detected on default endpoints (${defaultCandidates.join(", ")}). ` +
        "Set proxyUrl explicitly or enable autoStart.",
    );
  }

  private getProxyPort(): number {
    const rawPort = this.config.proxyPort;
    if (!Number.isInteger(rawPort) || rawPort === undefined) return 8787;
    if (rawPort < 1 || rawPort > 65535) {
      throw new Error("proxyPort must be an integer between 1 and 65535");
    }
    return rawPort;
  }

  private getDefaultProxyCandidates(port: number): string[] {
    return [`http://127.0.0.1:${port}`, `http://localhost:${port}`];
  }

  /**
   * No-op: plugin never starts or manages external processes.
   */
  async stop(): Promise<void> {
    this.proxyUrl = null;
  }

  getUrl(): string | null {
    return this.proxyUrl;
  }

  // --- Internal ---

  private async startHeadroomProxy(proxyUrl: string): Promise<void> {
    const parsed = new URL(proxyUrl);
    const host = parsed.hostname;
    const port = parsed.port || "80";
    const specs = this.buildLaunchSpecs(host, port);
    const errors: string[] = [];

    for (const spec of specs) {
      if (!this.canExecute(spec.checkCommand, spec.checkArgs)) {
        this.logger.debug(`Launcher unavailable: ${spec.label}`);
        continue;
      }

      try {
        const child = spawn(spec.command, spec.args, {
          detached: true,
          stdio: "ignore",
        });
        child.unref();
        this.logger.info(`Auto-start launcher selected: ${spec.label}`);
        return;
      } catch (error) {
        errors.push(`${spec.label}: ${String(error)}`);
      }
    }

    throw new Error(
      "No usable Headroom launcher found. Tried PATH, local npm, global npm, and Python. " +
        "Install headroom-ai (npm or pip) and ensure one launcher is available.\n" +
        (errors.length > 0 ? `Launch errors: ${errors.join("; ")}` : ""),
    );
  }

  private buildLaunchSpecs(host: string, port: string): LaunchSpec[] {
    const commonArgs = ["proxy", "--host", host, "--port", port];
    const specs: LaunchSpec[] = [];

    // 1) PATH
    specs.push({
      label: "PATH: headroom",
      command: "headroom",
      args: commonArgs,
      checkCommand: "headroom",
      checkArgs: ["--version"],
    });

    // 2) Local npm install (inside plugin install path)
    const moduleDir = dirname(fileURLToPath(import.meta.url)); // .../dist
    const packageRoot = dirname(moduleDir);
    const localBinDir = join(packageRoot, "node_modules", ".bin");
    const localBins = process.platform === "win32"
      ? [join(localBinDir, "headroom.cmd"), join(localBinDir, "headroom")]
      : [join(localBinDir, "headroom")];
    for (const localBin of localBins) {
      if (!existsSync(localBin)) continue;
      specs.push({
        label: `Local npm: ${localBin}`,
        command: localBin,
        args: commonArgs,
        checkCommand: localBin,
        checkArgs: ["--version"],
      });
    }

    // 3) Global npm install
    const npmPrefix = this.getNpmGlobalPrefix();
    if (npmPrefix) {
      const globalBins = process.platform === "win32"
        ? [join(npmPrefix, "headroom.cmd"), join(npmPrefix, "headroom")]
        : [join(npmPrefix, "bin", "headroom"), join(npmPrefix, "headroom")];

      for (const globalBin of globalBins) {
        if (!existsSync(globalBin)) continue;
        specs.push({
          label: `Global npm: ${globalBin}`,
          command: globalBin,
          args: commonArgs,
          checkCommand: globalBin,
          checkArgs: ["--version"],
        });
      }
    }

    // 4) Python module fallback
    const pythonCommands = this.getPythonCommands();
    for (const pyCmd of pythonCommands) {
      specs.push({
        label: `Python: ${pyCmd} -m headroom.cli`,
        command: pyCmd,
        args: ["-m", "headroom.cli", ...commonArgs],
        checkCommand: pyCmd,
        checkArgs: ["-c", "import headroom"],
      });
    }

    return specs;
  }

  private getPythonCommands(): string[] {
    const commands: string[] = [];
    const configured = typeof this.config.pythonPath === "string"
      ? this.config.pythonPath.trim()
      : "";
    if (configured.length > 0) {
      commands.push(configured);
    }
    for (const fallback of ["python", "python3", "py"]) {
      if (!commands.includes(fallback)) commands.push(fallback);
    }
    return commands;
  }

  private canExecute(command: string, args: string[]): boolean {
    try {
      const result = spawnSync(command, args, {
        stdio: "ignore",
        timeout: 5000,
      });
      if (result.error) return false;
      return result.status === 0;
    } catch {
      return false;
    }
  }

  private getNpmGlobalPrefix(): string | null {
    try {
      const result = spawnSync("npm", ["prefix", "-g"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 5000,
      });
      if (result.error || result.status !== 0) return null;
      const prefix = (result.stdout ?? "").trim();
      return prefix.length > 0 ? prefix : null;
    } catch {
      return null;
    }
  }
}

export function normalizeAndValidateProxyUrl(proxyUrl: string): string {
  let parsed: URL;
  try {
    parsed = new URL(proxyUrl);
  } catch {
    throw new Error(`Invalid proxyUrl: "${proxyUrl}"`);
  }

  if (parsed.protocol !== "http:") {
    throw new Error("proxyUrl must use http://");
  }
  if (parsed.hostname !== "127.0.0.1" && parsed.hostname !== "localhost") {
    throw new Error("proxyUrl host must be localhost or 127.0.0.1");
  }

  if (parsed.pathname !== "/" || parsed.search || parsed.hash) {
    throw new Error("proxyUrl must not include a path, query, or hash");
  }

  return parsed.origin;
}

/**
 * Probe a configured URL and verify whether it is a running Headroom proxy.
 */
export async function probeHeadroomProxy(proxyUrl: string): Promise<ProxyProbeResult> {
  const origin = normalizeAndValidateProxyUrl(proxyUrl);

  try {
    const health = await fetch(`${origin}/health`, {
      signal: AbortSignal.timeout(3_000),
    });
    if (!health.ok) {
      return { reachable: false, isHeadroom: false, reason: `health HTTP ${health.status}` };
    }
  } catch {
    return { reachable: false, isHeadroom: false, reason: "health check failed" };
  }

  try {
    const retrieveStats = await fetch(`${origin}/v1/retrieve/stats`, {
      signal: AbortSignal.timeout(3_000),
    });
    if (retrieveStats.ok) {
      return { reachable: true, isHeadroom: true };
    }
    return {
      reachable: true,
      isHeadroom: false,
      reason: `retrieve stats HTTP ${retrieveStats.status}`,
    };
  } catch {
    return {
      reachable: true,
      isHeadroom: false,
      reason: "retrieve stats endpoint unavailable",
    };
  }
}

async function waitForHeadroomProxy(proxyUrl: string, timeoutMs: number): Promise<ProxyProbeResult> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const result = await probeHeadroomProxy(proxyUrl);
    if (result.reachable && result.isHeadroom) {
      return result;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return probeHeadroomProxy(proxyUrl);
}
