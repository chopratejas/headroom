/**
 * Manages the Headroom proxy process lifecycle.
 *
 * - Detects if a proxy is already running (e.g., user has `headroom proxy` for Claude Code)
 * - If not, spawns one as a child process with auto-assigned port
 * - Health checks, restart on crash, graceful shutdown
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const DEFAULT_PORT = 8787;
const HEALTH_CHECK_INTERVAL_MS = 30_000;
const STARTUP_TIMEOUT_MS = 15_000;
const RESTART_DELAY_MS = 2_000;
const MAX_RESTART_ATTEMPTS = 3;

export interface ProxyManagerConfig {
  proxyUrl?: string;
  pythonPath?: string;
  autoStart?: boolean;
  proxyPort?: number;
}

export interface ProxyManagerLogger {
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
  debug(message: string): void;
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
  private process: ChildProcess | null = null;
  private proxyUrl: string | null = null;
  private weStartedIt = false;
  private restartCount = 0;
  private healthInterval: ReturnType<typeof setInterval> | null = null;
  private disposed = false;

  constructor(config: ProxyManagerConfig = {}, logger?: ProxyManagerLogger) {
    this.config = config;
    this.logger = logger ?? defaultLogger;
  }

  /**
   * Ensure a proxy is available. Returns the URL.
   *
   * 1. If proxyUrl is configured, use it
   * 2. Check if proxy is already running on default port
   * 3. If autoStart, spawn one
   */
  async start(): Promise<string> {
    // Option 1: Explicit URL configured
    if (this.config.proxyUrl) {
      const url = this.config.proxyUrl.replace(/\/+$/, "");
      if (await this.healthCheck(url)) {
        this.proxyUrl = url;
        this.logger.info(`Connected to proxy at ${url}`);
        return url;
      }
      throw new Error(`Headroom proxy not reachable at ${url}`);
    }

    // Option 2: Check default port
    const defaultUrl = `http://127.0.0.1:${DEFAULT_PORT}`;
    if (await this.healthCheck(defaultUrl)) {
      this.proxyUrl = defaultUrl;
      this.logger.info(`Found running proxy at ${defaultUrl}`);
      this.startHealthMonitor();
      return defaultUrl;
    }

    // Option 3: Auto-start
    if (this.config.autoStart !== false) {
      return this.spawnProxy();
    }

    throw new Error(
      "Headroom proxy not running. Start with: headroom proxy --port 8787\n" +
        "Or install: pip install 'headroom-ai[proxy]'",
    );
  }

  /**
   * Spawn the headroom proxy as a child process.
   */
  private async spawnProxy(): Promise<string> {
    const pythonPath = await this.findPython();
    if (!pythonPath) {
      throw new Error(
        "Python not found. Install Python 3.10+ and run: pip install 'headroom-ai[proxy]'",
      );
    }

    // Check if headroom-ai is installed
    const installed = await this.checkHeadroomInstalled(pythonPath);
    if (!installed) {
      throw new Error(
        "headroom-ai Python package not found.\n" +
          "Install with: pip install 'headroom-ai[proxy]'",
      );
    }

    const port = this.config.proxyPort ?? 0; // 0 = OS picks a free port
    const actualPort = port === 0 ? await this.findFreePort() : port;
    const url = `http://127.0.0.1:${actualPort}`;

    this.logger.info(`Starting proxy on port ${actualPort}...`);

    // Log file
    const logDir = join(homedir(), ".headroom", "logs");
    const logPath = join(logDir, "openclaw-proxy.log");

    let logStream: ReturnType<typeof createWriteStream> | null = null;
    try {
      const { mkdirSync } = await import("node:fs");
      mkdirSync(logDir, { recursive: true });
      logStream = createWriteStream(logPath, { flags: "a" });
    } catch {
      // Can't create log file — use /dev/null
    }

    const proc = spawn(
      pythonPath,
      ["-m", "headroom.cli", "proxy", "--port", String(actualPort)],
      {
        env: { ...process.env, PYTHONIOENCODING: "utf-8" },
        stdio: ["ignore", logStream ? "pipe" : "ignore", logStream ? "pipe" : "ignore"],
        detached: false,
      },
    );

    if (logStream) {
      proc.stdout?.pipe(logStream);
      proc.stderr?.pipe(logStream);
    }

    proc.on("exit", (code) => {
      if (!this.disposed && this.weStartedIt) {
        this.logger.warn(`Proxy exited with code ${code}`);
        this.handleCrash();
      }
    });

    this.process = proc;
    this.weStartedIt = true;

    // Wait for healthy
    const healthy = await this.waitForHealthy(url, STARTUP_TIMEOUT_MS);
    if (!healthy) {
      proc.kill();
      this.process = null;
      throw new Error(
        `Proxy failed to start within ${STARTUP_TIMEOUT_MS / 1000}s. Check ${logPath}`,
      );
    }

    this.proxyUrl = url;
    this.logger.info(`Proxy started on port ${actualPort} (PID: ${proc.pid})`);
    this.startHealthMonitor();
    return url;
  }

  /**
   * Stop the proxy if we started it.
   */
  async stop(): Promise<void> {
    this.disposed = true;
    if (this.healthInterval) {
      clearInterval(this.healthInterval);
      this.healthInterval = null;
    }
    if (this.process && this.weStartedIt) {
      this.logger.info("Stopping proxy...");
      this.process.kill("SIGTERM");
      // Give it 3s to shutdown gracefully
      await new Promise<void>((resolve) => {
        const timeout = setTimeout(() => {
          this.process?.kill("SIGKILL");
          resolve();
        }, 3000);
        this.process?.on("exit", () => {
          clearTimeout(timeout);
          resolve();
        });
      });
      this.process = null;
    }
  }

  getUrl(): string | null {
    return this.proxyUrl;
  }

  // --- Internal ---

  private async healthCheck(url: string): Promise<boolean> {
    try {
      const resp = await fetch(`${url}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }

  private async waitForHealthy(url: string, timeoutMs: number): Promise<boolean> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      if (await this.healthCheck(url)) return true;
      await new Promise((r) => setTimeout(r, 500));
    }
    return false;
  }

  private startHealthMonitor(): void {
    if (this.healthInterval) return;
    this.healthInterval = setInterval(async () => {
      if (this.proxyUrl && !(await this.healthCheck(this.proxyUrl))) {
        this.logger.warn("Proxy health check failed");
        if (this.weStartedIt) this.handleCrash();
      }
    }, HEALTH_CHECK_INTERVAL_MS);
  }

  private async handleCrash(): Promise<void> {
    if (this.disposed) return;
    if (this.restartCount >= MAX_RESTART_ATTEMPTS) {
      this.logger.error(`Proxy crashed ${MAX_RESTART_ATTEMPTS} times. Giving up.`);
      return;
    }
    this.restartCount++;
    this.logger.info(`Restarting proxy (attempt ${this.restartCount}/${MAX_RESTART_ATTEMPTS})...`);
    await new Promise((r) => setTimeout(r, RESTART_DELAY_MS));
    try {
      await this.spawnProxy();
    } catch (e) {
      this.logger.error(`Restart failed: ${e}`);
    }
  }

  private async findPython(): Promise<string | null> {
    if (this.config.pythonPath) return this.config.pythonPath;

    for (const cmd of ["python3", "python"]) {
      try {
        const { execSync } = await import("node:child_process");
        const version = execSync(`${cmd} --version 2>&1`, { encoding: "utf-8" }).trim();
        if (version.includes("Python 3.")) return cmd;
      } catch {
        continue;
      }
    }
    return null;
  }

  private async checkHeadroomInstalled(pythonPath: string): Promise<boolean> {
    try {
      const { execSync } = await import("node:child_process");
      execSync(`${pythonPath} -c "import headroom"`, {
        encoding: "utf-8",
        stdio: "pipe",
      });
      return true;
    } catch {
      return false;
    }
  }

  private async findFreePort(): Promise<number> {
    const { createServer } = await import("node:net");
    return new Promise((resolve, reject) => {
      const server = createServer();
      server.listen(0, () => {
        const addr = server.address();
        if (addr && typeof addr === "object") {
          const port = addr.port;
          server.close(() => resolve(port));
        } else {
          reject(new Error("Could not find free port"));
        }
      });
    });
  }
}
