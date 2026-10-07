// Start the local LLM ShardX server and, when configured, its Cloudflare Tunnel.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const envPath = path.join(ROOT, ".env");

// Load local settings without overriding variables already set by the shell.
try {
  if (typeof process.loadEnvFile === "function") {
    process.loadEnvFile(envPath);
  } else if (fs.existsSync(envPath)) {
    for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const separator = trimmed.indexOf("=");
      if (separator < 1) continue;
      const key = trimmed.slice(0, separator).trim();
      let value = trimmed.slice(separator + 1).trim();
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      if (!process.env[key]) process.env[key] = value;
    }
  }
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const serverEnv = { ...process.env };
// The tunnel credential is only needed by cloudflared, not the app server.
delete serverEnv.CLOUDFLARE_TUNNEL_TOKEN;
delete serverEnv.TUNNEL_TOKEN;

const server = spawn(process.execPath, [path.join(ROOT, "scripts", "server.mjs")], {
  cwd: ROOT,
  env: serverEnv,
  stdio: "inherit",
});
const children = [server];

if (process.env.CLOUDFLARE_TUNNEL_TOKEN) {
  const cloudflared = resolveCloudflared();
  const tunnelEnv = { ...process.env };
  const tunnelToken = tunnelEnv.CLOUDFLARE_TUNNEL_TOKEN;
  delete tunnelEnv.CLOUDFLARE_TUNNEL_TOKEN;
  tunnelEnv.TUNNEL_TOKEN = tunnelToken;
  const tunnel = spawn(cloudflared, ["tunnel", "--no-autoupdate", "run"], {
    cwd: ROOT,
    env: tunnelEnv,
    stdio: "inherit",
  });
  children.push(tunnel);
  tunnel.on("error", (error) => {
    console.error(`[LLM ShardX Tunnel] Could not start ${cloudflared}: ${error.message}`);
    if (error.code === "ENOENT") {
      console.error('[LLM ShardX Tunnel] Set CLOUDFLARED_BIN in .env to the full path of cloudflared.exe if it is installed outside the standard folders.');
    }
    stopChildren(1);
  });
  tunnel.on("spawn", () => {
    console.log("[LLM ShardX Tunnel] Starting the configured Cloudflare tunnel.");
  });
  tunnel.on("exit", (code) => {
    if (!stopping) {
      console.error(`[LLM ShardX Tunnel] Stopped${code === 0 ? "." : ` with exit code ${code ?? "unknown"}.`}`);
      stopChildren(code || 1);
    }
  });
} else {
  console.log("[LLM ShardX Tunnel] Not configured; serving locally only. Add CLOUDFLARE_TUNNEL_TOKEN to .env to start the tunnel with npm run serve.");
}

let stopping = false;
function stopChildren(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  process.exitCode = exitCode;
  for (const child of children) {
    if (child.exitCode === null && !child.killed) child.kill();
  }
}

server.on("error", (error) => {
  console.error(`[LLM ShardX Server] Could not start: ${error.message}`);
  stopChildren(1);
});
server.on("exit", (code) => {
  if (!stopping) stopChildren(code || 0);
});
process.on("SIGINT", () => stopChildren(0));
process.on("SIGTERM", () => stopChildren(0));

function resolveCloudflared() {
  if (process.env.CLOUDFLARED_BIN) return process.env.CLOUDFLARED_BIN;
  if (process.platform === "win32") {
    const installRoots = [process.env["ProgramFiles(x86)"], process.env.ProgramFiles].filter(Boolean);
    const candidates = installRoots.map((root) => path.join(root, "cloudflared", "cloudflared.exe"));
    candidates.push("C:\\Program Files (x86)\\cloudflared\\cloudflared.exe");
    const installedPath = candidates.find((candidate) => fs.existsSync(candidate));
    if (installedPath) return installedPath;
    return "cloudflared.exe";
  }
  return "cloudflared";
}
