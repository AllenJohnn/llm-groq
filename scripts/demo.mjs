// LLM ShardX Complete Demonstration Launcher
// Starts both the static web server and local PeerServer signaling server,
// and optionally creates public Cloudflare Quick Tunnels for remote rooms.
// Usage:
//   npm run demo
//   node scripts/demo.mjs [--port 8080] [--signal-port 9000]

import http from "http";
import os from "os";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import readline from "readline";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const demoEnv = loadDemoEnv();

const HTTP_PORT = parseInt(process.argv.includes("--port") 
  ? process.argv[process.argv.indexOf("--port") + 1] 
  : (process.env.PORT || 8080), 10);

const SIGNAL_PORT = parseInt(process.argv.includes("--signal-port") 
  ? process.argv[process.argv.indexOf("--signal-port") + 1] 
  : (process.env.SIGNAL_PORT || 9000), 10);

function getLocalIp() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === "IPv4" && !iface.internal) {
        return iface.address;
      }
    }
  }
  return "127.0.0.1";
}

const localIp = getLocalIp();
const children = [];

console.log("\n╔══════════════════════════════════════════════════════════════════╗");
console.log("║               LLM ShardX Unified Demonstration Suite               ║");
console.log("║         Decentralized Browser-Native P2P LLM Inference           ║");
console.log("╚══════════════════════════════════════════════════════════════════╝\n");

// Check if static server (port 8080) is responding
async function isPortInUse(port) {
  return new Promise((resolve) => {
    const s = http.get(`http://127.0.0.1:${port}/`, (res) => {
      resolve(true);
    }).on("error", () => {
      resolve(false);
    });
    s.setTimeout(1000, () => {
      s.destroy();
      resolve(false);
    });
  });
}

async function startServers() {
  // 1. Signaling server (port 9000)
  const signalRunning = await isPortInUse(SIGNAL_PORT);
  let signalProcess = null;
  if (!signalRunning) {
    console.log(`[Demo] Starting Local Signaling Server on ws://localhost:${SIGNAL_PORT}...`);
    signalProcess = spawn("node", [path.join(ROOT, "scripts", "signal-server.mjs"), "--port", String(SIGNAL_PORT)], {
      env: withoutTunnelCredentials(process.env),
      stdio: ["ignore", "pipe", "pipe"],
    });
    children.push(signalProcess);
    signalProcess.stdout.on("data", (d) => process.stdout.write(d.toString()));
    signalProcess.stderr.on("data", (d) => process.stderr.write(d.toString()));
  } else {
    console.log(`[Demo] Local Signaling Server already active on port ${SIGNAL_PORT}.`);
  }

  // 2. Static server (port 8080)
  const httpRunning = await isPortInUse(HTTP_PORT);
  let httpProcess = null;
  if (!httpRunning) {
    console.log(`[Demo] Starting Web & Groq Proxy Server on http://localhost:${HTTP_PORT}...`);
    httpProcess = spawn("node", [path.join(ROOT, "scripts", "server.mjs"), "--port", String(HTTP_PORT)], {
      cwd: ROOT,
      env: withoutTunnelCredentials(process.env),
      stdio: ["ignore", "pipe", "pipe"],
    });
    children.push(httpProcess);
    httpProcess.stdout.on("data", (d) => process.stdout.write(d.toString()));
    httpProcess.stderr.on("data", (d) => process.stderr.write(d.toString()));
  } else {
    console.log(`[Demo] Static Web Server already active on port ${HTTP_PORT}.`);
  }

  // Wait 1.5s for servers to settle
  await new Promise((r) => setTimeout(r, 1500));

  const hostUrl = `http://localhost:${HTTP_PORT}/room?signal=localhost:${SIGNAL_PORT}&local-demo=1`;
  const workerLocalUrl = `http://localhost:${HTTP_PORT}/room?signal=localhost:${SIGNAL_PORT}&local-demo=1`;
  const workerLanUrl = `http://${localIp}:${HTTP_PORT}/room?signal=${localIp}:${SIGNAL_PORT}&local-demo=1`;

  const publicRoomUrl = await startPublicAccess();

  console.log("\n====================================================================");
  console.log("                    LIVE DEMONSTRATION ENDPOINTS                    ");
  console.log("====================================================================");
  console.log(`\n \x1b[1;36m🏠 HOST TAB (Single-System):\x1b[0m`);
  console.log(`   \x1b[4m${hostUrl}\x1b[0m`);
  console.log(`\n \x1b[1;32m💻 WORKER TAB (2nd Browser Tab / Split Screen):\x1b[0m`);
  console.log(`   \x1b[4${workerLocalUrl}\x1b[0m`);
  console.log(`\n \x1b[1;33m📱 MOBILE / LAN PEER (Phones / Other Laptops on Wi-Fi):\x1b[0m`);
  console.log(`   \x1b[4m${workerLanUrl}\x1b[0m`);
  if (publicRoomUrl) {
    const isFixed = Boolean(readDemoSetting("CLOUDFLARE_ROOM_URL") && readDemoSetting("CLOUDFLARE_SIGNAL_HOST"));
    console.log(`\n \x1b[1;35m🌐 PUBLIC ROOM (${isFixed ? "Cloudflare Named Tunnel" : "Cloudflare Quick Tunnel"}):\x1b[0m`);
    console.log(`   ${publicRoomUrl}`);
    console.log(isFixed
      ? "   Share this stable link with devices outside your Wi-Fi."
      : "   Share this link with devices outside your Wi-Fi; the temporary links change when the demo restarts.");
  } else {
    console.log("\n[Demo] Public link unavailable. Configure both public hostnames and a rotated tunnel token in .env, or install cloudflared for temporary links.");
  }
  console.log("\n====================================================================");
  console.log("                   SINGLE-SYSTEM DEMONSTRATION STEPS                ");
  console.log("====================================================================");
  console.log(" 1. Open HOST TAB in Chrome / Edge.");
  console.log(" 2. Enter device name (e.g. 'Laptop Host') and click 'Create room'.");
  console.log(" 3. Note the 4-letter Room Code (e.g. 'ABCD').");
  console.log(" 4. Open WORKER TAB in a second window placed side-by-side.");
  console.log(" 5. Enter device name (e.g. 'Worker Tab'), enter the Room Code, and click 'Join room'.");
  console.log(" 6. On the Host screen, select 'SmolLM 135M' or 'Qwen3 0.6B' and click 'Start Swarm'.");
  console.log(" 7. Watch the layers divide 50/50 between the two tabs and stream tokens collaboratively!\n");

  function openUrl(url) {
    const startCmd = process.platform === "win32" ? "start" : process.platform === "darwin" ? "open" : "xdg-open";
    spawn(startCmd, [url], { shell: true, detached: true });
  }

  console.log("Demonstration Controls:");
  console.log("  [h] Open Host Tab in browser");
  console.log("  [w] Open Worker Tab in browser");
  if (publicRoomUrl) console.log("  [p] Open public room link in browser");
  console.log("  [q] Quit Demo\n");

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  rl.on("line", (line) => {
    const cmd = line.trim().toLowerCase();
    if (cmd === "h") {
      console.log(`Opening Host: ${hostUrl}`);
      openUrl(hostUrl);
    } else if (cmd === "w") {
      console.log(`Opening Worker: ${workerLocalUrl}`);
      openUrl(workerLocalUrl);
    } else if (cmd === "p" && publicRoomUrl) {
      console.log(`Opening public room: ${publicRoomUrl}`);
      openUrl(publicRoomUrl);
    } else if (cmd === "q" || cmd === "exit") {
      console.log("Stopping demonstration servers...");
      stopChildren();
      process.exit(0);
    }
  });

  process.on("SIGINT", () => {
    console.log("\nShutting down demonstration servers...");
    stopChildren();
    process.exit(0);
  });
}

async function startPublicAccess() {
  const roomUrl = normalizeRoomUrl(readDemoSetting("CLOUDFLARE_ROOM_URL"));
  const signalHost = normalizeHostname(readDemoSetting("CLOUDFLARE_SIGNAL_HOST"));
  const token = readDemoSetting("CLOUDFLARE_TUNNEL_TOKEN");

  if (roomUrl && signalHost) {
    if (token) startNamedTunnel(token);
    else console.log("[Demo] Using fixed Cloudflare hostnames; assuming the named tunnel service is already running.");
    return `${roomUrl}/room?signal=${signalHost}:443&local-demo=1`;
  }

  if (token) {
    console.warn("[Demo] Named tunnel hostnames are missing; using temporary Cloudflare links. Set CLOUDFLARE_ROOM_URL and CLOUDFLARE_SIGNAL_HOST for a fixed link.");
  }

  // Cloudflare Quick Tunnels are temporary and need no account token. Tunnel
  // both the app and PeerServer so devices outside the LAN can reach signaling.
  const cloudflared = resolveCloudflared();
  const targets = [
    ["web", `http://127.0.0.1:${HTTP_PORT}`],
    ["signal", `http://127.0.0.1:${SIGNAL_PORT}`],
  ];

  return Promise.all(targets.map(([name, origin]) => new Promise((resolve) => {
    let settled = false;
    let output = "";
    const finish = (url) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      resolve(url);
    };
    const tunnel = spawn(cloudflared, ["tunnel", "--no-autoupdate", "--url", origin], {
      cwd: ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });
    children.push(tunnel);

    const readOutput = (chunk) => {
      output = (output + chunk.toString()).slice(-8192);
      const url = output.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/i)?.[0];
      if (url) finish(url);
    };
    tunnel.stdout.on("data", readOutput);
    tunnel.stderr.on("data", readOutput);
    tunnel.on("error", (error) => {
      if (error.code === "ENOENT") {
        console.warn("[Demo] cloudflared was not found. Install it or set CLOUDFLARED_BIN to its executable path.");
      } else {
        console.warn(`[Demo] Could not start the ${name} Cloudflare tunnel: ${error.message}`);
      }
      finish(null);
    });
    tunnel.on("exit", (code) => {
      if (!settled) {
        if (code) console.warn(`[Demo] The ${name} Cloudflare tunnel exited with code ${code}.`);
        finish(null);
      }
    });

    const timeout = setTimeout(() => {
      console.warn(`[Demo] Timed out waiting for the ${name} Cloudflare link.`);
      finish(null);
    }, 30000);
  }))).then(([web, signal]) => web && signal
    ? `${web}/room?signal=${new URL(signal).host}&local-demo=1`
    : null);
}

function startNamedTunnel(token) {
  const cloudflared = resolveCloudflared();
  const tunnelEnv = { ...process.env, TUNNEL_TOKEN: token };
  delete tunnelEnv.CLOUDFLARE_TUNNEL_TOKEN;
  const tunnel = spawn(cloudflared, ["tunnel", "--no-autoupdate", "run"], {
    cwd: ROOT,
    env: tunnelEnv,
    stdio: "inherit",
  });
  children.push(tunnel);
  tunnel.on("error", (error) => {
    console.error(`[Demo] Could not start the named Cloudflare tunnel: ${error.message}`);
  });
  tunnel.on("spawn", () => {
    console.log("[Demo] Started the configured named Cloudflare tunnel.");
  });
  tunnel.on("exit", (code) => {
    if (code && code !== 0) console.error(`[Demo] Named Cloudflare tunnel exited with code ${code}.`);
  });
}

function resolveCloudflared() {
  const configuredBinary = process.env.CLOUDFLARED_BIN || readCloudflaredBinFromEnv();
  if (configuredBinary) return configuredBinary;
  if (process.platform === "win32") {
    const roots = [process.env["ProgramFiles(x86)"], process.env.ProgramFiles].filter(Boolean);
    const installed = roots.map((root) => path.join(root, "cloudflared", "cloudflared.exe"))
      .find((candidate) => fs.existsSync(candidate));
    return installed || "cloudflared.exe";
  }
  return "cloudflared";
}

function readCloudflaredBinFromEnv() {
  return readDemoSetting("CLOUDFLARED_BIN");
}

function loadDemoEnv() {
  const values = {};
  try {
    for (const line of fs.readFileSync(path.join(ROOT, ".env"), "utf8").split(/\r?\n/)) {
      const match = line.match(/^\s*(CLOUDFLARE_TUNNEL_TOKEN|CLOUDFLARE_ROOM_URL|CLOUDFLARE_SIGNAL_HOST|CLOUDFLARED_BIN)\s*=\s*(.*)\s*$/);
      if (match && !values[match[1]]) values[match[1]] = unquoteEnvValue(match[2]);
    }
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  return values;
}

function readDemoSetting(name) {
  const value = (process.env[name] || demoEnv[name])?.trim();
  return value || null;
}

function withoutTunnelCredentials(env) {
  const childEnv = { ...env };
  delete childEnv.CLOUDFLARE_TUNNEL_TOKEN;
  delete childEnv.TUNNEL_TOKEN;
  return childEnv;
}

function unquoteEnvValue(value) {
  return value.replace(/^(?:"([\s\S]*)"|'([\s\S]*)')$/, (_match, double, single) => double ?? single);
}

function normalizeRoomUrl(value) {
  if (!value) return null;
  try {
    const url = new URL(value.includes("://") ? value : `https://${value}`);
    if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash) return null;
    return url.origin;
  } catch {
    return null;
  }
}

function normalizeHostname(value) {
  if (!value) return null;
  try {
    const url = new URL(value.includes("://") ? value : `https://${value}`);
    if (url.username || url.password || url.pathname !== "/" || url.search || url.hash) return null;
    return url.hostname;
  } catch {
    return null;
  }
}

function stopChildren() {
  for (const child of children) {
    if (child.exitCode === null && !child.killed) child.kill();
  }
}

startServers().catch((err) => {
  console.error("Demo failed to start:", err);
  process.exit(1);
});
