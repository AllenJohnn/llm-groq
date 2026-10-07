// LLM ShardX Complete Demonstration Launcher
// Starts both the static web server and local PeerServer signaling server,
// and outputs exact URLs for single-system and multi-device demonstrations.
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
      stdio: ["ignore", "pipe", "pipe"],
    });
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
      stdio: ["ignore", "pipe", "pipe"],
    });
    httpProcess.stdout.on("data", (d) => process.stdout.write(d.toString()));
    httpProcess.stderr.on("data", (d) => process.stderr.write(d.toString()));
  } else {
    console.log(`[Demo] Static Web Server already active on port ${HTTP_PORT}.`);
  }

  // Wait 1.5s for servers to settle
  await new Promise((r) => setTimeout(r, 1500));

  const hostUrl = `http://localhost:${HTTP_PORT}/room?signal=localhost:${SIGNAL_PORT}`;
  const workerLocalUrl = `http://localhost:${HTTP_PORT}/room?signal=localhost:${SIGNAL_PORT}`;
  const workerLanUrl = `http://${localIp}:${HTTP_PORT}/room?signal=${localIp}:${SIGNAL_PORT}`;

  console.log("\n====================================================================");
  console.log("                    LIVE DEMONSTRATION ENDPOINTS                    ");
  console.log("====================================================================");
  console.log(`\n \x1b[1;36m🏠 HOST TAB (Single-System):\x1b[0m`);
  console.log(`   \x1b[4m${hostUrl}\x1b[0m`);
  console.log(`\n \x1b[1;32m💻 WORKER TAB (2nd Browser Tab / Split Screen):\x1b[0m`);
  console.log(`   \x1b[4${workerLocalUrl}\x1b[0m`);
  console.log(`\n \x1b[1;33m📱 MOBILE / LAN PEER (Phones / Other Laptops on Wi-Fi):\x1b[0m`);
  console.log(`   \x1b[4m${workerLanUrl}\x1b[0m`);
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
    } else if (cmd === "q" || cmd === "exit") {
      console.log("Stopping demonstration servers...");
      if (signalProcess) signalProcess.kill();
      if (httpProcess) httpProcess.kill();
      process.exit(0);
    }
  });

  process.on("SIGINT", () => {
    console.log("\nShutting down demonstration servers...");
    if (signalProcess) signalProcess.kill();
    if (httpProcess) httpProcess.kill();
    process.exit(0);
  });
}

startServers().catch((err) => {
  console.error("Demo failed to start:", err);
  process.exit(1);
});
