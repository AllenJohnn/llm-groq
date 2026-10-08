// PaaS/container entrypoint for the web app and PeerJS signaling service.
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mode = (process.env.CONTAINER_SERVICE || "all").toLowerCase();
const webPort = process.env.PORT || "8080";
const signalPort = mode === "signal"
  ? process.env.PORT || process.env.SIGNAL_PORT || "9000"
  : process.env.SIGNAL_PORT || "9000";

if (!["all", "web", "signal"].includes(mode)) {
  console.error("CONTAINER_SERVICE must be 'all', 'web', or 'signal'.");
  process.exit(1);
}

if (mode === "all" && String(webPort) === String(signalPort)) {
  console.error(`Web and signaling ports must differ (both are ${webPort}).`);
  process.exit(1);
}

const children = [];
let stopping = false;

function start(name, script, args = []) {
  const child = spawn(process.execPath, [path.join(ROOT, script), ...args], {
    cwd: ROOT,
    env: process.env,
    stdio: "inherit",
  });
  children.push(child);
  child.on("error", (error) => {
    console.error(`[container] Could not start ${name}: ${error.message}`);
    shutdown(1);
  });
  child.on("exit", (code, signal) => {
    if (!stopping) {
      console.error(`[container] ${name} exited (code ${code ?? "none"}, signal ${signal ?? "none"}).`);
      shutdown(code || 1);
    }
  });
}

function shutdown(code = 0) {
  if (stopping) return;
  stopping = true;
  process.exitCode = code;
  for (const child of children) {
    if (child.exitCode === null && !child.killed) child.kill("SIGTERM");
  }
}

if (mode === "all" || mode === "web") {
  start("web server", "scripts/server.mjs", ["--port", String(webPort)]);
}
if (mode === "all" || mode === "signal") {
  start("signaling server", "scripts/signal-server.mjs", ["--port", String(signalPort)]);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
