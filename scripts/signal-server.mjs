// Local PeerServer for LLM ShardX Zero-Config Offline & Local Signaling
// Eliminates reliance on public 0.peerjs.com, works completely offline.
// Usage:
//   node scripts/signal-server.mjs [--port 9000]

import { PeerServer } from "peer";

const port = parseInt(process.argv.includes("--port") 
  ? process.argv[process.argv.indexOf("--port") + 1] 
  : (process.env.SIGNAL_PORT || 9000), 10);

console.log("\n=======================================================");
console.log("       LLM ShardX Local Signaling Server (PeerServer)    ");
console.log("=======================================================\n");

const clients = new Map();

const peerServer = PeerServer({
  port,
  path: "/",
  corsOptions: {
    origin: true,
  },
}, (server) => {
  console.log(`[LLM ShardX Signal] Running on ws://localhost:${port}/ (HTTP: http://localhost:${port}/)`);
  console.log(`[LLM ShardX Signal] Connect clients via: ?signal=localhost:${port}\n`);
});

peerServer.on("connection", (client) => {
  const id = client.getId();
  clients.set(id, Date.now());
  const isHost = id.startsWith("webslice-room-");
  const type = isHost ? "🏠 HOST" : "💻 PEER";
  console.log(`[+] ${type} connected: ${id} (total active: ${clients.size})`);
});

peerServer.on("disconnect", (client) => {
  const id = client.getId();
  clients.delete(id);
  console.log(`[-] Disconnected: ${id} (total active: ${clients.size})`);
});

process.on("SIGINT", () => {
  console.log("\n[LLM ShardX Signal] Shutting down...");
  process.exit(0);
});
