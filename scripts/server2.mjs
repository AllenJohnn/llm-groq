// Zero-dependency local web & API server for LLM ShardX
// Serves static files, rewrites /room -> /p2p.html, and handles /api/groq proxy endpoint.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
// import groqHandler from "../api/groq.js";
// import configHandler from "../api/config.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// Load .env
try {
  if (typeof process.loadEnvFile === "function") {
    process.loadEnvFile(path.join(ROOT, ".env"));
  }
} catch {}
try {
  const envPath = path.join(ROOT, ".env");
  if (fs.existsSync(envPath)) {
    const text = fs.readFileSync(envPath, "utf8");
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eqIdx = trimmed.indexOf("=");
      if (eqIdx !== -1) {
        const k = trimmed.slice(0, eqIdx).trim();
        let v = trimmed.slice(eqIdx + 1).trim();
        if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
          v = v.slice(1, -1);
        }
        if (!process.env[k]) process.env[k] = v;
      }
    }
  }
} catch {}

const PORT = parseInt(
  process.argv.includes("--port")
    ? process.argv[process.argv.indexOf("--port") + 1]
    : process.env.PORT || 8080,
  10
);

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/plain; charset=utf-8",
  ".gguf": "application/octet-stream",
  ".safetensors": "application/octet-stream",
  ".bin": "application/octet-stream",
};

const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD");
  res.setHeader("Access-Control-Allow-Headers", "*");

  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  const urlObj = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  let pathname = decodeURIComponent(urlObj.pathname);

  // Route /api/groq to serverless handler
  if (pathname === "/api/groq" || pathname === "/api/groq.js") {
    try {
      await groqHandler(req, res);
    } catch (err) {
      if (!res.headersSent) {
        res.statusCode = 500;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ error: { message: err.message } }));
      }
    }
    return;
  }

  if (pathname === "/api/config") {
    const config = await configHandler();
    res.statusCode = config.status;
    for (const [name, value] of config.headers) res.setHeader(name, value);
    res.end(await config.text());
    return;
  }

  // Rewrites
  if (pathname === "/room" || pathname === "/room/") {
    pathname = "/p2p.html";
  } else if (pathname === "/") {
    pathname = "/index.html";
  }

  // Security: Prevent path traversal and block sensitive files (.env, .git, etc.)
  const safePath = path.normalize(path.join(ROOT, pathname));
  const baseName = path.basename(safePath);
  if (!safePath.startsWith(ROOT) || baseName.startsWith(".env") || baseName.startsWith(".git")) {
    res.statusCode = 403;
    res.end("Forbidden");
    return;
  }

  // Check file existence
  let filePath = safePath;
  try {
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      filePath = path.join(filePath, "index.html");
    }
  } catch {
    res.statusCode = 404;
    res.setHeader("Content-Type", "text/plain");
    res.end("404 Not Found");
    return;
  }

  try {
    const stat = fs.statSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || "application/octet-stream";

    // Handle range requests (for weights if requested)
    const range = req.headers.range;
    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      const chunksize = end - start + 1;
      const stream = fs.createReadStream(filePath, { start, end });
      res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${stat.size}`,
        "Accept-Ranges": "bytes",
        "Content-Length": chunksize,
        "Content-Type": contentType,
      });
      stream.pipe(res);
      return;
    }

    res.writeHead(200, {
      "Content-Length": stat.size,
      "Content-Type": contentType,
      "Accept-Ranges": "bytes",
    });

    if (req.method === "HEAD") {
      res.end();
      return;
    }

    fs.createReadStream(filePath).pipe(res);
  } catch (err) {
    res.statusCode = 500;
    res.end("Internal Server Error: " + err.message);
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`[LLM ShardX Server] Listening on http://localhost:${PORT}/ (and http://0.0.0.0:${PORT}/)`);
  console.log(`[LLM ShardX Server] Groq proxy active at http://localhost:${PORT}/api/groq`);
});
