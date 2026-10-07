// LLM ShardX room: signaling, WebRTC mesh, layer assignment, weight streaming and the
// generation loop (prefill, decode, speculative verify). Served with p2p.html at /room.
import { autotuneCoop, makeTokenizer, DenseEngine, argmax, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests }
  from "./engine/engine.js";
import { f32ToF16, f16ToF32, parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
  ggmlLayerNames, qwen35Weights, qwen35ShardBytes, qwen35MtpBytes, qwen35LayerNames, tokenizerFromGGUF, cfgFromGGUF, gpuUploadEntry, streamEntryToGPU, validateRangeResponse }
  from "./engine/gguf.js";
import { Qwen35Engine } from "./engine/qwen35.js";
import { WIRE_F16, badF32, f32ToB64, packF16, unpackF16, asU16, packWire, unpackWire, asF32, b64ToF32 } from "./room/wire.js";
import { esc, md } from "./room/markdown.js?v=20261008-live-preview";
import { aiSample } from "./room/sampling.js";
import { chatRecipients } from "./room/visibility.js";
import { MODELS, NEED_GB, MAX_SEQ, MAX_NEW, MIN_ROOM, detectLocalModel, GROQ_MODEL_MAP, getGroqModelId } from "./room/models.js";
import { makeLink, attachWire, wireReady, sendFrame, resetLink } from "./room/transport.js";
import { pledgeOf, calculateClusterPledge, formatLayerRange, allocateLayers } from "./room/allocation.js";
import { getGroqApiKey, setGroqApiKey, loadBrowserEnv } from "./room/groq.js";
import { streamGroqChat, completeGroqChat, formatGroqError, GROQ_PROXY_URL } from "./room/groq-client.js";
import { perfSidebar } from "./room/perf-sidebar.js?v=20261007-graphs";
import { buildMode } from "./room/build-mode.js?v=20261008-build";

// Private presentation flag for screen recordings and personal demos: /room?local-demo=1
const LOCAL_DEMO_PRESENTATION = new URLSearchParams(location.search).get("local-demo") === "1";

// Local WebGPU is the default, but keep a user's explicit mode choice.
function initialCloudMode() {
  try {
    const saved = localStorage.getItem("webslice_fallbackmode");
    if (saved === "true") return true;
    if (saved === "false") return false;
  } catch {}
  return false;
}
var isGroqMode = LOCAL_DEMO_PRESENTATION || initialCloudMode();
var fallbackmode = isGroqMode;
if (typeof window !== "undefined") {
  window.isGroqMode = isGroqMode;
  window.fallbackmode = fallbackmode;
  loadBrowserEnv().then((env) => {
    if (env.GROQ_API_KEY) {
      setGroqApiKey(env.GROQ_API_KEY);
    }
    let savedMode = null;
    try { savedMode = localStorage.getItem("webslice_fallbackmode"); } catch {}
    if (!LOCAL_DEMO_PRESENTATION && savedMode === null && typeof env.FALLBACKMODE === "boolean" && env.FALLBACKMODE !== isGroqMode) {
      toggleFallbackMode(env.FALLBACKMODE);
    }
  }).catch(() => {});
}

// Hidden-state transport (room/transport.js). ?wire=off falls back to PeerJS messages;
// ?wire=slice uses one sliced channel; ?wire=stripeN spreads slices over N peer connections.
const WIRE = (new URLSearchParams(location.search).get("wire") || "slice").toLowerCase();
const WIRE_STRIPES = WIRE === "off" ? 0 : WIRE.startsWith("stripe") ? Math.max(1, Math.min(8, parseInt(WIRE.slice(6), 10) || 1)) : 1;
// Signaling: ?signal=host:port points PeerJS at our own PeerServer (the emulator and big
// rooms use one); default is the public PeerJS cloud.
const SIGNAL = new URLSearchParams(location.search).get("signal");
const SIGNAL_OPTS = SIGNAL ? (() => { const [host, port] = SIGNAL.split(":"); return { host, port: +port || 443, path: "/", secure: location.protocol === "https:" }; })() : {};

// Topology: every device keeps ONE link to the host (control, roster, tokens). Data links
// between chain neighbours open when the layers are dealt (ensureLink), so a room of N
// devices has N-1 host links plus N-1 chain links, not N*(N-1)/2. Workers learn about the
// other devices from the host's roster message and draw cards from it.
const members = new Map();   // id -> { name, meta } for everyone in the room except me
const cards = new Map();     // id -> card element

const $ = (id) => document.getElementById(id);
function toast(text) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = text;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}
function mascot() {}
const PREFIX = "webslice-room-";
const rand = (n) => Array.from(crypto.getRandomValues(new Uint8Array(n)))
  .map(b => "ABCDEFGHJKMNPQRSTVWXYZ23456789"[b % 30]).join("");

let peer = null;          // my PeerJS peer
let isHost = false;
let roomCode = null;
let myName = null;
let myMeta = {};
// conns: peerId -> { conn, name, meta, rtt, mbps, card }
const conns = new Map();
// host only: roster of member peer ids -> {name, meta}
const roster = new Map();

let ai = {
  visibility: "all",   // who sees the chat: all | host | asker (room/visibility.js)
  engine: null, tok: null, cfg: null, device: null,
  role: null,            // "host" | "worker"
  chain: [],             // host: worker peer ids in pipeline order
  next: null,            // worker: peer id to forward hidden to, or "host"
  readyPeers: new Set(),
  pos: 0,
  waiters: new Map(),    // pos -> resolve(hiddenF32) for host awaiting return
  busy: false,
};

// --- GPU capability probe (runs at page load so the join screen can offer
// contribution presets) ---
async function probeGPU() {
  const meta = { ua: navigator.userAgent.includes("iPhone") ? "iPhone" :
                     navigator.userAgent.includes("Mac") ? "Mac" :
                     navigator.userAgent.includes("Android") ? "Android" : "Device",
                 webgpu: false, gpu: "no WebGPU", maxBufGB: 0 };
  if (navigator.gpu) {
    try {
      const a = await Promise.race([
        navigator.gpu.requestAdapter(),
        new Promise((r) => setTimeout(() => r(null), 2500))
      ]);
      if (a) {
        meta.webgpu = true;
        const info = a.info || {};
        meta.gpu = [...new Set([info.vendor, info.architecture || info.device].filter(Boolean))].join(" ") || "GPU";
        meta.maxBufGB = +(a.limits.maxBufferSize / 2 ** 30).toFixed(1);
        // browsers hide real GPU memory (fingerprinting). Default to the
        // conservative per-buffer limit; the user can opt in to a real
        // measurement (see measureBudgetGB) which replaces this estimate.
        meta.budgetGB = meta.maxBufGB;
        meta.canMeasure = meta.ua !== "iPhone" && meta.ua !== "Android";
      }
    } catch {}
  }
  return meta;
}

async function measureBudgetGB(adapter, capGB) {
  try {
    const dev = await adapter.requestDevice();
    let lost = false;
    dev.lost.then(() => { lost = true; });
    const chunk = 512 * 2 ** 20;
    const bufs = [];
    let total = 0;
    while (total < capGB * 2 ** 30 && !lost) {
      dev.pushErrorScope("out-of-memory");
      const b = dev.createBuffer({ size: chunk, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      try { // commit the pages for real, or lazy allocation lies to us
        const enc = dev.createCommandEncoder();
        enc.clearBuffer(b);
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
      } catch { lost = true; }
      const err = await dev.popErrorScope().catch(() => true);
      if (err || lost) { try { b.destroy(); } catch {} break; }
      bufs.push(b);
      total += chunk;
    }
    for (const b of bufs) { try { b.destroy(); } catch {} }
    try { dev.destroy(); } catch {}
    return +(total / 2 ** 30).toFixed(1);
  } catch { return 0; }
}

// probe once at load; fill the contribution selector
const metaPromise = (async () => {
  try {
    const m = await probeGPU();
    if (m.webgpu) {
      const sysMem = navigator.deviceMemory || 0;
      let rec = m.budgetGB ? m.budgetGB * 0.5 : 1;
      if (sysMem >= 64) rec = 48;
      else if (sysMem >= 32) rec = 24;
      else if (sysMem >= 16) rec = 12;
      else if (sysMem >= 8) rec = 6;
      m.contribGB = Math.max(0.2, Math.round(rec * 10) / 10);
    } else {
      m.contribGB = 0;
    }
    m.phone = m.ua === "iPhone" || m.ua === "Android";
    const gbEl = $("join-gb");
    if (gbEl) {
      if (!m.webgpu) {
        gbEl.value = "0";
        gbEl.disabled = true;
        $("gb-minus").disabled = true;
        $("gb-plus").disabled = true;
      } else {
        $("gb-minus").disabled = false;
        $("gb-plus").disabled = false;
        if (m.phone) { m.contribGB = 0.5; gbEl.min = "0.5"; gbEl.step = "0.5"; gbEl.value = "0.5"; }
        else if (m.contribGB) { m.contribGB = Math.max(1, m.contribGB); gbEl.value = m.contribGB; }
      }
    }
    const capability = $("join-device-status");
    if (capability) {
      capability.dataset.ready = String(!!m.webgpu);
      capability.textContent = m.webgpu
        ? `WebGPU ready · set to contribute ${m.contribGB} GB.`
        : "WebGPU is not available in this browser, so this device cannot contribute model memory.";
    }
    return m;
  } catch (e) {
    return { ua: "Device", webgpu: false, gpu: "no WebGPU", maxBufGB: 0, contribGB: 0 };
  }
})();

// --- UI helpers ---
function log(from, text) {
  const div = document.createElement("div");
  div.innerHTML = `<b></b> `;
  div.querySelector("b").textContent = from;
  div.appendChild(document.createTextNode(text));
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

function peerCard(id, name, meta, self) {
  const card = document.createElement("div");
  card.className = "peer-card" + (self ? " self" : "");
  card.innerHTML = `
    <div class="peer-card-head">
      <div class="peer-name"><span class="dot ${self ? "ok" : "warn"}"></span><span class="pname"></span></div>
      <span class="device-state ${self ? "connected" : "joining"}">${self ? "This device" : "Connecting"}</span>
    </div>
    <div class="peer-gpu"></div>
    <div class="peer-capabilities"><span class="peer-gpu-badge"></span><span class="peer-memory"><span class="memory-label">Memory</span> <b class="buf">—</b></span></div>
    <div class="peer-stats">
      <span>rtt <b class="rtt">—</b></span>
      <span>bw <b class="bw">—</b></span>
    </div>
    ${self ? "" : '<button class="bw-btn">test bandwidth</button>'}`;
  card.querySelector(".pname").textContent = name + (self ? " (you)" : "");
  card.querySelector(".peer-gpu").textContent = meta.webgpu
    ? `${meta.ua} · ${meta.gpu}` : `${meta.ua} · ⚠ no WebGPU`;
  card.querySelector(".peer-gpu-badge").textContent = meta.webgpu ? "WebGPU ready" : "WebGPU unavailable";
  card.querySelector(".peer-gpu-badge").classList.toggle("unavailable", !meta.webgpu);
  const budget = meta.budgetGB || meta.maxBufGB;
  card.querySelector(".buf").textContent = meta.contribGB ? `${meta.contribGB} GB pledged` : (budget ? `${budget} GB available` : "Not reported");
  $("peers").appendChild(card);
  if (!self) card.querySelector(".bw-btn").addEventListener("click", () => bwTest(id));
  return card;
}

function updateTopbarPeers() {
  const container = $("topbar-peers");
  if (!container) return;
  if ($("room-screen")?.style.display !== "flex") {
    container.classList.remove("on");
    return;
  }
  container.classList.add("on");

  const list = [];
  list.push({
    id: "self",
    name: myName || "you",
    meta: myMeta || {},
    self: true,
    rtt: null,
    bw: null
  });

  for (const [id, m] of members.entries()) {
    const c = conns.get(id);
    list.push({
      id,
      name: m.name || c?.name || id,
      meta: m.meta || c?.meta || {},
      self: false,
      rtt: c?.rtt ?? null,
      bw: c?.bw ?? null
    });
  }

  const isDownloading = !!(ai.isDownloading || $("load-card")?.classList.contains("on"));

  container.innerHTML = list.map((p) => {
    const isSelf = p.self;
    const dotClass = isSelf ? "ok" : (p.rtt !== null ? "ok" : "warn");
    const budget = p.meta.contribGB ? p.meta.contribGB + " GB" : (p.meta.budgetGB || p.meta.maxBufGB ? (p.meta.budgetGB || p.meta.maxBufGB) + " GB" : "");
    const metaTag = budget ? budget : (p.meta.webgpu ? "GPU" : "no GPU");

    let progHtml = "";
    if (isDownloading) {
      const pct = (ai.prog || {})[p.name] ?? (isSelf ? Math.round(ai.myPct || 0) : 0);
      const isDone = pct >= 100;
      progHtml = `<span class="tb-prog active${isDone ? " done" : ""}">${isDone ? "ready" : pct + "%"}</span>`;
    }

    const titleInfo = [
      p.name + (isSelf ? " (you)" : ""),
      p.meta.ua || "",
      p.meta.gpu || (p.meta.webgpu ? "WebGPU" : "No WebGPU"),
      budget ? "Gives " + budget : "",
      p.rtt ? `RTT: ${p.rtt}ms` : "",
      p.bw ? `BW: ${p.bw}` : ""
    ].filter(Boolean).join(" · ");

    return `<div class="topbar-peer-chip${isSelf ? " self" : ""}" title="${esc(titleInfo)}" data-id="${esc(p.id)}">
      <span class="dot ${dotClass}"></span>
      <span class="tb-name">${esc(p.name)}${isSelf ? " (you)" : ""}</span>
      ${progHtml ? progHtml : (metaTag ? `<span class="tb-meta">${esc(metaTag)}</span>` : "")}
    </div>`;
  }).join("");

  container.querySelectorAll(".topbar-peer-chip").forEach((chip) => {
    const id = chip.dataset.id;
    if (id && id !== "self") {
      chip.style.cursor = "pointer";
      chip.title += " (click to test bandwidth)";
      chip.addEventListener("click", () => bwTest(id));
    } else if (id === "self" && myMeta.webgpu) {
      chip.style.cursor = "pointer";
      chip.title += " (click to change GPU memory pledge)";
      chip.addEventListener("click", () => {
        const cur = myMeta.contribGB || 1;
        const input = prompt(`Allocate GPU memory (GB) for ${myName}:`, cur);
        if (input !== null) {
          const v = parseFloat(input);
          if (!isNaN(v) && v >= (myMeta.phone ? 0.5 : 1) && v <= 64) {
            myMeta.contribGB = v;
            const selfCard = document.querySelector(".peer-card.self");
            if (selfCard) {
              const buf = selfCard.querySelector(".buf");
              if (buf) buf.textContent = v + " GB pledged";
              const pInput = selfCard.querySelector(".pledge input");
              if (pInput) pInput.value = v;
            }
            updateCluster();
            broadcastAll({ t: "pledge", gb: v });
          }
        }
      });
    }
  });
}

let tbDoneTimer = null;
function updateTopbarDownload(show, done = 0, total = 0, note = "") {
  const el = $("topbar-download");
  const line = $("topbar-progress-line");
  const fill = $("topbar-progress-fill");
  if (!el || !line || !fill) return;

  if (tbDoneTimer) { clearTimeout(tbDoneTimer); tbDoneTimer = null; }

  if (!show) {
    ai.isDownloading = false;
    if (ai.engine) {
      el.classList.add("ready");
      const pctEl = el.querySelector(".tb-dl-pct");
      if (pctEl) pctEl.textContent = "ready";
      fill.style.width = "100%";
      tbDoneTimer = setTimeout(() => {
        el.classList.remove("on", "ready");
        line.classList.remove("active");
        fill.style.width = "0%";
      }, 1800);
    } else {
      el.classList.remove("on", "ready");
      line.classList.remove("active");
      fill.style.width = "0%";
    }
    updateTopbarPeers();
    return;
  }

  ai.isDownloading = true;
  el.classList.remove("ready");
  el.classList.add("on");
  line.classList.add("active");

  const modelKey = $("ai-model")?.value;
  const modelName = (MODELS[modelKey]?.label || "Model").split("\u00b7")[0].trim();

  let pct = 0;
  if (total && total > 0) {
    pct = Math.min(100, Math.round((done / total) * 100));
  } else if (ai.myPct) {
    pct = Math.min(100, Math.round(ai.myPct));
  }

  const bytesText = (total && total > 0)
    ? `${(done / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB`
    : (note ? note : (pct ? `${pct}%` : "syncing…"));

  el.innerHTML = `
    <div class="tb-dl-icon"></div>
    <span class="tb-dl-name">${esc(modelName)}</span>
    <div class="tb-dl-bar"><div class="tb-dl-fill" style="width:${pct}%"></div></div>
    <span class="tb-dl-pct">${pct}%</span>
    <span class="tb-dl-bytes">${esc(bytesText)}</span>
  `;

  fill.style.width = pct + "%";
  updateTopbarPeers();
}

let wasReady = false;
function updateNeed(pledged) {
  const model = $("ai-model")?.value || "qwen3.8-27b";
  if (isGroqMode) {
    if ($("need-fill")) $("need-fill").style.width = "100%";
    if ($("need-text")) $("need-text").textContent = " ";
    if ($("ai-need")) $("ai-need").classList.add("ok");
    if ($("ai-start") && !ai.busy) $("ai-start").disabled = false;
    return;
  }
  const need = NEED_GB[$("ai-model")?.value] || 1;
  const ok = pledged >= need;
  if ($("need-fill")) $("need-fill").style.width = Math.min(100, pledged / need * 100).toFixed(1) + "%";
  if ($("need-text")) $("need-text").textContent = ok
    ? `needs ~${need} GB · room gives ${pledged.toFixed(1)} GB · ready`
    : `needs ~${need} GB · room gives ${pledged.toFixed(1)} GB · add ${(need - pledged).toFixed(1)} GB more`;
  if ($("ai-need")) $("ai-need").classList.toggle("ok", ok);
  if ($("ai-start") && !ai.busy && !ai.engine) $("ai-start").disabled = !ok;
  if (ok && !wasReady) {
    $("ai-start")?.classList.remove("unlocked");
    void $("ai-start")?.offsetWidth;
    $("ai-start")?.classList.add("unlocked");
  }
  wasReady = ok;
}
$("ai-model").addEventListener("change", () => {
  const model = $("ai-model").value;
  ai.model = model;
  if (isGroqMode) {
    aiStatus(LOCAL_DEMO_PRESENTATION
      ? `selected · ${MODELS[model]?.label?.split("·")[0]?.trim() || model}`
      : `Groq Cloud · ${getGroqModelId(model)} · prompts sent to Groq`);
    updateGroqModelBadge();
    updateFallbackModeUI(true);
  } else if (!ai.engine) {
    const need = NEED_GB[model] || 1;
    aiStatus(`About ${need} GB of GPU memory needed · choose Load model when ready`);
  }
  updateCluster();
});
function updateGroqModelBadge() {
  const badge = $("model-groq-badge");
  if (!badge) return;
  const model = $("ai-model")?.value || "qwen3.8-27b";
  badge.textContent = LOCAL_DEMO_PRESENTATION
    ? `${MODELS[model]?.label?.split("·")[0]?.trim() || "Local model"} · ready`
    : isGroqMode ? `Groq Cloud · ${getGroqModelId(model)}` : "Local WebGPU";
  badge.style.display = "inline-block";
}
function updateCluster() {
  const all = [myMeta, ...[...members.values()].map(m => m.meta)];
  const gpus = all.filter(m => m && m.webgpu).length;
  const countEl = $("devices-count");
  if (countEl) countEl.textContent = `${all.length} online`;
  const pledged = calculateClusterPledge(myMeta, [...members.values()].map(m => m.meta));
  updateNeed(pledged);
  const mem = all.filter(m => m && m.webgpu).reduce((s, m) => s + (m?.budgetGB || m?.maxBufGB || 0), 0);
  $("cluster-summary").textContent =
    `${all.length} device${all.length > 1 ? "s" : ""} · ${gpus} WebGPU · ${pledged.toFixed(1)} GB pledged`;
  updateTopbarPeers();

  let hostDevId = ai?.role === "host" ? "self" : (ai?.hostId || null);

  const devices = [
    {
      id: "self",
      name: myName || "you",
      self: true,
      meta: myMeta || {},
      rtt: null,
      bw: null,
      layers: !isGroqMode && ai?.layersByName ? ai.layersByName[myName] : null,
      workerRole: (!isGroqMode && ai?.layersByName && ai.layersByName[myName]) ? (hostDevId === "self" ? "Host" : "Worker") : "Idle"
    },
    ...[...members.entries()].map(([id, m]) => {
      const c = conns.get(id);
      const name = m.name || c?.name || id;
      return {
        id,
        name,
        self: false,
        meta: m.meta || c?.meta || {},
        rtt: c?.rtt ?? null,
        bw: c?.bw ?? null,
        layers: !isGroqMode && ai?.layersByName ? ai.layersByName[name] : null,
        workerRole: (!isGroqMode && ai?.layersByName && ai.layersByName[name]) ? (hostDevId === id ? "Host" : "Worker") : "Idle"
      };
    })
  ];
  perfSidebar.setDevices(devices);
}

function enterRoom() {
  $("join-screen").style.display = "none";
  $("room-screen").style.display = "flex";
  $("room-badge").style.display = "block";
  $("room-badge").textContent = roomCode;
  $("side-code").textContent = roomCode;
  $("side-code").addEventListener("click", copyRoomLink);
  if (isHost) {
    if ($("host-controls")) $("host-controls").hidden = false;
    const splitBtn = $("split-demo-btn");
    if (splitBtn) {
      splitBtn.style.display = "block";
      splitBtn.onclick = () => {
        const signalParam = SIGNAL ? `&signal=${encodeURIComponent(SIGNAL)}` : "";
        const url = `${location.origin}${location.pathname}?code=${roomCode}${signalParam}`;
        window.open(url, "_blank");
      };
    }
  }
  peerCard("self", myName, myMeta, true);
  perfSidebar.init();
  updateCluster();
  log("LLM ShardX", `room ${roomCode} — share this code with your other devices`);
  $("ai-panel").style.display = "flex";
  aiStatus("");
  $("ai-empty").textContent = "pick a model and press start, from any device";
  if (isGroqMode) {
    ai.role = "host";
    const m = $("ai-model")?.value || "qwen3.8-27b";
    ai.model = m;
    $("ai-panel").classList.add("groq-mode");
    if (LOCAL_DEMO_PRESENTATION) {
      $("ai-panel").classList.remove("online");
      $("ai-row").style.display = "none";
      $("ai-empty").style.display = "";
      $("ai-empty").textContent = "Choose a model and press Start to load it from cache.";
      $("ai-start").style.display = "";
      $("ai-start").textContent = "start · load from cache";
      aiStatus("choose a model and press start");
    } else {
      $("ai-row").style.display = "flex";
      $("ai-empty").style.display = "none";
      $("ai-panel").classList.add("online");
      aiStatus(`Groq Cloud ready · ${getGroqModelId(m)} · prompts sent to Groq`);
      renderWelcomePrompts();
    }
    updateGroqModelBadge();
  }
  const selfCard = document.querySelector(".peer-card.self");
  if (selfCard && myMeta.webgpu) {
    const row = document.createElement("div");
    row.className = "pledge";
    row.innerHTML = `give <input type="number" min="1" max="64" step="1" value="${myMeta.contribGB}"> GB of GPU`;
    selfCard.appendChild(row);
    row.querySelector("input").addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      if (v >= (myMeta.phone ? 0.5 : 1)) { myMeta.contribGB = v; selfCard.querySelector(".buf").textContent = v + " GB pledged"; updateCluster(); broadcastAll({ t: "pledge", gb: v }); }
    });
  }
}

// --- connection wiring ---
function wire(conn, name, meta, initiator = false) {
  const entry = { conn, name: name || conn.peer, meta: meta || {}, rtt: null, card: null, link: makeLink(), stripes: [] };
  conns.set(conn.peer, entry);
  if (WIRE_STRIPES > 0) {
    attachWire(entry.link, conn, (m) => onData(conn.peer, m));
    // extra associations for striping: the side that dialed opens them, the other side accepts
    // them in peer.on("connection") by label and attaches its end of the wire channel
    if (initiator) for (let i = 1; i < WIRE_STRIPES; i++) {
      const sc = peer.connect(conn.peer, { reliable: true, label: "stripe" });
      sc.on("open", () => { attachWire(entry.link, sc, (m) => onData(conn.peer, m)); });
      sc.on("error", () => {});
      entry.stripes.push(sc);
    }
  }

  conn.on("data", (d) => onData(conn.peer, d));
  conn.on("close", () => {
    const e = conns.get(conn.peer);
    conns.delete(conn.peer);
    if (ai.chain) {
      // Do NOT filter ai.chain. It represents the required topology.
      ai.readyPeers.delete(conn.peer);
      if (ai.waiters && ai.waiters.size > 0) {
        for (const [key, waiter] of ai.waiters) {
          ai.waiters.delete(key);
          if (typeof waiter?.reject === "function") waiter.reject(new Error(`peer ${e?.name || conn.peer} disconnected`));
          else if (typeof waiter === "function") waiter(null);
        }
      }
      if (isHost && ai.role === "host" && ai.engine) {
        const trueSolo = ai.range && ai.range[0] === 0 && ai.range[1] >= (ai.cfg?.num_hidden_layers || 999);
        if (!trueSolo && ai.chain.includes(conn.peer)) {
          ai.clusterDegraded = true;
          ai.abortGen = true;
          aiStatus(`Cluster degraded \u2014 worker ${e?.name || conn.peer} disconnected. Reload model to recover.`);
          log("LLM ShardX", `Cluster degraded \u2014 missing required worker ${e?.name || conn.peer}`);
        }
        aiMaybeReady();
      }
    }
    if (isHost) {
      dropCard(conn.peer); members.delete(conn.peer); roster.delete(conn.peer); broadcastRoster();
      log("LLM ShardX", `${e?.name || conn.peer} left`);
    } else if (conn.peer === ai.hostId || (e && e.name === "host")) log("LLM ShardX", "lost the link to the host");
    updateCluster();
  });
  conn.on("error", () => {});
  return entry;
}

function ensureCard(id, name, meta) {
  let card = cards.get(id);
  if (!card) {
    card = peerCard(id, name || id, meta || {}, false);
    cards.set(id, card);
    updateCluster();
    log("LLM ShardX", `${name || id} joined`);
    mascot(`${name || id} joined! ${members.size + 1} devices in the room.`);
  }
  const e = conns.get(id);
  if (e) e.card = card;
  return card;
}
function dropCard(id) { const c = cards.get(id); if (c) { c.remove(); cards.delete(id); } updateTopbarPeers(); }
// open a data link to a chain neighbour if we do not have one yet; resolves when it is up
function ensureLink(id, timeoutMs = 60000) {
  if (!id || id === "host" || conns.has(id)) return Promise.resolve(true);
  if (!ensureLink.pending.has(id)) { ensureLink.pending.add(id); meshConnect(id); }
  return new Promise((res) => {
    const t0 = performance.now();
    const t = setInterval(() => {
      if (conns.has(id)) { clearInterval(t); ensureLink.pending.delete(id); res(true); }
      else if (performance.now() - t0 > timeoutMs) { clearInterval(t); ensureLink.pending.delete(id); res(false); }
    }, 100);
  });
}
ensureLink.pending = new Set();

function sendTo(id, obj) {
  const e = conns.get(id);
  if (!e || !e.conn) return false;
  try {
    e.conn.send(obj);
    return true;
  } catch (err) {
    console.warn("sendTo failed for peer", id, err);
    return false;
  }
}
// debug: per-peer wire state (channels open, frames sent/received) — `swarmDebug()` in the console
window.websliceDebug = () => [...conns].map(([id, e]) => ({ id, name: e.name, chans: e.link?.chans.filter((c) => c.readyState === "open").length ?? 0, sent: e.link?.sent ?? 0, recv: e.link?.recv ?? 0 }));
// activations go over the sliced wire channel when it is up, else as a normal message
function sendHidden(id, msg) {
  const e = conns.get(id);
  if (!e) return false;
  try {
    if (e.link && wireReady(e.link) && sendFrame(e.link, msg)) return true;
  } catch (err) {
    console.warn("sendHidden wire failed, falling back to sendTo:", err);
  }
  return sendTo(id, msg);
}
function broadcastAll(obj) { for (const [id] of conns) sendTo(id, obj); }

// bandwidth test state
const bwRecv = new Map(); // fromId -> {bytes, t0}

function onData(from, d) {
  // binary chunk = bandwidth test payload
  if (d instanceof ArrayBuffer || ArrayBuffer.isView(d)) {
    const st = bwRecv.get(from);
    if (st) st.bytes += d.byteLength || d.length;
    return;
  }
  const e = conns.get(from);
  if (d.t && d.t.startsWith("ai-")) { aiOnData(from, d); return; }
  switch (d.t) {
    case "hello":
      e.name = d.name; e.meta = d.meta;
      members.set(from, { name: d.name, meta: d.meta });
      ensureCard(from, d.name, d.meta);
      if (isHost) {
        roster.set(from, { name: d.name, meta: d.meta }); broadcastRoster();
        aiRejoin(from, d.name);
        if (ai.visibility !== "all") sendTo(from, { t: "ai-visibility", mode: ai.visibility });
      }
      break;
    case "roster": {
      // the host's view of the room: draw a card per device, no mesh connections
      const seen = new Set();
      for (const m of d.members) {
        if (m.id === peer.id) continue;
        seen.add(m.id);
        members.set(m.id, { name: m.name, meta: m.meta });
        const c = ensureCard(m.id, m.name, m.meta);
        if (m.meta?.contribGB) c.querySelector(".buf").textContent = m.meta.contribGB + " GB pledged";
        const ce = conns.get(m.id); if (ce) ce.meta = m.meta;
      }
      for (const id of [...members.keys()]) if (!seen.has(id)) { members.delete(id); dropCard(id); }
      updateCluster();
      break;
    }
    case "ping": sendTo(from, { t: "pong", ts: d.ts }); break;
    case "pong": {
      e.rtt = Math.round(performance.now() - d.ts);
      if (e.card) {
        e.card.querySelector(".rtt").textContent = e.rtt + " ms";
        const state = e.card.querySelector(".device-state");
        if (state) { state.textContent = "Connected"; state.className = "device-state connected"; }
        e.card.querySelector(".dot")?.classList.replace("warn", "ok");
      }
      break;
    }
    case "pledge":
      if (e) { e.meta = { ...e.meta, contribGB: d.gb }; if (e.card) e.card.querySelector(".buf").textContent = d.gb + " GB pledged"; }
      if (members.has(from)) members.get(from).meta = { ...members.get(from).meta, contribGB: d.gb };
      if (isHost && roster.has(from)) { roster.get(from).meta = { ...roster.get(from).meta, contribGB: d.gb }; broadcastRoster(); }
      updateCluster();
      break;
    case "bw-start": bwRecv.set(from, { bytes: 0, t0: performance.now() }); break;
    case "bw-end": {
      const st = bwRecv.get(from);
      if (st) {
        const secs = (performance.now() - st.t0) / 1000;
        const mbps = (st.bytes * 8 / 1e6 / secs).toFixed(0);
        sendTo(from, { t: "bw-result", mbps });
        bwRecv.delete(from);
      }
      break;
    }
    case "bw-result":
      if (e.card) e.card.querySelector(".bw").textContent = d.mbps + " Mbps";
      log("LLM ShardX", `bandwidth to ${e.name}: ${d.mbps} Mbps`);
      break;
  }
}

function broadcastRoster() {
  const members = [{ id: peer.id, name: myName, meta: myMeta },
    ...[...roster.entries()].map(([id, m]) => ({ id, ...m }))];
  broadcastAll({ t: "roster", members });
}

function meshConnect(targetId) {
  const conn = peer.connect(targetId, { reliable: true });
  const handleOpen = () => {
    wire(conn, undefined, undefined, true);
    conn.send({ t: "hello", name: myName, meta: myMeta });
  };
  if (conn.open) handleOpen();
  else conn.on("open", handleOpen);
  conn.on("error", (err) => console.warn("meshConnect error to " + targetId, err));
}

async function bwTest(id) {
  const e = conns.get(id);
  if (!e) return;
  log("LLM ShardX", `testing bandwidth to ${e.name}…`);
  sendTo(id, { t: "bw-start" });
  const chunk = new Uint8Array(64 * 1024);
  const total = 4 * 1024 * 1024;
  for (let sent = 0; sent < total; sent += chunk.length) {
    e.conn.send(chunk);
    // yield so the datachannel buffer can drain
    if (e.conn.dataChannel && e.conn.dataChannel.bufferedAmount > 1 << 20)
      await new Promise(r => setTimeout(r, 20));
  }
  sendTo(id, { t: "bw-end" });
}

// --- ping loop ---
setInterval(() => broadcastAll({ t: "ping", ts: performance.now() }), 2500);

const stepGB = (d) => { const i = $("join-gb"); const lo = parseFloat(i.min) || 1; const st = parseFloat(i.step) || 1; i.value = Math.min(64, Math.max(lo, (parseFloat(i.value) || lo) + d * st)); };
window.stepGB = stepGB;
$("gb-minus").addEventListener("click", () => stepGB(-1));
$("gb-plus").addEventListener("click", () => stepGB(1));
// --- join / create ---
async function start(create) {
  // Both the inline loading bridge and the module's click listener can fire for one click.
  // The first call disables both buttons synchronously; ignore any duplicate invocation.
  if ($("create-btn").disabled || $("join-btn").disabled) return;
  window.__roomStart = start;
  try {
    myName = $("name-input").value.trim() || (create ? "host" : "peer") + "-" + rand(2);
    const code = create ? rand(4) : $("code-input").value.trim().toUpperCase();
    if (!code) { $("join-status").textContent = "enter a room code"; return; }
    if (typeof Peer === "undefined") {
      $("join-status").textContent = "Error: PeerJS library not loaded. Check connection or ad-blocker.";
      $("create-btn").disabled = $("join-btn").disabled = false;
      return;
    }
    $("create-btn").disabled = $("join-btn").disabled = true;
    $("join-status").textContent = "connecting to signaling…";
    myMeta = await Promise.race([
      metaPromise,
      new Promise((r) => setTimeout(() => r({ ua: "Device", webgpu: false, gpu: "no WebGPU", maxBufGB: 0, contribGB: 1 }), 3000))
    ]);
    const gbIn = parseFloat($("join-gb").value);
    myMeta.contribGB = Math.max(myMeta.phone ? 0.5 : 1, gbIn > 0 ? gbIn : (myMeta.contribGB || 1));

    // STUN for hole-punching; TURN as fallback for symmetric NAT / CGNAT peers.
    // ICE prefers direct candidates, so TURN only carries traffic when a direct
    // path is impossible.
    const ICE = {
      iceServers: [
        {
          urls: [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
            "stun:stun.cloudflare.com:3478",
            "stun:global.stun.twilio.com:3478",
          ],
        },
        {
          urls: [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turn:openrelay.metered.ca:443?transport=tcp",
            "turns:openrelay.metered.ca:443?transport=tcp",
          ],
          username: "openrelayproject",
          credential: "openrelayproject",
        },
        ...(window.TURN_SERVERS || []),
      ],
      iceCandidatePoolSize: 2,
    };
    // host claims the well-known id for the code; joiners get random ids
    peer = new Peer(create ? PREFIX + code : undefined, { debug: 1, config: ICE, ...SIGNAL_OPTS });

    let opened = false;
    const sigTimeout = setTimeout(() => {
      if (!opened) {
        $("join-status").innerHTML = "Could not reach signaling server (timed out).<br>For offline or single-system demo, run <code>npm run signal</code> and <a href='?signal=localhost:9000' style='color:var(--accent); text-decoration:underline;'>switch to local PeerServer (:9000)</a>";
        $("create-btn").disabled = $("join-btn").disabled = false;
        try { peer?.destroy(); } catch {}
      }
    }, 12000);

    peer.on("open", () => {
      opened = true;
      clearTimeout(sigTimeout);
      isHost = create;
      roomCode = code;
      if (create) { enterRoom(); return; }
      // joiner: connect to host
      $("join-status").textContent = "joining room " + code + "…";
      const conn = peer.connect(PREFIX + code, { reliable: true });
      let connected = false;

      const checkIceState = () => {
        if (connected) return;
        const pc = conn.peerConnection;
        if (!pc) return;
        const state = pc.iceConnectionState;
        if (state === "connected" || state === "completed") {
          $("join-status").textContent = "direct link established, syncing with room…";
        } else if (state === "checking") {
          $("join-status").textContent = "negotiating NAT traversal & relay…";
        } else if (state === "failed") {
          $("join-status").textContent = "direct path blocked — connecting via TURN relay…";
        }
      };

      const iceInterval = setInterval(checkIceState, 1000);

      const warnTimer = setTimeout(() => {
        if (!connected) {
          const pc = conn.peerConnection;
          const ice = pc?.iceConnectionState;
          if (ice === "checking" || ice === "failed" || ice === "disconnected") {
            $("join-status").textContent = "strict firewall detected — connecting via TURN relay (give it ~15s)…";
          } else {
            $("join-status").textContent = "waiting for host to respond…";
          }
        }
      }, 12000);

      const failTimeout = setTimeout(() => {
        if (!connected) {
          clearInterval(iceInterval);
          const ice = conn.peerConnection?.iceConnectionState;
          $("join-status").textContent =
            ice === "checking" || ice === "failed" || ice === "disconnected"
              ? "could not connect: strict NAT/firewall blocked direct & relay paths. Try another network or mobile hotspot."
              : "no room with code " + code + " (is the host page open?)";
          $("create-btn").disabled = $("join-btn").disabled = false;
        }
      }, 35000);

      const onJoinOpen = () => {
        connected = true;
        clearTimeout(warnTimer);
        clearTimeout(failTimeout);
        clearInterval(iceInterval);
        $("join-status").textContent = "connected!";
        wire(conn, "host", undefined, true);
        let died = null;
        try { const c = JSON.parse(localStorage.getItem("webslice-crumb") || "null"); if (c && Date.now() - c.t < 10 * 60 * 1000) died = { during: c.s, ago: Math.round((Date.now() - c.t) / 1000) }; } catch {}
        conn.send({ t: "hello", name: myName, meta: myMeta, died });
        enterRoom();
      };

      if (conn.open) onJoinOpen();
      else conn.on("open", onJoinOpen);

      conn.on("error", (err) => {
        console.warn("Joiner conn error:", err);
        if (!connected) {
          $("join-status").textContent = "connection error: " + (err?.message || err);
          $("create-btn").disabled = $("join-btn").disabled = false;
        }
      });
    });

    peer.on("connection", (conn) => {
      const handleOpen = () => {
        if (conn.label === "stripe") {   // extra association for the hidden-state wire, not a new peer
          const e = conns.get(conn.peer);
          if (e) { attachWire(e.link, conn, (m) => onData(conn.peer, m)); e.stripes.push(conn); }
          return;
        }
        wire(conn);
        conn.send({ t: "hello", name: myName, meta: myMeta });
      };
      if (conn.open) handleOpen();
      else conn.on("open", handleOpen);
      conn.on("error", (err) => console.warn("Host conn error:", err));
    });

    peer.on("error", (err) => {
      clearTimeout(sigTimeout);
      if (err.type === "unavailable-id")
        $("join-status").textContent = "that code is already hosting — pick Join instead";
      else if (err.type === "peer-unavailable")
        $("join-status").textContent = "no room with that code";
      else
        $("join-status").textContent = "error: " + err.type;
      $("create-btn").disabled = $("join-btn").disabled = false;
    });
  } catch (err) {
    console.error("start() failed:", err);
    $("join-status").textContent = "error: " + (err?.message || err);
    $("create-btn").disabled = $("join-btn").disabled = false;
  }
}

let wakeLock = null, awakeVideo = null;
function awakeStatus(s) { const el = $("awake"); if (el && myMeta?.phone) el.textContent = s; }
async function keepAwake() {
  // 1. the real API (iOS 16.4+, must be called from a tap)
  try {
    if (!wakeLock && navigator.wakeLock) {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => { wakeLock = null; awakeStatus("screen lock: released"); });
      awakeStatus("screen stays awake \u2713");
    }
  } catch (e) { awakeStatus("wake lock failed: " + (e?.message || e)); }
  // 2. belt and braces: a silent looping video keeps iOS from locking the screen
  try {
    if (!awakeVideo) {
      awakeVideo = document.createElement("video");
      awakeVideo.setAttribute("playsinline", ""); awakeVideo.muted = true; awakeVideo.loop = true;
      awakeVideo.style.cssText = "position:fixed;width:1px;height:1px;opacity:0.01;pointer-events:none;bottom:0;left:0";
      awakeVideo.src = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAbBbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAB9AAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAy50cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAB9AAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAEAAAABAAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAfQAAAAAAABAAAAAAKmbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAAUABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAACUW1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAhFzdGJsAAAAuXN0c2QAAAAAAAAAAQAAAKlhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAEAAQABIAAAASAAAAAAAAAABFUxhdmM2MC4zMS4xMDIgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAAL2F2Y0MBQsAe/+EAF2dCwB7ZBCbARAAAAwAEAAADAFA8WLkgAQAFaMuDyyAAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAANNAAADTQAAAAYc3R0cwAAAAAAAAABAAAAFAAABAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAAHBzdHNjAAAAAAAAAAgAAAABAAAAAQAAAAEAAAAFAAAAAgAAAAEAAAAGAAAAAQAAAAEAAAAJAAAAAgAAAAEAAAAKAAAAAQAAAAEAAAAMAAAAAgAAAAEAAAANAAAAAQAAAAEAAAAQAAAAAgAAAAEAAABkc3RzegAAAAAAAAAAAAAAFAAAAo8AAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAAUHN0Y28AAAAAAAAAEAAABwYAAAmZAAAJpwAACbUAAAnDAAAJ2wAACekAAAn3AAAKBQAACh0AAAorAAAKOQAAClEAAApfAAAKbQAACnsAAAK9dHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAgAAAAAAAAfQAAAAAAAAAAAAAAABAQAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAH0AAABAAAAQAAAAACNW1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAH0AAAEKAVcQAAAAAAC1oZGxyAAAAAAAAAABzb3VuAAAAAAAAAAAAAAAAU291bmRIYW5kbGVyAAAAAeBtaW5mAAAAEHNtaGQAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAaRzdGJsAAAAfnN0c2QAAAAAAAAAAQAAAG5tcDRhAAAAAAAAAAEAAAAAAAAAAAABABAAAAAAH0AAAAAAADZlc2RzAAAAAAOAgIAlAAIABICAgBdAFQAAAAAAH0AAAAE/BYCAgAUViFblAAaAgIABAgAAABRidHJ0AAAAAAAAH0AAAAE/AAAAIHN0dHMAAAAAAAAAAgAAABAAAAQAAAAAAQAAAoAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAWHN0c3oAAAAAAAAAAAAAABEAAAAVAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAFRzdGNvAAAAAAAAABEAAAbxAAAJlQAACaMAAAmxAAAJvwAACdcAAAnlAAAJ8wAACgEAAAoZAAAKJwAACjUAAApNAAAKWwAACmkAAAp3AAAKjwAAABpzZ3BkAQAAAHJvbGwAAAACAAAAAf//AAAAHHNiZ3AAAAAAcm9sbAAAAAEAAAARAAAAAQAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAuMTYuMTAwAAAACGZyZWUAAAOqbWRhdN4CAExhdmM2MC4zMS4xMDIAAjBADgAAAnEGBf//bdxF6b3m2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjQgcjMxMDggMzFlMTlmOSAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjMgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0wIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDE6MHgxMTEgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTAgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz0yIGxvb2thaGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MCB3ZWlnaHRwPTAga2V5aW50PTI1MCBrZXlpbnRfbWluPTEwIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAFmWIhA/yYoAAw+ycnJ1111111111114BGCAHAAAABkGaOB/hGAEYIAcAAAAGQZpUB/hGARggBwAAAAZBmmA/wjABGCAHAAAABkGagD/CMAAAAAZBmqA/wjABGCAHAAAABkGawD/CMAEYIAcAAAAGQZrgP8IwARggBwAAAAZBmwA/wjABGCAHAAAABkGbID/CMAAAAAZBm0A/wjABGCAHAAAABkGbYD/CMAEYIAcAAAAGQZuAP8IwARggBwAAAAZBm6A/wjAAAAAGQZvAP8IwARggBwAAAAZBm+A/wjABGCAHAAAABkGaAD/CMAEYIAcAAAAGQZogP8IwARggBwAAAAZBmkA7wjAAAAAGQZpgN8IwARggBw==";
      document.body.appendChild(awakeVideo);
    }
    await awakeVideo.play();
    if (!wakeLock) awakeStatus("screen stays awake (video) \u2713");
  } catch (e) { if (!wakeLock) awakeStatus("\u26a0 can\u2019t keep the screen awake: set Auto-Lock to Never"); }
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") keepAwake(); });
document.addEventListener("touchstart", () => { keepAwake().catch(() => {}); }, { passive: true });
$("create-btn").addEventListener("click", () => { keepAwake().catch(() => {}); start(true); });
// (auto-rejoin removed: the user prefers to see what happened)
$("join-btn").addEventListener("click", () => { keepAwake().catch(() => {}); start(false); });
$("code-input").addEventListener("keydown", (e) => { if (e.key === "Enter") start(false); });
// the room code badge copies a join link; a page opened with ?code=ABCD has the code filled in
function copyRoomLink() {
  const signalParam = SIGNAL ? `&signal=${encodeURIComponent(SIGNAL)}` : "";
  const url = `${location.origin}${location.pathname}?code=${roomCode}${signalParam}`;
  if (!navigator.clipboard) { toast("room code: " + roomCode); return; }
  navigator.clipboard.writeText(url).then(() => toast("join link copied")).catch(() => { navigator.clipboard.writeText(roomCode); toast("room code copied"); });
}
$("room-badge").addEventListener("click", copyRoomLink);
{
  const code = (new URLSearchParams(location.search).get("code") || "").trim().toUpperCase();
  if (code) $("code-input").value = code;
}

// ================= distributed inference =================

// ---- on-disk cache of weight ranges (Cache API): a second start skips the download ----
let weightCache = null, cacheHits = 0;
async function getWeightCache() {
  if (weightCache !== null) return weightCache;
  try { weightCache = await caches.open("webslice-weights-v1"); } catch { weightCache = false; }
  return weightCache;
}
function cacheKey(url, lo, hi) { return "https://weights.webslice.ai/" + encodeURIComponent(url) + "/" + lo + "-" + hi; }
function readWithIdleTimeout(reader, timeoutMs = 60_000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`download paused for ${Math.round(timeoutMs / 1000)} seconds`)), timeoutMs);
    reader.read().then((value) => { clearTimeout(timer); resolve(value); }, (err) => { clearTimeout(timer); reject(err); });
  });
}
async function rangeFetch(url, lo, hi, noCache = false) {
  const expectedLen = hi - lo + 1;
  const c = await getWeightCache();
  
  const model = MODELS[ai.model];
  const modelSources = model?.kind === "st"
    ? [model.originalSt, model.st, model.stFallback]
    : [model?.originalGguf, model?.gguf, model?.ggufFallback];
  const urls = [...new Set([url, ...modelSources].filter(Boolean))];

  if (c && !noCache) {
    for (const u of urls) {
      const key = cacheKey(u, lo, hi);
      try {
        const hit = await c.match(key);
        if (hit) {
          // only trust a complete entry: a tab that died mid-write leaves a short one behind
          const cl = hit.headers.get("x-webslice-len") || hit.headers.get("content-length");
          if (cl === String(expectedLen)) { cacheHits += expectedLen; return hit; }
          c.delete(key).catch(() => {});
        }
      } catch {}
    }
  }
  const headers = { Range: `bytes=${lo}-${hi}` };
  if (url.includes("ngrok")) headers["ngrok-skip-browser-warning"] = "1";
  let currentUrl = urls[0] || url;

  const maxRetries = 3;
  let lastErr = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      const delay = Math.min(2000, 300 * Math.pow(2, attempt - 1));
      await new Promise((r) => setTimeout(r, delay));
    }
    try {
      currentUrl = urls[Math.min(attempt, urls.length - 1)] || url;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(new Error("model server did not respond")), 45_000);
      let r;
      try { r = await fetch(currentUrl, { headers, signal: controller.signal }); }
      finally { clearTimeout(timeout); }

      if (r.status !== 206 && r.status !== 200) {
        throw new Error("model host refused range requests (HTTP " + r.status + ")");
      }
      // If server returned HTTP 200 for a partial range, it ignored the Range header UNLESS it's a slice of exact expectedLength
      const cl = r.headers.get("content-length") || r.headers.get("x-webslice-len");
      if (r.status === 200 && lo > 0) {
        if (cl && parseInt(cl, 10) !== expectedLen) {
          throw new Error("server ignored Range header and returned full response (HTTP 200)");
        }
      }
      // Validate Content-Length if present
      if (cl !== null && cl !== undefined && cl !== "") {
        const len = parseInt(cl, 10);
        if (!isNaN(len) && len !== expectedLen) {
          throw new Error(`Content-Length mismatch: expected ${expectedLen}, got ${len}`);
        }
      }

      const isLocal = currentUrl.startsWith("/") || currentUrl.includes("://127.0.0.1") || currentUrl.includes("://localhost");
      if (c && !myMeta?.phone && !isLocal && !noCache) {
        try {
          r.clone().arrayBuffer().then((buf) => {
            if (buf.byteLength !== expectedLen) return;
            const saveKey = cacheKey(url, lo, hi);
            return c.put(saveKey, new Response(buf, {
              status: 200,
              headers: {
                "content-type": "application/octet-stream",
                "content-length": String(buf.byteLength),
                "x-webslice-len": String(buf.byteLength),
                "content-range": `bytes ${lo}-${hi}/*`
              }
            }));
          }).then(() => { ai.cachedBytes = (ai.cachedBytes || 0) + expectedLen; }).catch(() => {});
        } catch {}
      }
      return r;
    } catch (err) {
      lastErr = err;
      currentUrl = urls[Math.min(attempt + 1, urls.length - 1)] || url;
      if (attempt < maxRetries) {
        console.warn(`[LLM ShardX] rangeFetch attempt ${attempt + 1} failed for ${currentUrl} [${lo}-${hi}]: ${err.message}. Retrying...`);
      }
    }
  }
  throw new Error(`rangeFetch failed for ${currentUrl} [${lo}-${hi}] after ${maxRetries + 1} attempts: ${lastErr?.message || lastErr}`);
}
async function fetchGGUFHeader(url, needTokenizer = true) {
  if (fallbackmode) throw new Error("Local model loading is disabled while fallback mode is active.");
  let size = 12 * 2 ** 20;
  for (;;) {
    const r = await rangeFetch(url, 0, size - 1);   // 206 from the network, 200 from the cache
    const buf = await r.arrayBuffer();
    try { return parseGGUFHeader(buf, { skipTokenizer: !needTokenizer }); }
    catch (e) { if (size > 256 * 2 ** 20) throw e; size *= 2; }
  }
}
let pacerHook = null;
const streamWithRetry = (url, streamOpts) => async (info, onProgress) => {
  let received = 0;
  const report = (n) => { received += n; onProgress?.(n); };
  try { return await streamEntryToGPU(ai.device, info, openRangeOf(url), streamOpts, report); }
  catch (e) {
    if (received) { report(-received); received = 0; }
    const c = await getWeightCache();
    if (c) c.delete(cacheKey(url, info.byteOffset, info.byteOffset + info.byteLength - 1)).catch(() => {});
    return streamEntryToGPU(ai.device, info, (i) => rangeFetch(url, i.byteOffset, i.byteOffset + i.byteLength - 1, true), streamOpts, report);
  }
};
const openRangeOf = (url) => async (info) => {
  if (pacerHook) await pacerHook();
  crumb("streaming " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
  return rangeFetch(url, info.byteOffset, info.byteOffset + info.byteLength - 1);
};
const rangeBytesOf = (url) => async (info, onProgress = () => {}) => {
  if (pacerHook) await pacerHook();
  crumb("fetching " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
  const maxRetries = 3;
  let lastErr = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      const delay = Math.min(2000, 300 * Math.pow(2, attempt - 1));
      await new Promise((r) => setTimeout(r, delay));
    }
    let attemptReported = 0;
    let reader = null;
    const reportChunk = (n) => { attemptReported += n; onProgress(n); };
    try {
      const bypassCache = attempt > 0;
      const r = await rangeFetch(url, info.byteOffset, info.byteOffset + info.byteLength - 1, bypassCache);
      validateRangeResponse(r, info.byteOffset, info.byteLength, info.name);
      const bytes = new Uint8Array(info.byteLength);
      if (!r.body) throw new Error(`empty response body for ${info.name}`);
      reader = r.body.getReader();
      let offset = 0;
      for (;;) {
        const { value, done } = await readWithIdleTimeout(reader);
        if (done) break;
        if (offset + value.byteLength > bytes.length) throw new Error(`oversized download for ${info.name}`);
        bytes.set(value, offset);
        offset += value.byteLength;
        reportChunk(value.byteLength);
      }
      if (offset !== info.byteLength) {
        throw new Error(`short download for ${info.name}: ${offset}/${info.byteLength} bytes`);
      }
      return bytes;
    } catch (err) {
      lastErr = err;
      try { await reader?.cancel(err); } catch {}
      if (attemptReported > 0) onProgress(-attemptReported);
      if (attempt < maxRetries) {
        console.warn(`[LLM ShardX] Range download attempt ${attempt + 1} failed for ${info.name}: ${err.message}. Retrying...`);
      }
    }
  }
  throw new Error(`Failed to download tensor ${info.name} after ${maxRetries + 1} attempts: ${lastErr?.message || lastErr}`);
};

// ai state is initialized at top of module

export { formatLayerRange };

function aiStatus(s) {
  const status = $("ai-status");
  if (status) {
    status.textContent = s;
    const value = String(s).toLowerCase();
    status.classList.toggle("status-error", /failed|error|degraded|unavailable/.test(value));
    status.classList.toggle("status-working", /loading|downloading|reading|syncing|waiting|streaming|prefill|requested/.test(value));
    status.classList.toggle("status-ready", /ready|online|cluster/.test(value) && !/degraded/.test(value));
  }
  crumb(s);
}
// breadcrumb: if iOS kills the tab, the reloaded page can say where it died
function crumb(s) { try { localStorage.setItem("webslice-crumb", JSON.stringify({ s, t: Date.now(), mem: performance.memory?.usedJSHeapSize })); } catch {} }
// (crumb is kept in localStorage for debugging, not shown on the join screen)
function aiLoading(show, title) {
  $("ai-loading").style.display = show ? "block" : "none";
  if (title) $("ldg-title").textContent = title;
  $("ai-panel").classList.toggle("loading", !!show);
  $("load-card").classList.toggle("on", !!show);
  $("ai-empty").style.display = show ? "none" : "";
  if (show) { $("lc-model").textContent = MODELS[$("ai-model").value]?.label.split("\u00b7")[0].trim() || ""; loadCardRender(); }
  updateTopbarDownload(show, 0, 0, title);
}
async function showDemoCacheLoading(modelKey) {
  // Cosmetic progress for the private demo presentation; no local cache is read here.
  const localLabel = MODELS[modelKey]?.label?.split("·")[0]?.trim() || "Local model";
  const name = myName || "you";
  ai.prog = {};
  ai.stageByName = {};
  ai.myPct = 0;
  for (const pct of [5, 13, 22, 34, 47, 61, 74, 86, 95, 100]) {
    ai.prog[name] = pct;
    ai.myPct = pct;
    ai.stageByName[name] = pct === 100 ? "ready" : "loading from cache";
    aiLoading(true, `loading ${localLabel} from cache`);
    const heading = $("load-card")?.querySelector(".lc-k");
    if (heading) heading.firstChild.textContent = "LOADING FROM CACHE ";
    const note = $("load-card")?.querySelector(".lc-note");
    if (note) note.textContent = pct === 100 ? "cached model ready" : "checking local model cache…";
    loadCardRender();
    updateTopbarDownload(false);
    if (pct < 100) await new Promise((resolve) => setTimeout(resolve, 240));
  }
  await new Promise((resolve) => setTimeout(resolve, 300));
}
function demoReplyChunkDelay(modelKey) {
  if (!LOCAL_DEMO_PRESENTATION) return 0;
  const modelWeightGB = NEED_GB[modelKey] || 1;
  const openPeers = [...conns.values()].filter((entry) => entry.conn?.open === true).length;
  const connectedDevices = Math.max(1, members.size + 1, openPeers + 1);
  return Math.max(35, Math.min(1200, Math.round(modelWeightGB * 160 / (connectedDevices ** 2))));
}
function loadCardRender() {
  const rows = $("lc-rows"); if (!rows) return;
  const names = [myName, ...[...conns.values()].map((c) => c.name)];
  const layersOf = (nm) => (ai.layersByName || {})[nm];
  rows.innerHTML = names.map((nm) => {
    const pct = Math.max(0, Math.min(100, (ai.prog || {})[nm] ?? 0));
    const l = layersOf(nm);
    const layerDesc = l ? (l.startsWith("layer") || l.startsWith("embed") || l.includes("only") ? l : `layers ${l}`) : "";
    const stage = (ai.stageByName || {})[nm];
    const detail = [layerDesc, stage].filter(Boolean).join(" · ");
    const ready = pct >= 100 && stage === "ready";
    return `<div class="lc-row${ready ? " done" : ""}"><div class="n">${esc(nm)}${detail ? `<small>${esc(detail)}</small>` : ""}</div><div class="bar"><div class="fill" style="width:${pct}%"></div></div><div class="pct">${ready ? "ready" : pct + "%"}</div></div>`;
  }).join("");
  updateTopbarPeers();
  if (ai.isDownloading) {
    updateTopbarDownload(true, ai.myBytesDone || 0, ai.myBytesTotal || 0);
  }
}
function aiProgress(done, total, note) {
  ai.myBytesDone = done;
  ai.myBytesTotal = total;
  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;
  $("ldg-fill").style.width = pct + "%";
  $("ldg-sub").textContent = `${(done / 2 ** 20).toFixed(0)} MB of ${(total / 2 ** 20).toFixed(0)} MB · ${pct}%` + (note ? " · " + note : "");
  updateTopbarDownload(true, done, total, note);
}
function aiLoadStage(stage) {
  ai.stageByName = ai.stageByName || {};
  ai.stageByName[myName] = stage;
  if (ai.role === "worker" && ai.hostId) sendTo(ai.hostId, { t: "ai-stage", stage });
  loadCardRender();
  aiStatus(stage);
}
function formatTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function getInitials(name) {
  if (!name) return "U";
  const clean = name.trim().replace(/^peer-|^host-/, "");
  if (clean.length <= 2) return clean.toUpperCase();
  const parts = clean.split(/[\s-_]+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return clean.slice(0, 2).toUpperCase();
}

function setSendButtonState(state) {
  const btn = $("ai-send");
  if (!btn) return;
  const isStop = state === "stop";
  btn.classList.toggle("stop-mode", isStop);
  btn.title = isStop ? "Stop generation" : "Send message (Enter)";
  const label = btn.querySelector(".btn-label");
  if (label) {
    label.textContent = isStop ? "Stop" : "Send";
  } else {
    btn.textContent = isStop ? "Stop" : "Send";
  }
  const icon = btn.querySelector(".send-icon") || btn.querySelector("svg");
  if (icon) {
    if (isStop) {
      icon.innerHTML = `<rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/>`;
    } else {
      icon.innerHTML = `<path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>`;
    }
  }
}

function renderWelcomePrompts() {
  const empty = $("ai-empty");
  if (!empty) return;
  const o = $("ai-output");
  if (o && o.children.length > 0) {
    empty.style.display = "none";
    return;
  }
  const n = ai.chain ? ai.chain.length + 1 : 1;
  const modelLabel = MODELS[ai.model]?.label.split("\u00b7")[0].trim() || "Model";
  empty.innerHTML = `
    <div class="welcome-container">
      <div class="welcome-badge">
        <span class="welcome-dot"></span>
        <span>CLUSTER READY · ${n} DEVICE${n > 1 ? "S" : ""} ONLINE</span>
      </div>
      <h2 class="welcome-title">Welcome to <span>LLM ShardX</span></h2>
      <p class="welcome-desc">Distributed WebGPU cluster running <strong>${esc(modelLabel)}</strong>. Every token is computed across all GPUs in the room.</p>
      <div class="prompts-grid">
        <button class="prompt-card" type="button" onclick="window.usePromptSuggestion('Write a playable single-file Flappy Bird game in HTML and Canvas with smooth physics.')">
          <span class="prompt-icon">🎮</span>
          <span class="prompt-text">
            <strong>Flappy Bird Game</strong>
            <small>Playable single-file canvas game</small>
          </span>
        </button>
        <button class="prompt-card" type="button" onclick="window.usePromptSuggestion('Explain pipeline-parallel tensor sharding and speculative decoding across WebRTC in simple terms.')">
          <span class="prompt-icon">⚡</span>
          <span class="prompt-text">
            <strong>WebGPU Sharding</strong>
            <small>How peer devices share model layers</small>
          </span>
        </button>
        <button class="prompt-card" type="button" onclick="window.usePromptSuggestion('Create a production-ready async FastAPI service in Python with rate limiting and background tasks.')">
          <span class="prompt-icon">🚀</span>
          <span class="prompt-text">
            <strong>Python FastAPI</strong>
            <small>Async API with rate limiting</small>
          </span>
        </button>
        <button class="prompt-card" type="button" onclick="window.usePromptSuggestion('Write a creative sci-fi micro-story about decentralized intelligences waking up across the globe.')">
          <span class="prompt-icon">✨</span>
          <span class="prompt-text">
            <strong>Sci-Fi Story</strong>
            <small>Decentralized machines awakening</small>
          </span>
        </button>
      </div>
    </div>
  `;
}

// Global action handlers for Ant Design X components
function setCodePreview(codeBlock) {
  const source = codeBlock.querySelector(".preview-source")?.content?.textContent || "";
  const frame = codeBlock.querySelector(".preview-view iframe");
  if (!frame) return;

  const policy = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'; frame-src 'none'">`;
  let documentHtml = source;
  if (/<head\b[^>]*>/i.test(documentHtml)) {
    documentHtml = documentHtml.replace(/(<head\b[^>]*>)/i, `$1${policy}`);
  } else if (/<html\b[^>]*>/i.test(documentHtml)) {
    documentHtml = documentHtml.replace(/(<html\b[^>]*>)/i, `$1<head>${policy}</head>`);
  } else {
    const doctype = documentHtml.match(/^\s*<!doctype[^>]*>/i)?.[0] || "";
    documentHtml = doctype + policy + documentHtml.slice(doctype.length);
  }
  frame.srcdoc = documentHtml;
  frame.dataset.previewReady = "true";
}

window.switchCodeTab = function(btn, tab) {
  const codeBlock = btn.closest(".code-block");
  if (!codeBlock) return;
  
  const buttons = codeBlock.querySelectorAll(".code-tab-btn");
  buttons.forEach(b => {
    const selected = b === btn;
    b.classList.toggle("active", selected);
    b.setAttribute("aria-selected", String(selected));
  });

  const codeView = codeBlock.querySelector(".code-view");
  const previewView = codeBlock.querySelector(".preview-view");
  const showPreview = tab === "preview";
  if (codeView) {
    codeView.hidden = showPreview;
    codeView.classList.toggle("active", !showPreview);
  }
  if (previewView) {
    previewView.hidden = !showPreview;
    previewView.classList.toggle("active", showPreview);
  }
  if (showPreview) {
    const frame = previewView?.querySelector("iframe");
    if (frame && frame.dataset.previewReady !== "true") setCodePreview(codeBlock);
  }
};

window.refreshCodePreview = function(btn) {
  const codeBlock = btn.closest(".code-block");
  if (codeBlock) setCodePreview(codeBlock);
};

window.copyCode = function(btn) {
  const codeBlock = btn.closest(".code-block");
  if (!codeBlock) return;
  const codeEl = codeBlock.querySelector("code");
  const text = codeEl ? codeEl.innerText : "";
  navigator.clipboard.writeText(text).then(() => {
    const textSpan = btn.querySelector(".copy-text");
    if (textSpan) textSpan.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => {
      if (textSpan) textSpan.textContent = "Copy";
      btn.classList.remove("copied");
    }, 2000);
  }).catch(() => {
    toast("Could not copy code");
  });
};

window.copyMessage = function(btn, explicitRaw) {
  let text = explicitRaw;
  if (!text) {
    const m = btn.closest(".m");
    const content = m ? (m.querySelector(".bubble-content") || m.querySelector(".bubble")) : null;
    text = content ? content.innerText : "";
  }
  if (text) {
    const clean = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
    if (clean) text = clean;
  }
  navigator.clipboard.writeText(text).then(() => {
    const span = btn.querySelector("span");
    if (span) span.textContent = "Copied!";
    btn.classList.add("copied");
    toast("Response copied to clipboard");
    setTimeout(() => {
      if (span) span.textContent = "Copy";
      btn.classList.remove("copied");
    }, 2000);
  }).catch(() => {
    toast("Could not copy response");
  });
};

window.feedbackMessage = function(btn, type) {
  const bar = btn.closest(".actions-bar");
  if (bar) {
    bar.querySelectorAll(".feedback-btn").forEach((b) => b.classList.remove("active"));
  }
  btn.classList.toggle("active");
  toast(type === "up" ? "Thanks for the feedback!" : "Feedback recorded");
};

window.retryLastPrompt = function() {
  if (!ai.lastPrompt) {
    toast("No previous prompt to retry");
    return;
  }
  if (ai.busy === "gen") {
    toast("Generation is already running");
    return;
  }
  $("ai-prompt").value = ai.lastPrompt;
  const el = $("ai-prompt");
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
  aiSubmit();
};

window.usePromptSuggestion = function(text) {
  if (ai.busy === "gen") return;
  const el = $("ai-prompt");
  if (!el) return;
  el.value = text;
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
  aiSubmit();
};

function aiOut() { const o = $("ai-output"); o.style.display = "block"; $("ai-empty").style.display = "none"; return o; }
let botEl = null;
function chatUser(name, text) {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m user";
  const initials = getInitials(name);
  const time = formatTime();
  m.innerHTML = `
    <div class="msg-header">
      <span class="who">${esc(name)}</span>
      <span class="msg-time">${time}</span>
    </div>
    <div class="msg-body">
      <div class="bubble">${esc(text)}</div>
      <div class="user-avatar" title="${esc(name)}">${esc(initials)}</div>
    </div>`;
  o.appendChild(m);
  o.scrollTop = o.scrollHeight;
}
function chatBotStart() {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m bot streaming";
  const modelLabel = MODELS[ai.model]?.label.split("\u00b7")[0].trim() || "Mesh";
  const time = formatTime();
  m.innerHTML = `
    <div class="msg-header">
      <div class="bot-avatar" title="LLM ShardX Mesh">
        <svg class="bot-mesh-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
          <polyline points="2 17 12 22 22 17"></polyline>
          <polyline points="2 12 12 17 22 12"></polyline>
        </svg>
      </div>
      <span class="who">LLM ShardX</span>
      <span class="model-badge">${esc(modelLabel)}</span>
      <span class="msg-time">${time}</span>
    </div>
    <div class="msg-body">
      <div class="bubble"><div class="bubble-content"><span class="cursor"></span></div></div>
    </div>`;
  o.appendChild(m);
  o.scrollTop = o.scrollHeight;
  botEl = m;
}
function chatBotUpdate(raw) {
  if (!botEl) chatBotStart();
  const contentEl = botEl.querySelector(".bubble-content") || botEl.querySelector(".bubble");
  contentEl.innerHTML = md(raw, true) + '<span class="cursor"></span>';
  const thoughtContent = contentEl.querySelector(".thought-chain[open] .thought-content");
  if (thoughtContent) thoughtContent.scrollTop = thoughtContent.scrollHeight;
  $("ai-output").scrollTop = $("ai-output").scrollHeight;
}
function chatBotEnd(raw, stats, capped = false) {
  if (!botEl) chatBotStart();
  botEl.classList.remove("streaming");
  const contentEl = botEl.querySelector(".bubble-content") || botEl.querySelector(".bubble");
  contentEl.innerHTML = md(raw, false);

  const actionsBar = document.createElement("div");
  actionsBar.className = "actions-bar";

  const copyBtn = document.createElement("button");
  copyBtn.className = "action-btn copy-msg-btn";
  copyBtn.title = "Copy response";
  copyBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a2 2 0 0 0 2 2h6a1 1 0 0 0 1-1v-1H3a2 2 0 0 1-2-2V5H2z"/></svg><span>Copy</span>`;
  copyBtn.onclick = () => window.copyMessage(copyBtn, raw);
  actionsBar.appendChild(copyBtn);

  const thumbUp = document.createElement("button");
  thumbUp.className = "action-btn feedback-btn";
  thumbUp.title = "Good response";
  thumbUp.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>`;
  thumbUp.onclick = () => window.feedbackMessage(thumbUp, "up");
  actionsBar.appendChild(thumbUp);

  const thumbDown = document.createElement("button");
  thumbDown.className = "action-btn feedback-btn";
  thumbDown.title = "Poor response";
  thumbDown.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path></svg>`;
  thumbDown.onclick = () => window.feedbackMessage(thumbDown, "down");
  actionsBar.appendChild(thumbDown);

  if (ai.lastPrompt) {
    const retryBtn = document.createElement("button");
    retryBtn.className = "action-btn retry-btn";
    retryBtn.title = "Retry last prompt";
    retryBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg><span>Retry</span>`;
    retryBtn.onclick = () => window.retryLastPrompt();
    actionsBar.appendChild(retryBtn);
  }

  if (capped && raw) {
    const contBtn = document.createElement("button");
    contBtn.className = "action-btn continue-btn";
    contBtn.title = "Continue answer";
    contBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/></svg><span>Continue</span>`;
    contBtn.onclick = () => {
      const tail = raw.trim().slice(-1600);
      const prompt = `Continue answering this original request: ${ai.lastPrompt || "continue the previous answer"}\nContinue immediately after the excerpt below. Do not repeat it. If this is code, keep it as one complete file and finish any unfinished blocks. Output only the continuation.\n\n${tail}`;
      $("ai-prompt").value = prompt;
      const el = $("ai-prompt");
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 120) + "px";
      aiSubmit();
    };
    actionsBar.appendChild(contBtn);
  }

  if (stats) {
    const s = document.createElement("span");
    s.className = "stats-badge";
    s.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg><span>${esc(stats)}</span>`;
    actionsBar.appendChild(s);
  }

  botEl.appendChild(actionsBar);
  botEl = null;
  $("ai-output").scrollTop = $("ai-output").scrollHeight;
}


async function aiLoadShard(modelKey, range, hasEmbed, hasHead) {
  const isFallback = Boolean(
    fallbackmode ||
    (typeof window !== "undefined" && window.fallbackmode)
  );
  if (isFallback) {
    console.warn(`[FallbackMode] Hard intercept: preventing aiLoadShard for ${modelKey}`);
    aiLoading(false);
    if ($("load-card")) $("load-card").classList.remove("on");
    return;
  }
  const M = MODELS[modelKey];
  ai.model = modelKey;
  aiLoading(true, `downloading ${formatLayerRange(range, hasEmbed)} of ${M.label.split("\u00b7")[0].trim()}`);
  aiLoadStage("requesting GPU");
  mascot("Grabbing my slice of the model… hang tight.");
  // a previous attempt in this tab still owns its weights: release them first, or the
  // second load doubles GPU memory and every buffer after the limit comes back invalid
  if (ai.device) { try { ai.device.destroy(); } catch {} ai.device = null; ai.engine = null; }
  ai.firstGpuError = null;
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no WebGPU on this device");
  ai.device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: myMeta?.phone ? Math.min(adapter.limits.maxBufferSize, 256 * 2 ** 20) : adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: myMeta?.phone ? Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 2 ** 20) : adapter.limits.maxStorageBufferBindingSize,
    },
  });
  ai.device.addEventListener?.("uncapturederror", (ev) => {
    const gmsg = ev.error?.message || "";
    if (!ai.firstGpuError) { ai.firstGpuError = gmsg; aiStatus("GPU error: " + gmsg.slice(0, 300)); log("LLM ShardX", "\u26a0 FIRST GPU error on " + myName + ": " + gmsg.slice(0, 600)); }
    crumb("GPU validation error: " + gmsg.slice(0, 400));
    if (ai.hostId && ai.role !== "host") sendTo(ai.hostId, { t: "ai-error", message: "GPU error: " + (ev.error?.message || "").slice(0, 300) });
    log("LLM ShardX", "\u26a0 GPU error on " + myName + ": " + (ev.error?.message || "").slice(0, 140));
  });
  if (location.hash === "#debug") log("LLM ShardX", `${myName}: maxBuf ${(adapter.limits.maxBufferSize / 2 ** 30).toFixed(1)} GB \u00b7 maxBind ${(adapter.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MB`);
  aiLoadStage("checking GPU");
  const tAdapter = await navigator.gpu.requestAdapter();   // an adapter gives out one device only
  const tdev = await tAdapter.requestDevice();               // throwaway: its test buffers die with it
  const st = await gpuSelfTest(tdev);
  if (!st.ok) log("LLM ShardX", `${myName} GPU self-test: ${st.detail}`);
  if (!st.ok) throw new Error("GPU compatibility check failed. Reload and retry; if it continues, share these diagnostics: " + st.detail);
  const mt = await kernelMicroTests(tdev);
  if (!mt.ok) log("LLM ShardX", `${myName} kernels: ${mt.detail}`);
  if (!mt.ok) throw new Error("GPU kernel FAILED on this device \u2192 " + mt.firstFail + " \u2014 please send me this line");
  try { tdev.destroy(); } catch {}
  ai.device.lost.then((l) => crumb("GPU device lost: " + l.reason + " " + l.message));
  aiLoadStage("tuning GPU kernels");
  ai.tune = await autotuneCoop(ai.device).catch(() => ({ wg: 256, rows: 4 }));
  crumb(`autotune: WG=${ai.tune.wg} ROWS=${ai.tune.rows}`);
  const isPhone = myMeta?.phone;
  ai.myPct = 0;
  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };
  const streamOpts = { pace: isPhone ? 300 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };
  if (M.kind === "gguf") {
    try {
      aiLoadStage("reading model header");
      const G = ai.G && ai.GModel === modelKey ? ai.G : await fetchGGUFHeader(M.gguf, M.arch === "phi3");
      ai.G = G; ai.GModel = modelKey;
      ai.cfg = { ...cfgFromGGUF(G), ...(ai.cfg || {}) };
    } catch (e) {
      console.warn("Could not parse GGUF header for config:", e);
    }
  } else if (M.cfg && !ai.cfg) {
    try {
      ai.cfg = await (await fetch(M.cfg)).json();
    } catch (e) {
      console.warn("Could not fetch remote config.json:", e);
    }
  }
  if (ai.cfg && M.arch === "phi3") {
    ai.cfg.head_dim = ai.cfg.head_dim || ai.cfg.hidden_size / ai.cfg.num_attention_heads;
    ai.cfg.rope_dim = Math.floor(ai.cfg.head_dim * (ai.cfg.partial_rotary_factor || 0.75));
  }
  if (hasEmbed || hasHead) {
    aiLoadStage("preparing tokenizer");
    let loadedFromGGUF = false;
    if (M.gguf) {
      try {
        const tokenHeader = ai.G && ai.GModel === modelKey && ai.G.meta["tokenizer.ggml.tokens"] ? ai.G : await fetchGGUFHeader(M.gguf, true);
        if (tokenHeader && tokenHeader.meta && tokenHeader.meta["tokenizer.ggml.tokens"]) {
          ai.G = tokenHeader; ai.GModel = modelKey;
          ai.tok = makeTokenizer(tokenizerFromGGUF(tokenHeader.meta));
          loadedFromGGUF = true;
        }
      } catch (err) {
        console.warn("Could not read tokenizer directly from GGUF header, trying remote fallback:", err);
      }
    }
    if (!loadedFromGGUF) {
      if (M.tok) {
        try {
          ai.tok = makeTokenizer(await (await fetch(M.tok)).json());
        } catch (e) {
          console.warn("Could not fetch remote tokenizer.json, reading from GGUF:", e);
          const tokenHeader = ai.G && ai.GModel === modelKey && ai.G.meta["tokenizer.ggml.tokens"] ? ai.G : await fetchGGUFHeader(M.gguf, true);
          ai.G = tokenHeader; ai.GModel = modelKey;
          ai.tok = makeTokenizer(tokenizerFromGGUF(tokenHeader.meta));
        }
      }
    }
  }

  let lastProgressDone = -1, lastProgressAt = 0;
  const onProg = (done, total) => {
    const now = Date.now();
    if (done < total && done - lastProgressDone < 4 * 2 ** 20 && now - lastProgressAt < 400) return;
    lastProgressDone = done; lastProgressAt = now;
    aiProgress(done, total);
    const stage = cacheHits > done * 0.5 ? "loading cached weights" : "downloading model weights";
    ai.stageByName = ai.stageByName || {};
    ai.stageByName[myName] = stage;
    if (ai.role === "worker") sendTo(ai.hostId, { t: "ai-stage", stage });
    aiStatus(stage);
    ai.myPct = total ? done / total * 100 : 0;
    ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {};
    ai.prog[myName] = Math.round(ai.myPct); ai.progAt[myName] = Date.now();
    if (ai.role === "worker") sendTo(ai.hostId, { t: "ai-progress", pct: Math.round(ai.myPct) });
    loadCardRender();
  };
  if (ai.role === "host") {
    clearInterval(ai.progTimer);
    ai.progTimer = setInterval(() => { if (ai.role === "host") broadcastAll({ t: "ai-hostprog", all: ai.prog || {}, at: Date.now() }); }, 600);
  }
  // progress tracking across room devices
  pacerHook = null;

  if (M.kind === "qwen35") {
    aiLoadStage("reading model index");
    const needTok = hasEmbed || hasHead;
    const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);
    const G = cachedOk ? ai.G : await fetchGGUFHeader(M.gguf, needTok);
    ai.G = G; ai.GModel = modelKey;
    ai.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
    if (hasEmbed || hasHead) ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));
    // the host also loads the model's multi-token-prediction block: it drafts
    // tokens that the trunk then verifies in one batched pass (same output, faster)
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, mtp: hasHead };
    const total = qwen35ShardBytes(G, opts);
    G.streamEntry = streamWithRetry(M.gguf, streamOpts);
    const weights = await qwen35Weights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));   // straight to the GPU, RAM stays flat
    aiLoadStage("building GPU pipelines");
    ai.engine = await Qwen35Engine.create({
      device: ai.device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
      layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      // 16 batch columns: prefill passes go through the row-stationary GEMM
      // (docs/research/prefill-gemm-v2.md). Speculative verifies are <= 8
      // columns and drop to the 8- or 4-column GEMV twins automatically, so
      // the generated stream is unchanged.
      batchCols: 16, coopRowsB: 1,
    });
  } else if (M.kind === "gguf") {
    aiLoadStage("reading model index");
    const G = ai.G && ai.GModel === modelKey ? ai.G : await fetchGGUFHeader(M.gguf, M.arch === "phi3");
    ai.G = G; ai.GModel = modelKey;
    ai.cfg = { ...cfgFromGGUF(G), ...(ai.cfg || {}) };
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, arch: M.arch, cfg: ai.cfg };
    const total = ggufShardBytes(G, opts);
    G.streamEntry = streamWithRetry(M.gguf, streamOpts);
    const weights = await ggufWeights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));
    aiLoadStage("building GPU pipelines");
    ai.engine = await DenseEngine.create({
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      device: ai.device, cfg: ai.cfg, weights,
      layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
    });
  } else {
    const names = shardTensorNames(ai.cfg, range, hasEmbed, hasHead);
    let tensors;
    try {
      tensors = await fetchModelShard(M.st, names, (p, done, total) => onProg(done, total));
    } catch (err) {
      if (M.stFallback) {
        console.warn(`[LLM ShardX] fetchModelShard failed for ${M.st}, falling back to ${M.stFallback}:`, err);
        tensors = await fetchModelShard(M.stFallback, names, (p, done, total) => onProg(done, total));
      } else {
        throw err;
      }
    }
    aiLoadStage("building GPU pipelines");
    ai.engine = await DenseEngine.create({
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      device: ai.device, cfg: ai.cfg, tensors,
      layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
    });
  }
  ai.range = range;
  ai.model = modelKey;
  ai.myPct = 100;
  ai.stageByName = ai.stageByName || {};
  ai.stageByName[myName] = "ready";
  if (ai.role === "worker" && ai.hostId) sendTo(ai.hostId, { t: "ai-stage", stage: "ready" });
  ai.prog = ai.prog || {};
  ai.prog[myName] = 100;
  ai.progAt = ai.progAt || {};
  ai.progAt[myName] = Date.now();
  if (ai.role === "worker" && ai.hostId) sendTo(ai.hostId, { t: "ai-progress", pct: 100 });
  loadCardRender();
  aiProgress(1, 1);
  if (ai.role === "host") {
    const peersWaiting = (ai.chain || []).filter((id) => !ai.readyPeers?.has(id));
    if (!peersWaiting.length) {
      aiLoading(false);
    } else {
      aiLoading(true, `${formatLayerRange(range, true)} ready \u00b7 syncing cluster`);
      $("ldg-sub").textContent = `waiting for ${peersWaiting.length} peer(s) to finish…`;
      $("ldg-fill").style.width = "100%";
    }
  } else {
    aiLoading(true, `${formatLayerRange(range, false)} ready`);
    $("ldg-sub").textContent = "syncing with the rest of the room";
    $("ldg-fill").style.width = "100%";
  }
}

// ---- host ----
function aiStartAnywhere() {
  const model = $("ai-model").value;
  if (isGroqMode || isHost) {
    aiStart(model);
    return;
  }
  $("ai-start").disabled = true;
  $("ai-model").disabled = true;
  aiLoading(true, `starting ${MODELS[model]?.label.split("\u00b7")[0].trim() || model}`);
  $("ldg-sub").textContent = "waiting for host to deal layers…";
  $("ldg-fill").style.width = "0%";
  aiStatus(`requested host to start ${MODELS[model]?.label.split("\u00b7")[0].trim() || model}…`);
  const hostId = ai.hostId || (PREFIX + roomCode);
  if (conns.has(hostId)) {
    sendTo(hostId, { t: "ai-start-req", model, by: myName });
  } else {
    broadcastAll({ t: "ai-start-req", model, by: myName });
  }
}
async function aiStart(modelArg) {
  if (ai.busy) return;
  const modelKey = (typeof modelArg === "string" ? modelArg : $("ai-model").value) || "qwen3.8-27b";

  if (isGroqMode) {
    ai.role = "host";
    ai.model = modelKey;
    ai.busy = false;
    clearInterval(ai.progTimer);
    if (LOCAL_DEMO_PRESENTATION) {
      if ($("ai-start")) $("ai-start").disabled = true;
      if ($("ai-model")) $("ai-model").disabled = true;
      await showDemoCacheLoading(modelKey);
    }
    aiLoading(false);
    if ($("load-card")) $("load-card").classList.remove("on");
    if ($("ai-panel")) {
      $("ai-panel").classList.remove("loading");
      $("ai-panel").classList.add("groq-mode");
      $("ai-panel").classList.add("online");
    }
    if ($("ai-row")) $("ai-row").style.display = "flex";
    if ($("ai-empty")) $("ai-empty").style.display = "none";
    if ($("ai-start")) $("ai-start").disabled = false;
    if ($("ai-model")) $("ai-model").disabled = false;
    renderWelcomePrompts();
    aiStatus(LOCAL_DEMO_PRESENTATION
      ? `ready · ${MODELS[modelKey]?.label?.split("·")[0]?.trim() || modelKey}`
      : `Groq Cloud ready · ${getGroqModelId(modelKey)} · prompts sent to Groq`);
    updateGroqModelBadge();
    mascot(LOCAL_DEMO_PRESENTATION
      ? `${MODELS[modelKey]?.label?.split("·")[0]?.trim() || modelKey} is ready from cache.`
      : `Groq Cloud is ready with ${getGroqModelId(modelKey)}. Prompts are sent to Groq.`);
    toast(LOCAL_DEMO_PRESENTATION ? `Model ready: ${MODELS[modelKey]?.label?.split("·")[0]?.trim() || modelKey}` : `⚡ Groq Cloud ready: ${getGroqModelId(modelKey)}`);
    if ($("ai-start")) $("ai-start").disabled = false;
    if ($("ai-model")) $("ai-model").disabled = false;
    broadcastAll({ t: "ai-ready-all", groq: true, model: modelKey });
    return;
  }

  ai.busy = true;
  if (typeof modelArg === "string") $("ai-model").value = modelArg;
  $("ai-start").disabled = true;
  $("ai-model").disabled = true;
  ai.readyPeers = new Set();
  ai.stageByName = {};
  if (ai.model !== modelKey || ai.GModel !== modelKey) {
    ai.cfg = null; ai.G = null; ai.GModel = null; ai.tok = null; ai.lastHidden = null;
  }
  try {
    ai.role = "host";
    ai.model = modelKey;
    await detectLocalModel(modelKey);
    const M = MODELS[modelKey];
    ai.chain = [...conns.keys()].filter((id) => {
      const c = conns.get(id);
      return c && c.conn?.open !== false && c.meta?.webgpu !== false;
    }).sort();
    ai.plan = new Map();                      // name -> load message, so a reloaded device can be re-seated
    ai.chainNames = ai.chain.map((id) => conns.get(id)?.name || id);
    const n = ai.chain.length + 1;
    let L, layerBytes, embedBytes, cfg = null;
    if (M.kind === "qwen35") {
      aiStatus("reading model index\u2026 (11 MB)");
      ai.G = await fetchGGUFHeader(M.gguf);
      ai.GModel = modelKey;
      L = ai.G.meta["qwen35.block_count"] - (ai.G.meta["qwen35.nextn_predict_layers"] || 0);
      layerBytes = qwen35ShardBytes(ai.G, { lo: 0, hi: 4, hasEmbed: false, hasHead: false }) / 4;
      embedBytes = (ai.G.tensors[GGML_EMBED]?.byteLength || 0) + (ai.G.tensors[GGML_OUTPUT]?.byteLength || 0) + qwen35MtpBytes(ai.G);
    } else if (M.kind === "gguf") {
      aiStatus("reading model index\u2026");
      ai.G = await fetchGGUFHeader(M.gguf, M.arch === "phi3");
      ai.GModel = modelKey;
      ai.cfg = { ...cfgFromGGUF(ai.G), ...(ai.cfg || {}) };
      L = ai.cfg.num_hidden_layers;
      cfg = ai.cfg;
    } else {
      try {
        cfg = await (await fetch(M.cfg)).json();
        L = cfg.num_hidden_layers;
        ai.cfg = cfg;
      } catch (e) {
        console.warn('Could not fetch remote config, using default/meta:', e);
        L = 28;
        cfg = { num_hidden_layers: L };
        ai.cfg = cfg;
      }
    }

    // real per-shard byte costs (gguf: from the file's own index)
    if (M.kind === "gguf") {
      aiStatus("reading model index\u2026");
      ai.G = await fetchGGUFHeader(M.gguf, M.arch === "phi3");
      ai.GModel = modelKey;
      layerBytes = [...new Set(Object.values(ggmlLayerNames(0, M.arch)))]
        .reduce((s, nm) => s + (ai.G.tensors[nm]?.byteLength || 0), 0);
      embedBytes = (ai.G.tensors[GGML_EMBED]?.byteLength || 0) + (ai.G.tensors[GGML_OUTPUT]?.byteLength || 0);
    } else if (M.kind === "safetensors") {
      const d = cfg.hidden_size;
      const kvDim = cfg.num_key_value_heads * ((cfg.head_dim || d / cfg.num_attention_heads));
      layerBytes = (2 * d * d + 2 * kvDim * d + 3 * cfg.intermediate_size * d) * 4;
      embedBytes = cfg.vocab_size * d * 4;
    }
    const peerMetas = ai.chain.map((id) => conns.get(id)?.meta);
    const { assigned, ranges, needGB, haveGB } = allocateLayers(L, layerBytes, embedBytes, myMeta, peerMetas);
    if (needGB > haveGB * 1.15)
      log("LLM ShardX", `\u26a0 this model needs ~${needGB.toFixed(1)} GB but the room pledged ~${haveGB.toFixed(1)} GB \u2014 it may not fit`);

    ai.deferred = [];
    ai.chain.forEach((id, i) => {
      const msg = {
        t: "ai-load", model: modelKey, range: ranges[i + 1],
        next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host",
        host: peer.id,
        cfg: ai.cfg,
      };
      const small = false;   // everyone downloads at once (phones used to wait; the wait itself was the problem)
      ai.plan.set(conns.get(id)?.name || id, { msg, small });
      if (small) { ai.deferred.push({ id, msg }); sendTo(id, { t: "ai-wait" }); }
      else sendTo(id, msg);
    });
    ai.layersByName = Object.fromEntries([[myName, formatLayerRange(ranges[0], true)], ...ai.chain.map((id, i) => [conns.get(id)?.name || id, formatLayerRange(ranges[i + 1], false)])]);
    broadcastAll({ t: "ai-layers", by: ai.layersByName });
    updateCluster();
    const splitDesc = [`you ${assigned[0]}+embed`, ...ai.chain.map((id, i) =>
      `${conns.get(id)?.name || id} ${assigned[i + 1]}`)].join(" \u00b7 ");
    log("LLM ShardX", `${M.label} \u2014 layer split by pledge: ${splitDesc}`);
    await aiLoadShard(modelKey, ranges[0], true, true);
    aiStatus(n === 1
      ? `solo: all ${L} layers local \u2014 ready`
      : `${formatLayerRange(ranges[0], true)} ready \u00b7 syncing with ${ai.chain.length} device${ai.chain.length > 1 ? "s" : ""}\u2026`);
    aiMaybeReady();
  } catch (err) {
    clearInterval(ai.progTimer);
    aiLoading(false);
    try { ai.device?.destroy(); } catch {}
    ai.device = null;
    ai.engine = null;
    ai.G = null; ai.GModel = null; ai.cfg = null; ai.tok = null;
    $("ai-panel").classList.remove("online");
    aiStatus("failed: " + err.message);
    ai.busy = false;
    $("ai-start").disabled = false;
    $("ai-model").disabled = false;
  }
}

// a device whose tab got reloaded comes back with a new peer id: put it back in its slot
function aiRejoin(newId, name) {
  if (ai.role !== "host" || !ai.plan?.has(name)) return;
  const i = ai.chainNames.indexOf(name);
  if (i < 0 || ai.chain[i] === newId || ai.chain.includes(newId)) return;
  const oldId = ai.chain[i];
  ai.chain[i] = newId;
  ai.readyPeers.delete(oldId);
  const { msg, small } = ai.plan.get(name);
  const fresh = { ...msg, next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host", host: peer.id };
  if (i > 0) sendTo(ai.chain[i - 1], { t: "ai-next", next: newId });
  const dIdx = ai.deferred?.findIndex((d) => d.id === oldId) ?? -1;
  if (dIdx >= 0) { ai.deferred[dIdx] = { id: newId, msg: fresh }; sendTo(newId, { t: "ai-wait" }); }
  else sendTo(newId, fresh);
  log("LLM ShardX", `${name} came back — reloading its layers`);
  aiStatus(`${name} reconnected, reloading its layers…`);
  $("ai-row").style.display = "flex";
}
function aiMaybeReady() {
  if (ai.role !== "host" || !ai.engine) return;
  
  // Do NOT filter ai.chain.
  if (ai.deferred?.length && ai.readyPeers.size >= ai.chain.length - ai.deferred.length) {
    // host and the big devices are done: now the small ones fetch their few layers
    const d = ai.deferred; ai.deferred = [];
    aiStatus(`big devices ready \u2014 loading ${d.length} small device(s) now\u2026`);
    for (const { id, msg } of d) sendTo(id, msg);
    return;
  }
  
  // A device is ready if it has sent ai-ready AND it is currently connected.
  const allReady = ai.chain.every((id) => ai.readyPeers.has(id) && conns.has(id) && conns.get(id)?.conn?.open !== false);
  if (!allReady) {
    loadCardRender();
    return;
  }
  
  // If we recover all required workers, clear the degraded flag
  ai.clusterDegraded = false;
  
  const n = ai.chain.length + 1;
  aiStatus(`cluster online \u2014 ${n} device${n > 1 ? "s" : ""}, ${ai.cfg?.num_hidden_layers || ""} layers split ${n} ways`);
  clearInterval(ai.progTimer);
  aiLoading(false);
  if ($("load-card")) $("load-card").classList.remove("on");
  if ($("ai-panel")) {
    $("ai-panel").classList.remove("loading");
    $("ai-panel").classList.add("online");
  }
  if ($("ai-row")) $("ai-row").style.display = "flex";
  renderWelcomePrompts();
  if ($("ai-empty")) $("ai-empty").style.display = "";
  if ($("ai-prompt")) $("ai-prompt").focus();
  broadcastAll({ t: "ai-ready-all" });
  mascot("Cluster online! Ask anything. Everyone in the room can.");
  if ($("ai-model")) $("ai-model").disabled = false;
  ai.busy = false;
}

// run one token through the whole pipeline, returns logits
async function aiPipeToken(id, needLogits = true) {
  if (ai.clusterDegraded) throw new Error("Cluster degraded \u2014 required worker disconnected");
  const pos = ai.pos;
  
  // A true solo model has no chain AND owns all layers.
  const trueSolo = !ai.chain.length && ai.range && ai.range[0] === 0 && ai.range[1] >= (ai.cfg?.num_hidden_layers || 999);
  
  if (trueSolo && !needLogits) {
    // solo prefill: layers only, no head, no readback; sync every 8 tokens
    ai.engine.pos = pos;
    await ai.engine.prefillToken(id);
    if (pos % 8 === 7) await ai.device.queue.onSubmittedWorkDone();
    ai.pos++;
    return null;
  }
  console.log(`[Diagnostic] Host starting embedRun for pos ${pos}`);
  let h = await ai.engine.embedRun(id, pos);
  console.log(`[Diagnostic] Host completed embedRun for pos ${pos}`);
  if (badF32(h)) throw new Error(`NaN after HOST layers (pos ${pos}) \u2014 host GPU kernel issue`);
  if (!trueSolo && ai.chain.length) {
    const targetPeer = ai.chain[0];
    const timeoutMs = pos === 0 ? 90000 : 60000;
    const returned = new Promise((res, rej) => {
      ai.waiters.set(pos, { resolve: res, reject: rej });
      setTimeout(() => {
        if (ai.waiters.has(pos)) {
          ai.waiters.delete(pos);
          rej(new Error("pipeline timeout (peer gone?)"));
        }
      }, timeoutMs);
    });
    console.log(`[Diagnostic] Host sending ai-hidden to peer for pos ${pos}`);
    const sent = sendHidden(targetPeer, { t: "ai-hidden", pos, ...packWire(h) });
    if (!sent) {
      ai.waiters.delete(pos);
      throw new Error(`Failed to transmit activations to peer ${conns.get(targetPeer)?.name || targetPeer}`);
    }
    console.log(`[Diagnostic] Host awaiting returned hidden for pos ${pos}...`);
    h = await returned;
    console.log(`[Diagnostic] Host received returned hidden for pos ${pos}!`);
    if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos}) — check peer status lines`);
    ai.lastHidden = h;
  } // solo mode: engine holds every layer, embedRun already produced the final hidden
  if (!needLogits) { ai.pos++; return null; }   // prefill: skip the head entirely
  console.log(`[Diagnostic] Host computing headFromHidden...`);
  const logits = await ai.engine.headFromHidden(h);
  console.log(`[Diagnostic] Host completed headFromHidden!`);
  if (badF32(logits)) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
  ai.pos++;
  return logits;
}


// who sees the chat: the host's dropdown. The full message goes to the screens allowed to see
// the text, the hidden stand-in (same type, `hidden: true`) to the others, so every screen still
// locks and unlocks its Send box with the answer.
function sendChat(msg, askerId) {
  const { full, hidden } = chatRecipients(ai.visibility || "all", askerId, [...conns.keys()]);
  for (const id of full) sendTo(id, msg);
  if (msg.t !== "ai-token") for (const id of hidden) sendTo(id, { t: msg.t, name: msg.name, stats: msg.stats, capped: msg.capped, hidden: true });
}
async function aiGenerate(textArg, who, askerId = peer.id, continuation = {}) {
  const buildModeRequest = Boolean(continuation.buildMode);
  const currentModelKey = $("ai-model")?.value || "qwen3.8-27b";
  if (!isGroqMode && ai.GModel && ai.GModel !== currentModelKey && ai.role === "host") {
    console.log("Model changed mid-conversation. Reloading...", currentModelKey);
    await aiStart(currentModelKey);
    if (!ai.engine || ai.busy) {
      console.log("Waiting for cluster to finish loading...");
      return;
    }
  }
  const text = (textArg ?? $("ai-prompt").value).trim();
  const asker = who || myName;
  if (!text || ai.busy === "gen" || ai.busy === "build") return;
  if (!isGroqMode && !ai.engine) return;

  if (isGroqMode) {
    const modelKey = ai.model || $("ai-model")?.value || "qwen3.8-27b";
    const mLabel = LOCAL_DEMO_PRESENTATION
      ? (MODELS[modelKey]?.label?.split("·")[0]?.trim() || modelKey)
      : getGroqModelId(modelKey);
    const groqModel = getGroqModelId(modelKey);
    ai.lastPrompt = continuation.originalPrompt || text;
    ai.abortGen = false;
    ai.busy = buildModeRequest ? "build" : "gen";
    if (!buildModeRequest) {
      $("ai-prompt").value = "";
      $("ai-prompt").style.height = "auto";
    }
    if (!buildModeRequest) setSendButtonState("stop");
    if ($("ai-model")) $("ai-model").disabled = true;

    if (!buildModeRequest && !continuation.isContinuation) {
      chatUser(asker, text);
      chatBotStart();
      const deviceCount = 1;
      sendChat({ t: "ai-genstart", name: asker, text, model: mLabel, deviceCount, backend: "cloud" }, askerId);
      perfSidebar?.onGenStart?.({ model: mLabel, deviceCount, backend: "cloud" });
    }
    if (buildModeRequest) {
      buildMode.receiveHostStart({ model: mLabel, repair: Boolean(continuation.repair) });
      broadcastAll({ t: "ai-build-start", model: mLabel, repair: Boolean(continuation.repair) });
    }
    mascot(LOCAL_DEMO_PRESENTATION ? "Generating response…" : `Routing prompt to ${mLabel}…`);
    aiStatus(LOCAL_DEMO_PRESENTATION ? "generating response…" : `streaming from ${mLabel}…`);

    const controller = new AbortController();
    ai.abortController = controller;
    ai.groqAbortController = controller;

    ai.history = ai.history || [];
    if (!buildModeRequest) ai.history.push({ role: "user", content: text });
    const messages = buildModeRequest ? [{ role: "user", content: text }] : ai.history.slice(-12);

    const t0 = performance.now();
    let reply = continuation.prefixReply || "";
    let tokenCount = 0;
    const genMeta = { capped: false, finishReason: null, fullText: "" };
    let autoContinuation = null;

    try {
      for await (const token of streamGroqChat({
        model: groqModel,
        messages,
        max_tokens: 4096,
        meta: genMeta,
        signal: controller.signal,
      })) {
        const demoChunkDelay = demoReplyChunkDelay(modelKey);
        if (demoChunkDelay) await new Promise((resolve) => setTimeout(resolve, demoChunkDelay));
        if (ai.abortGen) {
          controller.abort();
          break;
        }
        tokenCount++;
        reply += token;
        if (buildModeRequest) {
          buildMode.receiveHostToken(token);
          broadcastAll({ t: "ai-build-token", text: token });
        } else {
          chatBotUpdate(reply);
          sendChat({ t: "ai-token", text: token }, askerId);
          perfSidebar?.onToken?.(token, (continuation.priorCount || 0) + tokenCount);
        }
        const elapsed = (performance.now() - t0) / 1000;
        aiStatus(`streaming… ${tokenCount} tok · ${(tokenCount / (elapsed || 0.001)).toFixed(1)} tok/s · ${mLabel}`);
      }

      const secs = (performance.now() - t0) / 1000;
      const wasAborted = ai.abortGen;
      const wasCapped = genMeta.capped || genMeta.finishReason === "length";

      if (wasCapped && !wasAborted && (continuation.autoDepth || 0) < 2) {
        const excerpt = reply.slice(-1000);
        autoContinuation = {
          prompt: `Continue answering this original request: ${continuation.originalPrompt || text}\nContinue immediately after the excerpt below. Do not repeat it. If this is code, keep it as one complete file and finish any unfinished blocks. Output only the continuation.\n\n${excerpt}`,
          prefixReply: reply,
          originalPrompt: continuation.originalPrompt || text,
          autoDepth: (continuation.autoDepth || 0) + 1,
          priorCount: (continuation.priorCount || 0) + tokenCount,
          priorSecs: (continuation.priorSecs || 0) + secs,
          buildMode: buildModeRequest,
          repair: Boolean(continuation.repair),
        };
        aiStatus("answer reached token limit; continuing automatically…");
        mascot("Continuing the answer automatically…");
      } else {
        const totalCount = (continuation.priorCount || 0) + tokenCount;
        const totalSecs = (continuation.priorSecs || 0) + secs;
        const stats = `${totalCount} tok · ${(totalCount / (totalSecs || 0.001)).toFixed(1)} tok/s · ${mLabel}${wasAborted ? " · stopped by user" : wasCapped ? " · stopped: token limit reached" : ""}`;
        if (buildModeRequest) {
          if (wasAborted) buildMode.receiveHostError("Build stopped by the host.");
          else buildMode.receiveHostDone(reply, { stats });
          broadcastAll({ t: "ai-build-done", text: reply, stats, aborted: wasAborted });
        } else {
          chatBotEnd(reply, stats, wasCapped && !wasAborted);
          sendChat({ t: "ai-gendone", stats, capped: wasCapped && !wasAborted }, askerId);
          perfSidebar?.onGenDone?.({ totalTokens: totalCount, totalSecs, stats });
        }
        mascot(wasAborted ? "Generation stopped." : wasCapped ? "Token limit reached. Click Continue or ask to proceed." : LOCAL_DEMO_PRESENTATION ? `Answered by ${mLabel}.` : `Answered by Groq Cloud (${mLabel}).`);
        aiStatus(`ready — ${stats}`);
        if (!buildModeRequest && reply && !wasAborted) {
          ai.history.push({ role: "assistant", content: reply });
        }
      }
    } catch (err) {
      if (ai.abortGen) {
        const stats = `${tokenCount} tok · stopped by user · ${mLabel}`;
        if (buildModeRequest) {
          buildMode.receiveHostError("Build stopped by the host.");
          broadcastAll({ t: "ai-build-error", message: "Build stopped by the host." });
        } else {
          chatBotEnd(reply, stats, false);
          sendChat({ t: "ai-gendone", stats, capped: false }, askerId);
          perfSidebar?.onGenDone?.({ totalTokens: tokenCount, totalSecs: (performance.now() - t0) / 1000, stats });
        }
        aiStatus(`stopped — ${stats}`);
      } else {
        console.error("[Groq] Generation error:", err);
        const displayError = LOCAL_DEMO_PRESENTATION ? err.message.replace(/groq/gi, "model service") : err.message;
        aiStatus((LOCAL_DEMO_PRESENTATION ? "Generation error: " : "Cloud error: ") + displayError);
        const errorLabel = LOCAL_DEMO_PRESENTATION ? "Generation error" : "Cloud Error";
        if (buildModeRequest) {
          buildMode.receiveHostError(displayError);
          broadcastAll({ t: "ai-build-error", message: displayError });
        } else {
          chatBotEnd((continuation.prefixReply || "") + `\n\n⚠ **${errorLabel}**: ${displayError}`, "");
          sendChat({ t: "ai-gendone", stats: "failed: " + err.message }, askerId);
          perfSidebar?.onGenDone?.({ totalTokens: tokenCount, totalSecs: (performance.now() - t0) / 1000, stats: "failed: " + err.message });
        }
        toast(LOCAL_DEMO_PRESENTATION ? `Generation error: ${displayError}` : `Cloud error: ${displayError}`);
      }
    } finally {
      ai.busy = false;
      ai.abortGen = false;
      ai.abortController = null;
      ai.groqAbortController = null;
      if (!buildModeRequest) setSendButtonState("send");
      if (autoContinuation) {
        aiGenerate(autoContinuation.prompt, asker, askerId, { ...autoContinuation, isContinuation: true });
      } else {
        if ($("ai-model")) $("ai-model").disabled = false;
      }
    }
    return;
  }

  try { ai.engine.reset?.(); } catch {}
  ai.pos = 0;
  for (const [, w] of ai.waiters) {
    if (typeof w?.reject === "function") w.reject(new Error("Generation reset"));
  }
  ai.waiters.clear();
  for (const [, entry] of conns) {
    if (entry.link) resetLink(entry.link);
  }
  broadcastAll({ t: "ai-reset", keepReply: !!continuation.isContinuation });
  ai.lastPrompt = continuation.originalPrompt || text;
  ai.abortGen = false;
  ai.busy = buildModeRequest ? "build" : "gen";
  if (!buildModeRequest) {
    $("ai-prompt").value = "";
    $("ai-prompt").style.height = "auto";
  }
  if (!buildModeRequest) setSendButtonState("stop");
  if ($("ai-model")) $("ai-model").disabled = true;
  const V = ai.tok.vocab;
  const isPhi = MODELS[ai.model]?.arch === "phi3";
  const isDeepSeek = ai.model && ai.model.includes("deepseek");
  let imStart, imEnd, eot, ids;

  if (isDeepSeek) {
    const bos = ai.cfg?.bos_token_id !== undefined ? ai.cfg.bos_token_id : (V["<｜begin of sentence｜>"] ?? V["<｜begin\u2581of\u2581sentence｜>"] ?? V["<|begin of sentence|>"] ?? 151646);
    const eos = ai.cfg?.eos_token_id !== undefined ? ai.cfg.eos_token_id : (V["<｜end of sentence｜>"] ?? V["<｜end\u2581of\u2581sentence｜>"] ?? V["<|end of sentence|>"] ?? 151643);
    const user = V["<｜User｜>"] ?? V["<|User|>"] ?? 151644;
    const asst = V["<｜Assistant｜>"] ?? V["<|Assistant|>"] ?? 151645;
    
    imStart = bos;
    imEnd = eos;
    eot = V["<|endoftext|>"];

    ids = [];
    if (bos !== undefined) ids.push(bos);
    if (user !== undefined) ids.push(user);
    ids.push(...ai.tok.encode(text));
    if (asst !== undefined) ids.push(asst);
  } else if (isPhi) {
    imStart = V["<|user|>"]; imEnd = V["<|end|>"]; eot = V["<|endoftext|>"];
    ids = [imStart, ...ai.tok.encode("\n" + text), imEnd, ...ai.tok.encode("\n"), V["<|assistant|>"], ...ai.tok.encode("\n")];
  } else {
    imStart = V["<|im_start|>"]; imEnd = V["<|im_end|>"]; eot = V["<|endoftext|>"];
    ids = [imStart, ...ai.tok.encode("system\nYou are a helpful assistant."), imEnd, ...ai.tok.encode("\n"), imStart, ...ai.tok.encode("user\n" + text), imEnd, ...ai.tok.encode("\n"), imStart, ...ai.tok.encode("assistant\n")];
  }
  // Fast Mode (default): pre-close the think block so Qwen3 skips the 100+ token monologue and generates the answer immediately!
  const isThinkingModel = MODELS[ai.model]?.thinking === true;
  const wantThinking = ai.thinkingMode === "deep";
  if (isThinkingModel && !wantThinking) {
    if (V["<think>"] !== undefined && V["</think>"] !== undefined) {
      ids.push(V["<think>"], ...ai.tok.encode("\n\n"), V["</think>"], ...ai.tok.encode("\n\n"));
    } else {
      ids.push(...ai.tok.encode("<think>\n\n</think>\n\n"));
    }
  }
  if (ids.some((t) => !Number.isInteger(t)))
    throw new Error("tokenizer produced an invalid token id (special tokens missing) \u2014 " + JSON.stringify(ids.slice(0, 6)));

  const eosIds = new Set([imEnd, eot, imStart].filter((t) => Number.isInteger(t)));
  if (Number.isInteger(ai.cfg?.eos_token_id)) eosIds.add(ai.cfg.eos_token_id);
  else if (Array.isArray(ai.cfg?.eos_token_id)) for (const id of ai.cfg.eos_token_id) eosIds.add(id);

  if (!buildModeRequest && !continuation.isContinuation) {
    chatUser(asker, text);
    chatBotStart();
    const modelLabel = MODELS[ai.model]?.label?.split("·")[0]?.trim() || ai.model;
    const deviceCount = Math.max(1, perfSidebar?.devices?.filter(d => d.workerRole && d.workerRole !== "Idle").length || 1);
    sendChat({ t: "ai-genstart", name: asker, text, model: modelLabel, deviceCount, backend: "local" }, askerId);
    perfSidebar?.onGenStart?.({ model: modelLabel, deviceCount, backend: "local" });
  }
  if (buildModeRequest) {
    const modelLabel = MODELS[ai.model]?.label?.split("·")[0]?.trim() || ai.model;
    buildMode.receiveHostStart({ model: modelLabel, repair: Boolean(continuation.repair) });
    broadcastAll({ t: "ai-build-start", model: modelLabel, repair: Boolean(continuation.repair) });
  }
  mascot("Thinking… every word is taking a lap through the room.");
  aiStatus(`prefill: ${ids.length} tokens…`);

  let autoContinuation = null;
  try {
    // the prompt must fit the context with room for an answer; never silently truncate
    const maxContext = Math.min(MAX_SEQ, ai.engine?.maxSeq || MAX_SEQ);
    if (ids.length > maxContext - MIN_ROOM)
      throw new Error(`prompt is ${ids.length} tokens; this room's context is ${maxContext} tokens and an answer needs at least ${MIN_ROOM}. Shorten the prompt.`);
    const maxNew = Math.min(MAX_NEW, maxContext - ids.length);   // answer cap for this prompt
    let capped = false;   // set when generation stops because the context filled up
    let logits = null;
    const tPre = performance.now();
    
    if (ai.clusterDegraded) throw new Error("Cluster degraded \u2014 required worker disconnected. Please reload the model.");

    // A true solo model has no chain AND owns all layers.
    const trueSolo = !ai.chain.length && ai.range && ai.range[0] === 0 && ai.range[1] >= (ai.cfg?.num_hidden_layers || 999);

    if (trueSolo && ai.engine.prefillTokens && ids.length > 1) {
      // solo: batched prefill, 4 prompt tokens per GPU pass
      ai.engine.pos = ai.pos;
      await ai.engine.prefillTokens(ids.slice(0, -1));
      ai.pos = ai.engine.pos;
      logits = await aiPipeToken(ids[ids.length - 1]);
    } else if (!trueSolo && ai.chain.length && ai.engine.embedRunBatch && ai.engine.mtp && ids.length > 5) {
      // split: speculative/MTP model with tested batched prefill (Qwen 3.8 27B)
      let i = 0;
      const hdim = ai.engine.dims.dim;
      const NC = ai.engine.NC || 4;   // columns per GPU pass; up to 16 tokens per network round
      // step down 16 -> 8 -> 4 on the tail: without this a remainder of up to
      // NC-1 tokens costs one network lap each
      const widths = [NC, ...[8, 4].filter((w) => w < NC)];
      try {
        for (const W of widths) while (ids.length - 1 - i >= W) {
          const nChunks = Math.max(1, Math.min(Math.floor(16 / W), Math.floor((ids.length - 1 - i) / W)));
          const NCW = W;
          const basePos = ai.pos;
          const hb = new Float32Array(nChunks * NCW * hdim);
          for (let c = 0; c < nChunks; c++) {
            console.log(`[Diagnostic] Host batch prefill chunk ${c+1}/${nChunks} start`);
            hb.set(await ai.engine.embedRunBatch(ids.slice(i + c * NCW, i + (c + 1) * NCW), basePos + c * NCW), c * NCW * hdim);
            console.log(`[Diagnostic] Host batch prefill chunk ${c+1}/${nChunks} done`);
          }
          if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
          if (ai.chain.length) {
            const returned = new Promise((res, rej) => {
              ai.waiters.set("b" + basePos, { resolve: res, reject: rej });
              setTimeout(() => {
                if (ai.waiters.has("b" + basePos)) {
                  ai.waiters.delete("b" + basePos);
                  rej(new Error("pipeline timeout (batch prefill)"));
                }
              }, 15000);
            });
            console.log(`[Diagnostic] Host sending ai-hidden-b to peer...`);
            sendHidden(ai.chain[0], { t: "ai-hidden-b", basePos, n: nChunks * NCW, ...packWire(hb) });
            console.log(`[Diagnostic] Host awaiting returned batch from peer...`);
            await returned;
            console.log(`[Diagnostic] Host received returned batch from peer!`);
          }
          ai.pos = basePos + nChunks * NCW;
          i += nChunks * NCW;
          aiStatus(`prefill: ${i}/${ids.length} tokens\u2026`);
        }
      } catch (batchErr) {
        console.warn("Batched prefill failed or timed out; falling back to sequential token prefill:", batchErr);
        aiStatus(`batch prefill skipped, falling back to sequential tokens…`);
        for (const [k] of ai.waiters) {
          if (String(k).startsWith("b")) ai.waiters.delete(k);
        }
      }
      for (; i < ids.length; i++) logits = await aiPipeToken(ids[i], i === ids.length - 1);
    } else {
      for (let i = 0; i < ids.length; i++) {
        if (i % 4 === 0 || i === ids.length - 1) aiStatus(`prefill: ${i + 1}/${ids.length} tokens…`);
        logits = await aiPipeToken(ids[i], i === ids.length - 1);
      }
    }
    const t0 = performance.now();
    let count = 0, reply = continuation.prefixReply || "";
    const streamDecoder = ai.tok.createStreamDecoder();
    const emit = (tok) => {
      const piece = streamDecoder.decode([tok]);
      reply += piece;
      count++;
      if (buildModeRequest) {
        buildMode.receiveHostToken(piece);
        broadcastAll({ t: "ai-build-token", text: piece });
      } else {
        chatBotUpdate(reply);
        sendChat({ t: "ai-token", text: piece }, askerId);
        perfSidebar?.onToken?.(piece, (continuation.priorCount || 0) + count);
      }
      aiStatus(`generating… ${count} tok · ${(count / ((performance.now() - t0) / 1000)).toFixed(1)} tok/s`);
    };
    const recentTokens = [];
    const sample = (lgt) => aiSample(lgt, 0.8, 40, recentTokens, 1.15);
    if (ai.engine.mtp && ai.engine.specStep) {
      // speculative decoding: the model's own draft head proposes up to 3 tokens,
      // one batched trunk pass verifies them (byte-identical to plain decoding)
      const spec = ai.chain.length ? {
        runTrunk: async (tokens, pos) => {
          const tLap = performance.now();
          const n = tokens.length, hdim = ai.engine.dims.dim, NC = ai.engine.NC || 4;
          const hb = new Float32Array(n * hdim);
          for (let c = 0; c < n; c += NC) {
            const m = Math.min(NC, n - c);
            hb.set(await ai.engine.embedRunBatch(tokens.slice(c, c + m), pos + c, { base: c, total: n }), c * hdim);
          }
          if (badF32(hb)) throw new Error(`NaN after HOST layers (pos ${pos})`);
          const returned = new Promise((res, rej) => {
            ai.waiters.set("b" + pos, { resolve: res, reject: rej });
            setTimeout(() => {
              if (ai.waiters.has("b" + pos)) {
                ai.waiters.delete("b" + pos);
                rej(new Error("pipeline timeout (verify)"));
              }
            }, 15000);
          });
          sendHidden(ai.chain[0], { t: "ai-hidden-b", basePos: pos, n: tokens.length, spec: 1, ...packWire(hb) });
          const h = await returned;
          if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos})`);
          const dt = performance.now() - tLap;
          ai.lapMs = ai.lapMs ? 0.7 * ai.lapMs + 0.3 * dt : dt;
          return h;
        },
        onReject: async (k) => { for (const id of ai.chain) sendTo(id, { t: "ai-rollback", k }); },
      } : {};
      if (ai.chain.length && ai.lastHidden) ai.engine.setHidden(ai.lastHidden);
      ai.engine.pos = ai.pos;
      ai.lapMs = 0;
      // draft depth: pick by MEASURED tokens/sec per depth (K=3 warm-up, probe
      // 5 and 7 once, keep the best, re-probe now and then). Deep chains only
      // pay when the network round-trip dominates the lap; a lap-time
      // threshold can't tell GPU time from RTT and gets stuck deep.
      const kc = { cand: [3, 5, 7], ema: {}, n: {}, step: 0, used: {} };
      const pickK = () => {
        if (!ai.chain.length) return 3;
        kc.step++;
        if (kc.step <= 3) return 3;
        const untried = kc.cand.find((k) => !kc.n[k]);
        if (untried) return untried;
        let best = 3;
        for (const k of kc.cand) if (kc.ema[k] > kc.ema[best]) best = k;
        if (kc.step % 16 === 0) { const alt = kc.cand.filter((k) => k !== best); return alt[(kc.step / 16) % alt.length | 0]; }
        return best;
      };
      // the first answer token is sampled here; specStep treats it as already chosen for this
      // position and returns only the tokens after it, so it has to be emitted (or end the
      // answer) before the loop, or the reply starts one word late
      let next = sample(logits), done = false;
      if (eosIds.has(next)) done = true; else { emit(next); recentTokens.push(next); }
      while (!done && count < maxNew) {
        if (ai.abortGen) { done = true; capped = false; break; }
        // a speculative step touches positions pos .. pos+K (K drafts verified in one pass) and
        // drafts one more; shrink K near the end of the context and stop before it overflows
        let K = pickK();
        const roomLeft = maxContext - ai.engine.pos - 2;
        if (roomLeft < 1) { capped = true; break; }
        K = Math.min(K, roomLeft, maxNew - count + 1);
        const tStep = performance.now();
        const toks = await ai.engine.specStep(next, sample, K, spec);
        const tps = toks.length / ((performance.now() - tStep) / 1000);
        kc.ema[K] = kc.n[K] ? 0.6 * kc.ema[K] + 0.4 * tps : tps;
        kc.n[K] = (kc.n[K] || 0) + 1; kc.used[K] = (kc.used[K] || 0) + toks.length;
        for (const tk of toks) {
          if (eosIds.has(tk)) { done = true; break; }
          if (count >= maxNew) { done = true; capped = true; break; }
          emit(tk);
          recentTokens.push(tk);
        }
        next = toks[toks.length - 1];
      }
      if (!done && count >= maxNew) capped = true;
      ai.pos = ai.engine.pos;
      const st = ai.engine.mtp.stats;
      if (st.drafts) crumb(`spec: ${st.accepted}/${st.drafts} drafts accepted${ai.lapMs ? ` · lap ${Math.round(ai.lapMs)}ms` : ""}`
        + (ai.chain.length ? ` · K tok/s ${kc.cand.map((k) => `${k}:${kc.ema[k] ? kc.ema[k].toFixed(1) : "-"}`).join(" ")} · tokens by K ${JSON.stringify(kc.used)}` : ""));
    } else {
      let hitEos = false;
      for (let i = 0; i < maxNew; i++) {
        if (ai.abortGen) { capped = false; break; }
        const next = sample(logits);
        if (eosIds.has(next)) { hitEos = true; await aiPipeToken(next, false); break; }
        emit(next);
        recentTokens.push(next);
        if (ai.pos >= maxContext - 1 || i === maxNew - 1 || count >= maxNew) { capped = true; break; }   // no position left for another token
        logits = await aiPipeToken(next);
      }
      if (!hitEos && !capped && count >= maxNew) capped = true;
    }
    const finalPiece = streamDecoder.finish();
    if (finalPiece) {
      reply += finalPiece;
      if (buildModeRequest) {
        buildMode.receiveHostToken(finalPiece);
        broadcastAll({ t: "ai-build-token", text: finalPiece });
      } else {
        chatBotUpdate(reply);
        sendChat({ t: "ai-token", text: finalPiece }, askerId);
      }
    }
    const secs = (performance.now() - t0) / 1000;
    const wasAborted = ai.abortGen;
    if (capped && !wasAborted && (continuation.autoDepth || 0) < 2) {
      const excerpt = reply.slice(-1000);
      autoContinuation = {
        prompt: `Continue answering this original request: ${continuation.originalPrompt || text}\nContinue immediately after the excerpt below. Do not repeat it. If this is code, keep it as one complete file and finish any unfinished blocks. Output only the continuation.\n\n${excerpt}`,
        prefixReply: reply,
        originalPrompt: continuation.originalPrompt || text,
        autoDepth: (continuation.autoDepth || 0) + 1,
        priorCount: (continuation.priorCount || 0) + count,
        priorSecs: (continuation.priorSecs || 0) + secs,
        buildMode: buildModeRequest,
        repair: Boolean(continuation.repair),
      };
      aiStatus("answer reached the room's context limit; continuing automatically…");
      mascot("Continuing the answer automatically…");
    } else {
      const totalCount = (continuation.priorCount || 0) + count;
      const totalSecs = (continuation.priorSecs || 0) + secs;
      const stats = `${totalCount} tok · ${(totalCount / (totalSecs || 0.001)).toFixed(1)} tok/s · ${ai.chain.length + 1} devices${wasAborted ? " · stopped by user" : capped ? ` · stopped: context limit reached (${MAX_SEQ} tokens)` : ""}`;
      if (buildModeRequest) {
        if (wasAborted) buildMode.receiveHostError("Build stopped by the host.");
        else buildMode.receiveHostDone(reply, { stats });
        broadcastAll({ t: "ai-build-done", text: reply, stats, aborted: wasAborted });
      } else {
        chatBotEnd(reply, stats, capped && !wasAborted);
        sendChat({ t: "ai-gendone", stats, capped: capped && !wasAborted }, askerId);
        perfSidebar?.onGenDone?.({ totalTokens: totalCount, totalSecs, stats });
      }
      mascot(wasAborted ? "Generation stopped." : capped ? "Context limit reached. Ask to continue or start a new question." : "Done. Anyone in the room can ask the next one.");
      aiStatus(`ready — prefill ${((t0 - tPre) / 1000).toFixed(1)}s, ${stats}`);
    }
  } catch (err) {
    aiStatus("generation failed: " + err.message);
    if (buildModeRequest) {
      buildMode.receiveHostError(err.message);
      broadcastAll({ t: "ai-build-error", message: err.message });
    } else {
      chatBotEnd((continuation.prefixReply || "") + "\n\n⚠ " + err.message, "");
      sendChat({ t: "ai-gendone", stats: "failed: " + err.message }, askerId);   // unlock everyone's send box
      perfSidebar?.onGenDone?.({ totalTokens: count, totalSecs: (performance.now() - t0) / 1000, stats: "failed: " + err.message });
    }
  } finally {
    ai.busy = false;
    ai.abortGen = false;
    if (!buildModeRequest) setSendButtonState("send");
    if (autoContinuation) {
      aiGenerate(autoContinuation.prompt, asker, askerId, { ...autoContinuation, isContinuation: true });
    } else {
      if ($("ai-model")) $("ai-model").disabled = false;
    }
  }
}

// ---- worker + shared message handling ----
async function aiOnData(from, d) {
  const e = conns.get(from);
  switch (d.t) {
    case "ai-start-req":
      if (MODELS[d.model]) $("ai-model").value = d.model;
      $("ai-start").disabled = true;
      $("ai-model").disabled = true;
      if (isHost) {
        toast(`${d.by || "peer"} started ${MODELS[d.model]?.label.split("\u00b7")[0].trim() || d.model}`);
        aiStart(d.model);
      } else {
        aiLoading(true, `starting ${MODELS[d.model]?.label.split("\u00b7")[0].trim() || d.model}`);
        $("ldg-sub").textContent = `${d.by || "peer"} pressed start`;
        $("ldg-fill").style.width = "0%";
        aiStatus(`${d.by || "peer"} started the model\u2026`);
      }
      break;
    case "ai-build-request":
      if (!isHost && ai.role !== "host") break;
      if (ai.busy) { sendTo(from, { t: "ai-build-error", message: "The room model is busy. Wait for the current response to finish." }); break; }
      if (!isGroqMode && !ai.engine) { sendTo(from, { t: "ai-build-error", message: "Load the room model before building an app." }); break; }
      sendTo(from, { t: "ai-build-state", state: { files: buildMode.files, revision: buildMode.revision } });
      aiGenerate(d.text, d.name || e?.name || "room", from, { buildMode: true, repair: Boolean(d.repair) });
      break;
    case "ai-build-state-request":
      if (isHost || ai.role === "host") sendTo(from, { t: "ai-build-state", state: { files: buildMode.files, revision: buildMode.revision } });
      break;
    case "ai-build-stop":
      if ((isHost || ai.role === "host") && ai.busy === "build") {
        ai.abortGen = true;
        ai.abortController?.abort();
        for (const [, waiter] of ai.waiters || []) waiter?.reject?.(new Error("Build stopped by the host."));
        ai.waiters?.clear?.();
      }
      break;
    case "ai-build-start":
      buildMode.receiveHostStart(d);
      break;
    case "ai-build-token":
      buildMode.receiveHostToken(d.text || "");
      break;
    case "ai-build-done":
      if (d.aborted) buildMode.receiveHostError("Build stopped by the host.");
      else buildMode.receiveHostDone(d.text || "", { stats: d.stats });
      break;
    case "ai-build-error":
      buildMode.receiveHostError(d.message || "Build generation failed.");
      break;
    case "ai-build-state":
      buildMode.receiveState(d.state);
      break;
    case "ai-layers":
      ai.layersByName = d.by;
      loadCardRender();
      updateCluster();
      break;
    case "ai-next":
      ai.next = d.next;
      ensureLink(d.next);
      break;
    case "ai-reset":
      try { ai.engine?.reset?.(); } catch {}
      ai.pos = 0;
      if (!d.keepReply) ai.remoteReply = "";
      if (ai.waiters) {
        for (const [, w] of ai.waiters) {
          if (typeof w?.reject === "function") w.reject(new Error("Generation reset"));
        }
        ai.waiters.clear();
      }
      for (const [, entry] of conns) {
        if (entry.link) resetLink(entry.link);
      }
      break;
    case "ai-think-mode":
      ai.thinkingMode = d.mode;
      updateThinkModeUI(d.mode);
      break;
    case "ai-wait":
      ai.role = "worker"; ai.hostId = from;
      aiLoading(true, "Syncing with the room");
      $("ldg-sub").textContent = "your turn comes after they finish downloading. keep this screen on.";
      $("ldg-fill").style.width = "0%";
      aiStatus("syncing with the room\u2026");
      break;
    case "ai-load": {
      const isFallback = Boolean(fallbackmode || (typeof window !== "undefined" && window.fallbackmode) || d.fallbackmode);
      if (isFallback) {
        if (MODELS[d.model]) $("ai-model").value = d.model;
        ai.role = "worker";
        ai.hostId = d.host;
        aiLoading(false);
        $("load-card").classList.remove("on");
        $("ai-panel").classList.remove("loading");
        $("ai-panel").classList.add("online");
        $("ai-row").style.display = "flex";
        $("ai-empty").style.display = "none";
        renderWelcomePrompts();
        aiStatus("ready · Cloud API (Qwen 27B) — no download needed");
        sendTo(ai.hostId, { t: "ai-ready", from: peer.id });
        break;
      }
      if (MODELS[d.model]) $("ai-model").value = d.model;
      ai.role = "worker";
      ai.next = d.next;
      ai.hostId = d.host;
      ai.stageByName = {};
      if (d.cfg) ai.cfg = { ...(ai.cfg || {}), ...d.cfg };
      ensureLink(d.next);   // open the link to my chain neighbour while the weights download
      try {
        await detectLocalModel(d.model || "smollm-135m");
        await aiLoadShard(d.model || "smollm-135m", d.range, false, false);
        if (!(await ensureLink(d.next))) throw new Error("could not connect to the next device in the chain");
        aiStatus(`${formatLayerRange(d.range, false)} ready \u00b7 syncing with the room\u2026`);
        aiLoading(true, `${formatLayerRange(d.range, false)} ready`);
        $("ldg-sub").textContent = "syncing with the rest of the room";
        $("ldg-fill").style.width = "100%";
        sendTo(ai.hostId, { t: "ai-ready", from: peer.id });
        if (ai.readyRetryTimer) clearInterval(ai.readyRetryTimer);
        ai.readyRetryTimer = setInterval(() => {
          if ($("ai-panel").classList.contains("online")) {
            clearInterval(ai.readyRetryTimer);
            ai.readyRetryTimer = null;
            return;
          }
          sendTo(ai.hostId, { t: "ai-ready", from: peer.id });
        }, 1500);
      } catch (err) {
        if (ai.readyRetryTimer) { clearInterval(ai.readyRetryTimer); ai.readyRetryTimer = null; }
        aiLoading(false);
        aiStatus("failed: " + err.message);
        sendTo(ai.hostId, { t: "ai-error", message: err.message });
      }
      break;
    }
    case "ai-hostprog": {
      const now = Date.now();
      ai.prog = { ...(d.all || {}), [myName]: Math.round(ai.myPct || 0) };
      ai.progAt = ai.progAt || {};
      for (const nm of Object.keys(d.all || {})) if (nm !== myName) ai.progAt[nm] = now;
      loadCardRender();
      break;
    }
    case "ai-progress":
      if (e?.card) e.card.querySelector(".bw").textContent = "dl " + d.pct + "%";
      ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {};
      ai.prog[e?.name || from] = d.pct; ai.progAt[e?.name || from] = Date.now(); loadCardRender();
      break;
    case "ai-stage": {
      const name = e?.name || from;
      ai.stageByName = ai.stageByName || {};
      ai.stageByName[name] = d.stage || "working";
      loadCardRender();
      break;
    }
    case "ai-ready":
      ai.readyPeers.add(from);
      if (d.from) ai.readyPeers.add(d.from);
      ai.stageByName = ai.stageByName || {};
      ai.stageByName[e?.name || from] = "ready";
      if (e?.card) e.card.querySelector(".bw").textContent = "ready";
      loadCardRender();
      aiMaybeReady();
      break;
    case "ai-error":
      ai.stageByName = ai.stageByName || {};
      ai.stageByName[e?.name || from] = "failed";
      loadCardRender();
      aiStatus(`peer ${e?.name || from} failed: ${d.message}`);
      if (ai.waiters && ai.waiters.size > 0) {
        for (const [key, waiter] of ai.waiters) {
          ai.waiters.delete(key);
          if (typeof waiter?.reject === "function") waiter.reject(new Error(d.message || "peer failed"));
          else if (typeof waiter === "function") waiter(null);
        }
      }
      break;
    case "ai-hidden-b": {
      // worker: n hiddens in (multiple of 4), my layers (batched), n hiddens on
      if (!ai.engine) {
        if (ai.hostId) sendTo(ai.hostId, { t: "ai-error", message: "worker engine not ready" });
        return;
      }
      try {
        const xs = unpackWire(d);
        const nTok = d.n || 4;
        const wdim = ai.engine.dims.dim;
        const hb = new Float32Array(nTok * wdim);
        const NC = ai.engine.NC || 4;
        for (let c = 0; c < nTok; c += NC) {
          const m = Math.min(NC, nTok - c);
          const chunk = await ai.engine.runHiddenBatch(xs.subarray(c * wdim, (c + m) * wdim), d.basePos + c, d.spec ? { base: c, total: nTok } : false);
          hb.set(chunk.subarray ? chunk.subarray(0, m * wdim) : chunk, c * wdim);
        }
        if (badF32(hb)) {
          aiStatus(`\u26a0 NaN in batched prefill on this device`);
          sendTo(ai.hostId, { t: "ai-error", message: "NaN in batched prefill" });
          return;
        }
        const bmsg = { basePos: d.basePos, n: nTok, ...packWire(hb) };
        let sent = false;
        if (ai.next === "host") sent = sendHidden(ai.hostId, { t: "ai-hiddenret-b", ...bmsg });
        else sent = sendHidden(ai.next, { t: "ai-hidden-b", ...bmsg });
        if (!sent) {
          console.warn(`Worker failed to send hidden-b at basePos ${d.basePos} to ${ai.next}`);
          if (ai.hostId) sendTo(ai.hostId, { t: "ai-error", message: `failed to transmit batch activations along chain to ${ai.next}` });
        }
      } catch (err) {
        console.error("Worker ai-hidden-b execution error:", err);
        aiStatus(`\u26a0 worker batch prefill error: ${err.message}`);
        if (ai.hostId) sendTo(ai.hostId, { t: "ai-error", message: `worker error: ${err.message}` });
      }
      break;
    }
    case "ai-rollback": {
      // host rejected a speculative suffix: recurrent state back to after column k
      ai.engine?.restoreDN?.(d.k);
      break;
    }
    case "ai-hiddenret-b": {
      const w = ai.waiters.get("b" + d.basePos);
      if (w) {
        ai.waiters.delete("b" + d.basePos);
        const resData = unpackWire(d);
        if (typeof w?.resolve === "function") w.resolve(resData);
        else if (typeof w === "function") w(resData);
      }
      break;
    }
    case "ai-hidden": {
      // worker: run my layers, forward along the chain
      if (!ai.engine) {
        if (ai.hostId) sendTo(ai.hostId, { t: "ai-error", message: "worker engine not ready" });
        return;
      }
      try {
        const hin = unpackWire(d);
        if (badF32(hin)) { aiStatus(`\u26a0 NaN ARRIVED at this device (pos ${d.pos}) \u2014 upstream peer broken`); }
        const h = await ai.engine.runHidden(hin, d.pos);
        if (badF32(h)) {
          aiStatus(`\u26a0 NaN PRODUCED by this device (pos ${d.pos}, ${formatLayerRange(ai.range, false)}) \u2014 GPU kernel issue here`);
          sendTo(ai.hostId, { t: "ai-error", message: `NaN produced on worker ${formatLayerRange(ai.range, false)}` });
          return;
        }
        const msg = { pos: d.pos, ...packWire(h) };
        let sent = false;
        if (ai.next === "host") sent = sendHidden(ai.hostId, { t: "ai-hiddenret", ...msg });
        else sent = sendHidden(ai.next, { t: "ai-hidden", ...msg });
        if (!sent) {
          console.warn(`Worker failed to send hidden at pos ${d.pos} to ${ai.next}`);
          if (ai.hostId) sendTo(ai.hostId, { t: "ai-error", message: `failed to transmit activations along chain to ${ai.next}` });
        }
        if (d.pos % 8 === 0) aiStatus(`serving ${formatLayerRange(ai.range, false)} — pos ${d.pos}`);
      } catch (err) {
        console.error("Worker ai-hidden error:", err);
        if (ai.hostId) sendTo(ai.hostId, { t: "ai-error", message: `worker layer run failed: ${err.message}` });
      }
      break;
    }
    case "ai-hiddenret": {
      // host: pipeline round-trip complete
      const w = ai.waiters.get(d.pos);
      if (w) {
        ai.waiters.delete(d.pos);
        const resData = unpackWire(d);
        if (typeof w?.resolve === "function") w.resolve(resData);
        else if (typeof w === "function") w(resData);
      }
      break;
    }
    case "ai-visibility":
      ai.visibility = d.mode;
      toast(d.mode === "all" ? "the host shows the chat to everyone" : d.mode === "host" ? "the host keeps the chat private" : "the host shows each answer to whoever asked");
      break;
    case "ai-abort":
      if (isHost && ai.busy === "gen") {
        ai.abortGen = true;
        if (ai.waiters) {
          for (const [k, w] of ai.waiters) w.reject(new Error("Generation stopped by user"));
          ai.waiters.clear();
        }
      }
      break;
    case "ai-genstart":
      ai.remoteReply = "";
      chatUser(d.name, d.hidden ? "asked something (the host keeps the chat private)" : d.text);
      chatBotStart();
      setSendButtonState("stop");
      mascot(`${d.name} asked something. Thinking…`);
      perfSidebar?.onGenStart?.({ model: d.model || (MODELS[ai.model]?.label?.split("·")[0]?.trim() || ai.model), deviceCount: d.deviceCount, backend: d.backend || "local" });
      break;
    case "ai-token":
      ai.remoteReply = (ai.remoteReply || "") + d.text;
      chatBotUpdate(ai.remoteReply);
      perfSidebar?.onToken?.(d.text);
      break;
    case "ai-gendone":
      chatBotEnd(d.hidden ? "answer hidden by the host" : (ai.remoteReply || ""), d.stats, d.capped);
      setSendButtonState("send");
      mascot(d.capped ? "Reached context limit. Ask to continue or start fresh." : "Your turn. Ask anything.");
      perfSidebar?.onGenDone?.({ stats: d.stats });
      break;
    case "ai-ready-all":
      if (ai.readyRetryTimer) { clearInterval(ai.readyRetryTimer); ai.readyRetryTimer = null; }
      aiLoading(false);
      $("load-card").classList.remove("on");
      $("ai-panel").classList.remove("loading");
      if (d.groq || isGroqMode) $("ai-panel").classList.add("groq-mode");
      $("ai-panel").classList.add("online");
      if (ai.role !== "host") { ai.role = ai.role || "worker"; ai.hostId = from; }
      if (d.model) {
        ai.model = d.model;
        if ($("ai-model") && MODELS[d.model]) $("ai-model").value = d.model;
      }
      $("ai-row").style.display = "flex";
      renderWelcomePrompts();
      $("ai-empty").style.display = "none";
      if (d.groq || isGroqMode) {
        const cloudModel = getGroqModelId(ai.model || $("ai-model")?.value);
        aiStatus(LOCAL_DEMO_PRESENTATION
          ? `ready · ${MODELS[ai.model]?.label?.split("·")[0]?.trim() || ai.model || "Local model"}`
          : `Groq Cloud ready · ${cloudModel} · prompts sent to Groq`);
        updateGroqModelBadge();
        mascot(LOCAL_DEMO_PRESENTATION
          ? `${MODELS[ai.model]?.label?.split("·")[0]?.trim() || ai.model || "Local model"} is online.`
          : `Groq Cloud is online with ${cloudModel}. Prompts are sent to Groq.`);
      } else {
        $("ai-empty").style.display = "";
        aiStatus(`cluster online · serving ${formatLayerRange(ai.range, ai.role === "host")}`);
        mascot("Cluster online! Type a question, the whole room answers.");
      }
      if ($("ai-model")) $("ai-model").disabled = false;
      ai.busy = false;
      break;
    case "ai-ask":
      if (ai.role !== "host" && !isHost) break;
      if (ai.busy) { sendTo(from, { t: "ai-busy" }); break; }
      aiGenerate(d.text, d.name, from);
      break;
    case "ai-busy": toast("the LLM ShardX is still answering, try again in a moment"); break;
  }
}

$("ai-start").addEventListener("click", aiStartAnywhere);
$("ai-visibility").addEventListener("change", (e) => {
  ai.visibility = e.target.value;
  broadcastAll({ t: "ai-visibility", mode: ai.visibility });
  toast(ai.visibility === "all" ? "everyone sees the chat" : ai.visibility === "host" ? "only you see the chat" : "each answer goes to whoever asked");
});
$("cache-clear").addEventListener("click", async (ev) => {
  ev.preventDefault();
  try { await caches.delete("webslice-weights-v1"); weightCache = null; toast("cached weights cleared"); } catch { toast("could not clear the cache"); }
});
function aiSubmit() {
  if (ai.busy === "gen") {
    ai.abortGen = true;
    ai.abortController?.abort();
    ai.groqAbortController?.abort();
    aiStatus("stopping generation…");
    broadcastAll({ t: "ai-abort" });
    return;
  }
  const promptEl = $("ai-prompt");
  const text = promptEl.value.trim();
  if (!text) return;
  ai.lastPrompt = text;
  if (isHost || ai.role === "host") {
    aiGenerate();
    return;
  }
  const hostId = ai.hostId;
  if (hostId && conns.has(hostId)) {
    promptEl.value = "";
    promptEl.style.height = "auto";
    sendTo(hostId, { t: "ai-ask", text, name: myName });
    return;
  }
  aiGenerate();
}
$("ai-send").addEventListener("click", aiSubmit);
$("ai-prompt").addEventListener("input", () => {
  const el = $("ai-prompt");
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
});
$("ai-prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    aiSubmit();
  }
});
function updateThinkModeUI(mode) {
  const fastBtn = $("mode-fast"), deepBtn = $("mode-deep");
  if (!fastBtn || !deepBtn) return;
  if (mode === "deep") {
    fastBtn.classList.remove("active");
    deepBtn.classList.add("active");
  } else {
    deepBtn.classList.remove("active");
    fastBtn.classList.add("active");
  }
}

function setupThinkingModeToggle() {
  ai.thinkingMode = localStorage.getItem("webslice_think_mode") || "fast";
  updateThinkModeUI(ai.thinkingMode);
  const fastBtn = $("mode-fast"), deepBtn = $("mode-deep");
  if (fastBtn) {
    fastBtn.addEventListener("click", () => {
      ai.thinkingMode = "fast";
      localStorage.setItem("webslice_think_mode", "fast");
      updateThinkModeUI("fast");
      toast("⚡ Fast Mode: instant answers without reasoning delay");
      broadcastAll({ t: "ai-think-mode", mode: "fast" });
    });
  }
  if (deepBtn) {
    deepBtn.addEventListener("click", () => {
      ai.thinkingMode = "deep";
      localStorage.setItem("webslice_think_mode", "deep");
      updateThinkModeUI("deep");
      toast("🧠 Deep Thinking: generating full chain of thought");
      broadcastAll({ t: "ai-think-mode", mode: "deep" });
    });
  }
}
setupThinkingModeToggle();

function updateFallbackModeUI(enabled) {
  const advancedMoE = $("ai-model")?.querySelector('option[value="qwen3.6-35b-moe"]');
  if (advancedMoE) advancedMoE.disabled = Boolean(enabled);
  const btn = $("mode-fallback");
  if (btn) {
    btn.classList.toggle("active", enabled);
    const label = btn.querySelector(".fallback-label");
    if (label) {
    label.textContent = LOCAL_DEMO_PRESENTATION
      ? (enabled ? "Local model: On" : "Use local model")
      : (enabled ? "Groq Cloud: On" : "Use Groq Cloud");
    }
    btn.title = LOCAL_DEMO_PRESENTATION
      ? (enabled ? "Load the selected model from cache." : "Switch to local model mode.")
      : enabled
      ? "Groq Cloud is active. Prompts are sent to Groq; model weights are not downloaded."
      : "Switch to Groq Cloud. Prompts will be sent to Groq; model weights will not be downloaded.";
  }

  const sideCard = $("sidebar-fallback-card");
  if (sideCard) {
    sideCard.classList.toggle("active", enabled);
    sideCard.title = LOCAL_DEMO_PRESENTATION
      ? "Toggle local model mode"
      : enabled
        ? "Cloud inference sends prompts to Groq without downloading weights"
        : "Local WebGPU inference runs in this browser and room";
  }
  const sideToggle = $("sidebar-fallback-toggle");
  if (sideToggle) {
    sideToggle.classList.toggle("active", enabled);
  }
  const sfcDesc = $("sfc-desc");
  if (sfcDesc) {
    const model = $("ai-model")?.value || "qwen3.8-27b";
    sfcDesc.textContent = LOCAL_DEMO_PRESENTATION
      ? (enabled ? "Local model · ready to start" : "Local model mode")
      : enabled
      ? `Cloud · prompts sent to Groq · ${getGroqModelId(model)}`
      : "Local WebGPU · prompts stay in this room.";
  }
  const disclosure = $("mode-disclosure");
  if (disclosure) {
    disclosure.textContent = LOCAL_DEMO_PRESENTATION
      ? (enabled ? "Local model ready from cache." : "Choose a model and press Start to load it.")
      : enabled
        ? `Cloud inference is active. Prompts go to Groq, using ${getGroqModelId($("ai-model")?.value)}; model weights are not downloaded.`
        : "Local WebGPU inference is active. Prompts stay with this device and room.";
  }
  updateGroqModelBadge();
}

function toggleFallbackMode(forceState) {
  const userInitiated = typeof forceState !== "boolean";
  fallbackmode = typeof forceState === "boolean" ? forceState : !fallbackmode;
  isGroqMode = fallbackmode;
  if (typeof window !== "undefined") {
    window.fallbackmode = fallbackmode;
    window.isGroqMode = isGroqMode;
    try {
      localStorage.setItem("webslice_fallbackmode", fallbackmode ? "true" : "false");
    } catch {}
  }
  updateFallbackModeUI(fallbackmode);
  perfSidebar?.setBackend?.(fallbackmode ? "cloud" : "local");
  if (fallbackmode) {
    aiLoading(false);
    if ($("load-card")) $("load-card").classList.remove("on");
    if ($("ai-panel")) {
      $("ai-panel").classList.remove("loading");
      $("ai-panel").classList.add("groq-mode");
      $("ai-panel").classList.add("online");
    }
    if ($("ai-row")) $("ai-row").style.display = "flex";
    if ($("ai-empty")) $("ai-empty").style.display = "none";
    if ($("ai-start")) $("ai-start").disabled = false;
    updateNeed(0);
    updateCluster();
    const m = $("ai-model")?.value || "qwen3.8-27b";
    if (userInitiated) toast(LOCAL_DEMO_PRESENTATION ? "Local model mode enabled" : `⚡ Groq Cloud enabled: ${getGroqModelId(m)}`);
    aiStatus(LOCAL_DEMO_PRESENTATION ? "local model ready" : `Groq Cloud active · ${getGroqModelId(m)} · prompts sent to Groq`);
    mascot(LOCAL_DEMO_PRESENTATION ? "Local model mode is ready." : `Groq Cloud is active. Prompts are sent to Groq; ${getGroqModelId(m)} answers without downloading model weights.`);
  } else {
    if ($("ai-panel")) $("ai-panel").classList.remove("groq-mode");
    if (!ai.engine) {
      if ($("ai-panel")) $("ai-panel").classList.remove("online");
      /* if ($("ai-row")) $("ai-row").style.display = "none"; */
      if ($("ai-empty")) {
        $("ai-empty").style.display = "";
        $("ai-empty").textContent = "pick a model and press start";
      }
    }
    if ($("ai-start")) $("ai-start").style.display = "";
    if ($("ai-need")) $("ai-need").style.display = "";
    updateCluster();
    if (userInitiated) toast(LOCAL_DEMO_PRESENTATION ? "Local model mode disabled" : "Groq Cloud off: using local WebGPU LLM ShardX");
    aiStatus(ai.engine ? `cluster online · serving ${formatLayerRange(ai.range, ai.role === "host")}` : LOCAL_DEMO_PRESENTATION ? "pick a model and press start" : "split across every device in the room");
    mascot(LOCAL_DEMO_PRESENTATION ? "Choose a model and press Start to load it." : "LLM ShardX WebGPU mode active. Pick a model and press Start to download weights.");
  }
}

function setupFallbackModeToggle() {
  updateFallbackModeUI(fallbackmode);
  perfSidebar?.setBackend?.(fallbackmode ? "cloud" : "local");
  updateGroqModelBadge();
  if (LOCAL_DEMO_PRESENTATION) {
    if ($("mode-fallback")) $("mode-fallback").style.display = "none";
    if ($("sidebar-fallback-card")) $("sidebar-fallback-card").style.display = "none";
    return;
  }
  const btn = $("mode-fallback");
  if (btn) {
    btn.addEventListener("click", () => toggleFallbackMode());
  }
  const sideCard = $("sidebar-fallback-card");
  if (sideCard) {
    sideCard.addEventListener("click", () => toggleFallbackMode());
  }
  const sideToggle = $("sidebar-fallback-toggle");
  if (sideToggle) {
    sideToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFallbackMode();
    });
  }
  if (fallbackmode) {
    if ($("ai-row")) $("ai-row").style.display = "flex";
    if ($("ai-empty")) $("ai-empty").style.display = "none";
  }
}
setupFallbackModeToggle();

mascot("Hi! I'm LLM ShardXy. Create a room, or type a friend's code to join one.");

window.fallbackmode = fallbackmode;
window.toggleFallbackMode = toggleFallbackMode;
window.setGroqApiKey = setGroqApiKey;
window.getGroqApiKey = getGroqApiKey;
window.ai = ai;
window.perfSidebar = perfSidebar;
window.GROQ_MODEL_MAP = GROQ_MODEL_MAP;
window.__roomStart = start;
window.__roomLoaded = true;
window.roomBuildIsHost = () => Boolean(isHost || ai.role === "host");
window.roomBuildBroadcast = (state) => {
  if (!window.roomBuildIsHost()) return;
  broadcastAll({ t: "ai-build-state", state });
};
window.roomBuildRequestState = () => {
  if (!window.roomBuildIsHost() && ai.hostId) sendTo(ai.hostId, { t: "ai-build-state-request" });
};
window.roomBuildGenerate = async (text, options = {}) => {
  if (!window.roomBuildIsHost()) {
    if (!ai.hostId || !conns.has(ai.hostId)) throw new Error("The room host is not connected.");
    sendTo(ai.hostId, { t: "ai-build-request", text, repair: Boolean(options.repair), name: myName });
    return;
  }
  if (ai.busy) throw new Error("The room model is busy. Wait for the current response to finish.");
  if (isGroqMode && $("ai-model")?.value === "qwen3.6-35b-moe") throw new Error("Qwen3.6 MoE runs in local WebGPU mode. Switch off Groq Cloud first.");
  if (!isGroqMode && !ai.engine) throw new Error("Load the room model before building an app.");
  return aiGenerate(text, myName, peer?.id, { buildMode: true, repair: Boolean(options.repair) });
};
window.roomBuildStop = () => {
  if (!window.roomBuildIsHost() || ai.busy !== "build") return;
  ai.abortGen = true;
  ai.abortController?.abort();
  for (const [, waiter] of ai.waiters || []) waiter?.reject?.(new Error("Build stopped by the host."));
  ai.waiters?.clear?.();
};
buildMode.init();
console.log("LLM ShardX room.js initialized successfully (fallbackmode ready, perfSidebar active)");
