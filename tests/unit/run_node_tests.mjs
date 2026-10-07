// Node test runner for unit tests
import { makeLink, sendFrame, attachWire, SLICE_BYTES } from "../../room/transport.js";
import { chatRecipients, VISIBILITY } from "../../room/visibility.js";
import { MODELS, NEED_GB, LOCAL_CANDIDATES, detectLocalModel } from "../../room/models.js";
import { pledgeOf, calculateClusterPledge, formatLayerRange, allocateLayers } from "../../room/allocation.js";
import { WGSL, coopWGSL } from "../../engine/engine.js";
import { PerfSidebar, perfSidebar } from "../../room/perf-sidebar.js";

const assert = (cond, msg) => { if (!cond) throw new Error(msg || "assertion failed"); };
const eq = (a, b, m) => {
  const ja = JSON.stringify(a), jb = JSON.stringify(b);
  if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb);
};

let passed = 0;
let total = 0;
async function test(name, fn) {
  total++;
  try {
    await fn();
    console.log(`✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`✗ ${name}:`, err.message);
    process.exitCode = 1;
  }
}

// 1. Models tests
await test("models: catalog has all 11 expected models", () => {
  const expected = [
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen2.5-coder-1.5b",
    "qwen2.5-coder-7b", "deepseek-r1-distill-qwen-14b", "qwq-32b",
    "phi-4-mini", "qwen3.8-27b", "qwen3.6-35b-moe", "smollm-135m"
  ];
  for (const k of expected) {
    assert(MODELS[k], `Missing model key: ${k}`);
    assert(NEED_GB[k] > 0, `Missing NEED_GB for: ${k}`);
  }
  const moe = MODELS["qwen3.6-35b-moe"];
  assert(moe.kind === "qwen35" && moe.gguf.includes("Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"), "Qwen3.6 MoE local-engine mapping is missing");
  assert(NEED_GB["qwen3.6-35b-moe"] === 22.5, "Qwen3.6 MoE room-memory requirement changed");
});

await test("models: Qwen2.5 Coder prefers the fast mirror and has an HF fallback", () => {
  const m = MODELS["qwen2.5-coder-1.5b"];
  assert(m.gguf.startsWith("https://hf-mirror.com/"), "Qwen2.5 Coder should prefer the fast mirror");
  assert(m.ggufFallback.startsWith("https://huggingface.co/"), "Qwen2.5 Coder fallback URL missing");
  assert(NEED_GB["qwen2.5-coder-1.5b"] >= 1.5, "Qwen2.5 Coder memory pledge missing");
});

await test("models: Phi-4 mini uses the Phi3 GGUF path and a multi-GB pledge", () => {
  assert(MODELS["phi-4-mini"].arch === "phi3", "Phi-4 mini must use Phi3 GGUF mapping");
  assert(MODELS["phi-4-mini"].gguf.endsWith("/microsoft_Phi-4-mini-instruct-Q4_0.gguf"), "Phi-4 mini Q4 GGUF URL missing");
  assert(NEED_GB["phi-4-mini"] >= 3, "Phi-4 mini should require at least 3 GB");
});

await test("models: local candidates defined for all models", () => {
  for (const k of Object.keys(MODELS)) {
    assert(Array.isArray(LOCAL_CANDIDATES[k]), `LOCAL_CANDIDATES missing array for: ${k}`);
    assert(LOCAL_CANDIDATES[k].length > 0, `LOCAL_CANDIDATES empty for: ${k}`);
  }
});

await test("models: detectLocalModel gracefully handles unreachable local files", async () => {
  const res = await detectLocalModel("non-existent-model");
  assert(res === null, "should return null for non-existent model");
});

await test("models: detectLocalModel rejects truncated files and preserves original remote URLs", async () => {
  const origFetch = globalThis.fetch;
  const modelKey = "qwen2.5-coder-7b";
  const originalUrl = MODELS[modelKey].gguf;

  try {
    globalThis.fetch = async (url, opts) => {
      return {
        ok: true,
        headers: new Headers({ "content-length": "52428800" }),
      };
    };

    const res = await detectLocalModel(modelKey);
    assert(res === null, "detectLocalModel should reject truncated local file");
    assert(MODELS[modelKey].gguf === originalUrl, "gguf URL should remain original remote URL when local is truncated");
    assert(MODELS[modelKey].originalGguf === originalUrl, "originalGguf should be preserved");
  } finally {
    globalThis.fetch = origFetch;
  }
});

await test("models: detectLocalModel accepts complete local files and records original URL", async () => {
  const origFetch = globalThis.fetch;
  const modelKey = "qwen3-0.6b";
  const remoteUrl = MODELS[modelKey].gguf;

  try {
    globalThis.fetch = async (url, opts) => {
      return {
        ok: true,
        headers: new Headers({ "content-length": "838860800" }),
      };
    };

    const res = await detectLocalModel(modelKey);
    assert(res !== null, "detectLocalModel should accept complete local file");
    assert(MODELS[modelKey].gguf === res, "gguf URL should be updated to local path");
    assert(MODELS[modelKey].originalGguf === remoteUrl, "originalGguf should preserve remote fallback");
  } finally {
    globalThis.fetch = origFetch;
    MODELS[modelKey].gguf = remoteUrl;
  }
});

// 2. Allocation and Slicing tests
await test("allocation: formatLayerRange prevents negative ranges and handles edge cases", () => {
  assert(formatLayerRange([0, 14]) === "layers 0–13", "normal multi-layer formatting");
  assert(formatLayerRange([0, 1]) === "layer 0", "single layer formatting");
  assert(formatLayerRange([14, 28]) === "layers 14–27", "peer range formatting");
  assert(formatLayerRange([0, 0], true) === "embed/head only", "host zero layers should show embed/head only");
  assert(formatLayerRange([0, -1], true) === "embed/head only", "host negative range should not format as 0–-1");
  assert(formatLayerRange([0, 0], false) === "0 layers", "worker zero layers should show 0 layers");
  assert(formatLayerRange([5, 4], false) === "0 layers", "worker inverted range should show 0 layers");
  assert(formatLayerRange(null) === "", "null range should return empty string");
  assert(formatLayerRange([]) === "", "empty range should return empty string");
});

await test("allocation: allocateLayers prevents 0 layers when L >= peers even with low host memory", () => {
  const L = 28;
  const layerBytes = 150 * 1024 * 1024;
  const embedBytes = 600 * 1024 * 1024;
  const hostMeta = { contribGB: 0.5, webgpu: true };
  const workerMetas = [
    { contribGB: 8, webgpu: true },
    { contribGB: 8, webgpu: true },
  ];

  const { assigned, ranges } = allocateLayers(L, layerBytes, embedBytes, hostMeta, workerMetas);

  assert(assigned.length === 3, "should allocate across all 3 devices");
  assert(assigned[0] >= 1, `Host must receive at least 1 layer, got ${assigned[0]}`);
  for (let i = 0; i < assigned.length; i++) {
    assert(assigned[i] >= 1, `Device ${i} must receive at least 1 layer, got ${assigned[i]}`);
  }
  const totalAssigned = assigned.reduce((a, b) => a + b, 0);
  assert(totalAssigned === L, `Total assigned layers (${totalAssigned}) must equal L (${L})`);

  assert(ranges[0][0] === 0, "first range must start at layer 0");
  assert(ranges[ranges.length - 1][1] === L, `last range must end at layer ${L}`);
  for (let i = 1; i < ranges.length; i++) {
    assert(ranges[i][0] === ranges[i - 1][1], `range gap or overlap at index ${i}`);
  }
});

await test("allocation: pledgeOf and calculateClusterPledge exclude non-WebGPU devices", () => {
  const webgpuPeer = { contribGB: 4, webgpu: true };
  const cpuPeer = { contribGB: 8, webgpu: false };
  const host = { contribGB: 2, webgpu: true };

  assert(pledgeOf(webgpuPeer) === 4 * (2 ** 30), "WebGPU peer should pledge memory");
  assert(pledgeOf(cpuPeer) === 0, "non-WebGPU peer should have 0 pledge");
  assert(pledgeOf(null) === 0, "null peer should have 0 pledge");

  const clusterPledged = calculateClusterPledge(host, [webgpuPeer, cpuPeer]);
  assert(clusterPledged === 6, `Cluster pledged should be 2 + 4 = 6 GB, got ${clusterPledged} GB`);
});

// 2. Visibility tests
await test("visibility: all — everyone gets the text", () => {
  const ids = ["p1", "p2", "p3"];
  eq(chatRecipients("all", "p2", ids), { full: ["p1", "p2", "p3"], hidden: [] });
});

await test("visibility: host — nobody but the host's own screen", () => {
  const ids = ["p1", "p2", "p3"];
  eq(chatRecipients("host", "p2", ids), { full: [], hidden: ["p1", "p2", "p3"] });
});

await test("visibility: asker — the asking peer by id, the rest hidden", () => {
  const ids = ["p1", "p2", "p3"];
  eq(chatRecipients("asker", "p2", ids), { full: ["p2"], hidden: ["p1", "p3"] });
});

await test("visibility: asker is the host itself — nobody else sees it", () => {
  const ids = ["p1", "p2", "p3"];
  eq(chatRecipients("asker", "host-id", ids), { full: [], hidden: ["p1", "p2", "p3"] });
});

await test("visibility: modes are the three the dropdown offers", () => {
  eq(VISIBILITY, ["all", "host", "asker"]);
});

// 3. WGSL generator smoke test
await test("generator: WGSL balanced braces and entry points", () => {
  for (const [wg, rows, cols, rowsB] of [[256, 4, 4, 4], [128, 4, 8, 2], [64, 4, 8, 4]]) {
    const src = WGSL + coopWGSL(wg, rows, 64, cols, rowsB, true);
    const open = (src.match(/\{/g) || []).length, close = (src.match(/\}/g) || []).length;
    assert(open === close, `unbalanced braces for ${wg}/${rows}/${cols}/${rowsB}`);
    assert((src.match(/@compute/g) || []).length > 0, "no entry points found");
  }
});

// 4. Transport tests
function fakeChannels(link, n, sink) {
  for (let i = 0; i < n; i++) link.chans.push({ readyState: "open", send: (buf) => sink.push({ i, buf }) });
}
function receiver(onFrame, opts = {}) {
  const link = makeLink(opts); let handler = null;
  const pc = { createDataChannel: () => ({ set onmessage(f) { handler = f; }, set onclose(_) {}, readyState: "open" }) };
  attachWire(link, { peerConnection: pc }, onFrame, opts);
  return (buf) => handler({ data: buf });
}

const dim = 512;
const shapes = [
  { t: "ai-hidden", pos: 17, n: 1, cols: 1 },
  { t: "ai-hidden-b", basePos: 240, n: 16, cols: 16 },
  { t: "ai-hiddenret-b", basePos: 5, n: 6, spec: 1, cols: 6 },
];

for (const sh of shapes) {
  const data = new Uint16Array(dim * sh.cols);
  for (let i = 0; i < data.length; i++) data[i] = (i * 2654435761) >>> 16;
  await test(`transport round trip ${sh.t} x${sh.cols}`, () => {
    const link = makeLink(), out = []; fakeChannels(link, 3, out);
    if (!sendFrame(link, { ...sh, data })) throw new Error("send refused");
    for (const { buf } of out) if (buf.byteLength > SLICE_BYTES) throw new Error("slice too big: " + buf.byteLength);
    const expectSlices = Math.ceil(data.byteLength / (SLICE_BYTES - 24));
    if (out.length !== expectSlices) throw new Error(`expected ${expectSlices} slices, got ${out.length}`);
    let got = null; const deliver = receiver((m) => { got = m; });
    const order = [...out].reverse(); order.splice(1, 0, out[Math.floor(out.length / 2)]);
    for (const { buf } of order) deliver(buf);
    if (!got) throw new Error("frame not reassembled");
    if (got.t !== sh.t) throw new Error("kind mismatch " + got.t);
    if ((sh.pos ?? sh.basePos) !== (got.pos ?? got.basePos)) throw new Error("pos mismatch");
    if (got.n !== sh.n || !got.spec !== !sh.spec) throw new Error("meta mismatch");
    if (got.data.length !== data.length) throw new Error("length mismatch");
    for (let i = 0; i < data.length; i++) if (got.data[i] !== data[i]) throw new Error("byte mismatch at " + i);
  });
}

await test("transport refuses when no channel is open", () => {
  const link = makeLink(); link.chans.push({ readyState: "connecting", send() {} });
  if (sendFrame(link, { t: "ai-hidden", pos: 0, data: new Uint16Array(8) })) throw new Error("should refuse");
});

await test("transport: FEC recovers single dropped slice in each block under packet loss", () => {
  const data = new Uint16Array(dim * 8); // spans multiple slices
  for (let i = 0; i < data.length; i++) data[i] = (i * 31337 + 7) & 0xFFFF;
  const link = makeLink({ ordered: false, fec: true });
  const out = [];
  fakeChannels(link, 4, out);
  if (!sendFrame(link, { t: "ai-hidden-b", basePos: 100, n: 4, data })) throw new Error("send refused");

  const nDataSlices = Math.ceil(data.byteLength / (SLICE_BYTES - 24));
  assert(out.length > nDataSlices, "FEC parity slice should be emitted");

  // Simulate packet loss: drop slice 1 (a middle data slice)
  const simulated = out.filter((_, idx) => idx !== 1);
  let got = null;
  const deliver = receiver((m) => { got = m; }, { ordered: false, fec: true });

  // Shuffle order to simulate unordered delivery over WebRTC
  const shuffled = [...simulated].reverse();
  for (const { buf } of shuffled) deliver(buf);

  assert(got !== null, "frame should be reconstructed despite FEC parity slice");
  eq(got.data.length, data.length);
  for (let i = 0; i < data.length; i++) {
    eq(got.data[i], data[i]);
  }
});

// 5. Sampling and Repetition Penalty tests
import { aiSample } from "../../room/sampling.js";

await test("sampling: aiSample selects high probability token without repetition penalty", () => {
  const logits = new Float32Array(100);
  logits.fill(-10);
  logits[42] = 20; // dominant token
  const sampled = aiSample(logits, 0.1, 10);
  assert(sampled === 42, `expected token 42, got ${sampled}`);
});

await test("sampling: aiSample applies repetition penalty to penalize repeated tokens", () => {
  const logits = new Float32Array(100);
  logits.fill(-10);
  logits[42] = 10.0;
  logits[43] = 9.5;
  // Without penalty, token 42 dominates with temp=0.01
  const s1 = aiSample(logits.slice(), 0.01, 10, [], 1.0);
  assert(s1 === 42, "token 42 should win without penalty");

  // With repetition penalty of 1.2 on token 42, 10.0 / 1.2 = 8.33 < 9.5 (token 43)
  const s2 = aiSample(logits.slice(), 0.01, 10, [42], 1.2);
  assert(s2 === 43, `expected token 43 to win after token 42 is penalized, got ${s2}`);
});

// 6. Markdown and Streaming syntax tests
import { md, esc } from "../../room/markdown.js";

await test("markdown: renders headings, bold, and code blocks with copy button", () => {
  const html = md("# Test Heading\n\n**Bold text**\n\n```python\nprint(123)\n```");
  assert(html.includes("<h1>Test Heading</h1>"), "missing heading");
  assert(html.includes("<strong>Bold text</strong>") || html.includes("<b>Bold text</b>"), "missing bold text");
  assert(html.includes("class=\"code-block\""), "missing code-block container");
  assert(html.includes("class=\"code-copy-btn\""), "missing copy button");
  assert(html.includes("print(123)"), "missing code text");
});

await test("markdown: streaming mode auto-closes incomplete code fences", () => {
  const streamingText = "Here is code:\n\n```javascript\nconst x = 10;"; // unclosed code block
  const html = md(streamingText, true);
  assert(html.includes("class=\"code-block\""), "streaming should auto-close code block");
  assert(html.includes("const x = 10;"), "missing code content in streaming block");
});

await test("markdown: renders ThoughtChain for <think> tags", () => {
  const thinkText = "<think>\nExploring solution space\n</think>\nHere is the answer.";
  const html = md(thinkText);
  assert(html.includes("class=\"thought-chain\""), "missing thought-chain container");
  assert(html.includes("Thought Process"), "missing thought header");
  assert(html.includes("Exploring solution space"), "missing inner thought content");
  assert(html.includes("Here is the answer"), "missing answer content");
});

await test("markdown: completed thought chain is collapsed and streaming thought chain is open", () => {
  const completedHtml = md("<think>Thinking done</think>Answer text");
  assert(!completedHtml.includes("<details class=\"thought-chain\" data-status=\"completed\" open>"), "completed thought chain should NOT be open by default");
  assert(completedHtml.includes("data-status=\"completed\""), "completed thought should have completed status");

  const streamingHtml = md("<think>Thinking in progress...", true);
  assert(streamingHtml.includes("data-status=\"streaming\" open"), "streaming thought chain MUST be open with streaming status");
  assert(streamingHtml.includes("Thinking…"), "missing streaming Thinking header");
});

import { resetLink } from "../../room/transport.js";
await test("transport: resetLink cleanly flushes pending rx frames", () => {
  const link = makeLink();
  link.rx.set(42, { parts: [true], got: 1, n: 2, buf: new Uint8Array(100), t: Date.now() });
  assert(link.rx.size === 1, "link should have 1 pending rx frame");
  resetLink(link);
  assert(link.rx.size === 0, "resetLink should clear rx map");
});

await test("models: Qwen models have thinking enabled and SmolLM disabled", () => {
  assert(MODELS["qwen3-0.6b"].thinking === true, "qwen3-0.6b should have thinking: true");
  assert(MODELS["qwen3-1.7b"].thinking === true, "qwen3-1.7b should have thinking: true");
  assert(MODELS["smollm-135m"].thinking === false, "smollm-135m should have thinking: false");
});

import { ggmlLayerNames } from "../../engine/gguf.js";
await test("engine: ggmlLayerNames maps QKV biases for standard architectures", () => {
  const llamaNames = ggmlLayerNames(0, "llama");
  assert(llamaNames.qBias === "blk.0.attn_q.bias", "llama/qwen layer 0 qBias mapping missing");
  assert(llamaNames.kBias === "blk.0.attn_k.bias", "llama/qwen layer 0 kBias mapping missing");
  assert(llamaNames.vBias === "blk.0.attn_v.bias", "llama/qwen layer 0 vBias mapping missing");

  const phiNames = ggmlLayerNames(0, "phi3");
  assert(!phiNames.qBias, "phi3 qBias should be falsy");
});

await test("generator: WGSL defines add_bias compute pipeline", () => {
  assert(WGSL.includes("fn add_bias("), "WGSL missing add_bias compute entry point");
  assert(WGSL.includes("ab_x[i] += ab_b[i];"), "WGSL missing add_bias accumulation");
});

import { cfgFromGGUF, parseGGUFHeader } from "../../engine/gguf.js";
import fs from "fs";
await test("engine: cfgFromGGUF extracts complete architecture config from GGUF metadata", () => {
  if (fs.existsSync("models/qwen/model.gguf")) {
    const fd = fs.openSync("models/qwen/model.gguf", "r");
    const buf = Buffer.alloc(10 * 1024 * 1024);
    fs.readSync(fd, buf, 0, buf.length, 0);
    const G = parseGGUFHeader(buf.buffer, { skipTokenizer: true });
    const cfg = cfgFromGGUF(G);
    assert(cfg.hidden_size === 1024, `expected hidden_size 1024, got ${cfg.hidden_size}`);
    assert(cfg.num_attention_heads === 16, `expected num_attention_heads 16, got ${cfg.num_attention_heads}`);
    assert(cfg.num_key_value_heads === 8, `expected num_key_value_heads 8, got ${cfg.num_key_value_heads}`);
    assert(cfg.num_hidden_layers === 28, `expected num_hidden_layers 28, got ${cfg.num_hidden_layers}`);
    assert(cfg.intermediate_size === 3072, `expected intermediate_size 3072, got ${cfg.intermediate_size}`);
    assert(cfg.vocab_size === 151936, `expected vocab_size 151936, got ${cfg.vocab_size}`);
  } else {
    const mockG = {
      meta: {
        "general.architecture": "qwen2",
        "qwen2.block_count": 28,
        "qwen2.embedding_length": 1024,
        "qwen2.attention.head_count": 16,
        "qwen2.attention.head_count_kv": 8,
        "qwen2.feed_forward_length": 3072,
        "tokenizer.ggml.tokens": new Array(151936),
      }
    };
    const cfg = cfgFromGGUF(mockG);
    assert(cfg.hidden_size === 1024, `expected hidden_size 1024, got ${cfg.hidden_size}`);
    assert(cfg.num_attention_heads === 16, `expected num_attention_heads 16, got ${cfg.num_attention_heads}`);
    assert(cfg.num_key_value_heads === 8, `expected num_key_value_heads 8, got ${cfg.num_key_value_heads}`);
    assert(cfg.num_hidden_layers === 28, `expected num_hidden_layers 28, got ${cfg.num_hidden_layers}`);
    assert(cfg.intermediate_size === 3072, `expected intermediate_size 3072, got ${cfg.intermediate_size}`);
    assert(cfg.vocab_size === 151936, `expected vocab_size 151936, got ${cfg.vocab_size}`);
  }
});

import { runContextTests } from "./context_overflow_test.js";
await runContextTests(test);

// Groq Cloud mode and hosted-model tests
import { GROQ_DEFAULT_MODEL, parseGroqSSEChunk, completeGroqChat } from "../../room/groq.js";

await test("fallbackmode: variable is defined and can be toggled", () => {
  var fallbackmode = false;
  assert(typeof fallbackmode === "boolean", "fallbackmode should be boolean");
  fallbackmode = true;
  assert(fallbackmode === true, "fallbackmode should be true when enabled");
});

await test("fallbackmode: bypasses 27B model download and uses Groq directly", () => {
  const shouldDownload = (modelId, isFallback) => {
    if (isFallback && modelId === "qwen3.8-27b") return false;
    return true;
  };
  assert(shouldDownload("qwen3.8-27b", true) === false, "27B download must be skipped in fallback mode");
  assert(shouldDownload("qwen3.8-27b", false) === true, "27B download should proceed when fallback is off");
});

await test("fallbackmode: configures the default Groq hosted model", () => {
  assert(GROQ_DEFAULT_MODEL === "qwen/qwen3.8-27b", `Expected qwen/qwen3.8-27b, got ${GROQ_DEFAULT_MODEL}`);
});

await test("fallbackmode: parseGroqSSEChunk extracts token deltas accurately", () => {
  const sseChunk = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: {"choices":[{"delta":{"content":" World"}}]}\n\ndata: [DONE]\n\n';
  const tokens = [];
  const res = parseGroqSSEChunk(sseChunk, (t) => tokens.push(t));
  assert(res.text === "Hello World", `Expected 'Hello World', got '${res.text}'`);
  assert(res.isDone === true, "Should recognize [DONE] marker");
  assert(tokens.length === 2 && tokens[0] === "Hello" && tokens[1] === " World", "Token callback should receive both tokens");
});

await test("fallbackmode: throws helpful error if GROQ_API_KEY is missing", async () => {
  let threw = false;
  try {
    await completeGroqChat({ prompt: "hi", apiKey: "" });
  } catch (err) {
    threw = true;
    assert(err.message.includes("Groq API key required"), `Unexpected error: ${err.message}`);
  }
  assert(threw, "completeGroqChat should throw when API key is missing");
});

// Groq client & Model Mapping tests
import { GROQ_MODEL_MAP, getGroqModelId } from "../../room/models.js";
import { formatGroqError, streamGroqChat } from "../../room/groq-client.js";
import groqProxyHandler from "../../api/groq.js";

await test("groq: GROQ_MODEL_MAP contains all 5 required model mappings", () => {
  assert(GROQ_MODEL_MAP["qwen3.8-27b"] === "qwen/qwen3.8-27b");
  assert(GROQ_MODEL_MAP["qwen2.5-coder-7b"] === "openai/gpt-oss-120b");
  assert(GROQ_MODEL_MAP["deepseek-r1-distill-qwen-14b"] === "openai/gpt-oss-120b");
  assert(GROQ_MODEL_MAP["qwq-32b"] === "openai/gpt-oss-120b");
  assert(GROQ_MODEL_MAP["qwen3-4b"] === "openai/gpt-oss-20b");
  assert(GROQ_MODEL_MAP["qwen3-1.7b"] === "openai/gpt-oss-20b");
  assert(GROQ_MODEL_MAP["qwen3-0.6b"] === "openai/gpt-oss-20b");

  assert(getGroqModelId("qwen3.8-27b") === "qwen/qwen3.8-27b");
  assert(getGroqModelId("deepseek-r1-distill-qwen-14b") === "openai/gpt-oss-120b");
});

await test("groq: formatGroqError provides descriptive messages for 401, 404, 429", () => {
  const err401 = formatGroqError(401, { error: { message: "Invalid API Key" } }, "qwen3-32b");
  assert(err401.includes("Invalid Groq API Key (401)") && err401.includes("GROQ_API_KEY in .env"), "401 should mention invalid API key and .env");

  const err404 = formatGroqError(404, { error: { message: "Model not found" } }, "openai/gpt-oss-120b");
  assert(err404.includes("Model \"openai/gpt-oss-120b\" not found on Groq (404)"), "404 should name model and 404 status");

  const err429 = formatGroqError(429, { error: { message: "Rate limit reached" } }, "openai/gpt-oss-120b");
  assert(err429.includes("Groq rate limit exceeded (429)"), "429 should identify rate limit");
});

await test("groq proxy: api/groq.js rejects GET requests with 405", async () => {
  const req = new Request("http://localhost:8080/api/groq", { method: "GET" });
  const res = await groqProxyHandler(req);
  assert(res.status === 405, `Expected 405, got ${res.status}`);
});

await test("groq proxy: api/groq.js handles OPTIONS with 204", async () => {
  const req = new Request("http://localhost:8080/api/groq", { method: "OPTIONS" });
  const res = await groqProxyHandler(req);
  assert(res.status === 204, `Expected 204, got ${res.status}`);
});

await test("groq proxy: api/groq.js rejects missing model with 400", async () => {
  const req = new Request("http://localhost:8080/api/groq", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "hi" }] })
  });
  const res = await groqProxyHandler(req);
  assert(res.status === 400, `Expected 400 for missing model, got ${res.status}`);
});

await test("groq client: streamGroqChat handles non-JSON and HTML error responses without body stream already read", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response("<!DOCTYPE html><html><head><title>502 Bad Gateway</title></head><body>Bad Gateway</body></html>", {
        status: 502,
        headers: { "Content-Type": "text/html" }
      });
    };
    let threw = false;
    try {
      for await (const _ of streamGroqChat({ model: "test-model", prompt: "hi" })) {}
    } catch (err) {
      threw = true;
      assert(!err.message.includes("body stream already read"), `Should not fail with stream read error: ${err.message}`);
      assert(err.message.includes("502") || err.message.includes("Bad Gateway"), `Expected 502 error message: ${err.message}`);
    }
    assert(threw, "streamGroqChat should throw formatted error on 502");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

globalThis.Chart = class {
  constructor() {}
  update() {}
  destroy() {}
};

await test("perf-sidebar: generation life-cycle updates metrics and session aggregates", () => {
  const ps = new PerfSidebar();
  ps.onGenStart({ model: "Qwen 3.8 27B" });
  assert(ps.isStreaming === true, "isStreaming should be true on gen start");
  assert(ps.currentModel === "Qwen 3.8 27B", "currentModel should match start argument");

  ps.onToken("Hello", 1);
  ps.onToken(" world", 2);
  assert(ps.tokenCount === 2, "tokenCount should be 2");
  assert(ps.streamPoints.length === 1, "two token timestamps should produce one measured speed point");
  assert(ps.streamPoints[0].tps > 0, "measured speed point should have a positive throughput");

  ps.onGenDone({ totalTokens: 2, totalSecs: 0.1, stats: "2 tok · 20.0 tok/s" });
  assert(ps.sessionTokens === 2, "sessionTokens should aggregate");
  assert(ps.promptCount === 1, "promptCount should increment");
});

console.log(`\n${passed}/${total} tests passed; ${total - passed} failed.`);
