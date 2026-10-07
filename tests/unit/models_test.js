import { MODELS, NEED_GB, LOCAL_CANDIDATES, detectLocalModel } from "../../room/models.js";
import { pledgeOf, calculateClusterPledge, formatLayerRange, allocateLayers } from "../../room/allocation.js";

const assert = (cond, msg) => { if (!cond) throw new Error(msg || "assertion failed"); };

Deno.test("models: catalog has all 11 expected models", () => {
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

Deno.test("models: Qwen3 models enable thinking-mode prompt handling", () => {
  for (const key of ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"]) {
    assert(MODELS[key].thinking === true, `${key} should have thinking enabled`);
  }
  assert(MODELS["smollm-135m"].thinking === false, "SmolLM should not have thinking enabled");
});

Deno.test("models: Qwen2.5 Coder 1.5B uses the fast mirror and has an HF fallback", () => {
  const m = MODELS["qwen2.5-coder-1.5b"];
  assert(m.gguf.startsWith("https://hf-mirror.com/"), "Qwen2.5 Coder should prefer the fast mirror");
  assert(m.ggufFallback.startsWith("https://huggingface.co/"), "Qwen2.5 Coder fallback URL missing");
  assert(NEED_GB["qwen2.5-coder-1.5b"] >= 1.5, "Qwen2.5 Coder memory pledge missing");
});

Deno.test("models: added models have valid GGUF URLs and layer configs", () => {
  const m7b = MODELS["qwen2.5-coder-7b"];
  assert(m7b && m7b.gguf.includes("Qwen2.5-Coder-7B-Instruct-Q4_0.gguf"), "7B Coder URL missing");
  const m14b = MODELS["deepseek-r1-distill-qwen-14b"];
  assert(m14b && m14b.gguf.includes("DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf") && m14b.thinking, "14B R1 URL or thinking missing");
  const m32b = MODELS["qwq-32b"];
  assert(m32b && m32b.gguf.includes("Qwen_QwQ-32B-Q4_0.gguf") && m32b.thinking, "32B QwQ URL or thinking missing");
});

Deno.test("models: Phi-4 mini uses Phi3 GGUF path and fits a multi-GB room", () => {
  assert(MODELS["phi-4-mini"].arch === "phi3", "Phi-4 mini must use the Phi3 GGUF mapping");
  assert(MODELS["phi-4-mini"].gguf.endsWith("/microsoft_Phi-4-mini-instruct-Q4_0.gguf"), "Phi-4 mini Q4 GGUF URL missing");
  assert(NEED_GB["phi-4-mini"] >= 3, "Phi-4 mini pledge should include weights and runtime memory");
});

Deno.test("models: local candidates defined for all models", () => {
  for (const k of Object.keys(MODELS)) {
    assert(Array.isArray(LOCAL_CANDIDATES[k]), `LOCAL_CANDIDATES missing array for: ${k}`);
    assert(LOCAL_CANDIDATES[k].length > 0, `LOCAL_CANDIDATES empty for: ${k}`);
  }
});

Deno.test("models: detectLocalModel gracefully handles unreachable local files", async () => {
  const res = await detectLocalModel("non-existent-model");
  assert(res === null, "should return null for non-existent model");
});

Deno.test("models: detectLocalModel rejects truncated files and preserves original remote URLs", async () => {
  const origFetch = globalThis.fetch;
  const modelKey = "qwen2.5-coder-7b";
  const originalUrl = MODELS[modelKey].gguf;

  try {
    // Simulate candidate URL returning truncated content-length (e.g. 50 MB instead of ~4.5 GB)
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

Deno.test("models: detectLocalModel accepts complete local files and records original URL", async () => {
  const origFetch = globalThis.fetch;
  const modelKey = "qwen3-0.6b";
  const remoteUrl = MODELS[modelKey].gguf;

  try {
    // Simulate candidate URL returning complete content-length (800 MB >= required ~400 MB)
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
    // restore original URL
    MODELS[modelKey].gguf = remoteUrl;
  }
});

Deno.test("models: shared /models directory never overrides the model tokenizer", async () => {
  const origFetch = globalThis.fetch;
  const modelKey = "qwen3-0.6b";
  const model = MODELS[modelKey];
  const originalGguf = model.gguf;
  const originalTok = model.tok;
  const requested = [];

  try {
    globalThis.fetch = async (url) => {
      requested.push(String(url));
      return { ok: true, headers: new Headers({ "content-length": "838860800" }) };
    };

    const localPath = await detectLocalModel(modelKey);
    assert(localPath === "/models/qwen3-0.6b.gguf", "expected shared-directory local model candidate");
    assert(model.tok === originalTok, "shared tokenizer must not replace this model's tokenizer URL");
    assert(!requested.includes("/models/tokenizer.json"), "loader must not probe a shared tokenizer.json");
  } finally {
    globalThis.fetch = origFetch;
    model.gguf = originalGguf;
  }
});

Deno.test("allocation: formatLayerRange prevents negative ranges and handles edge cases", () => {
  assert(formatLayerRange([0, 14]) === "layers 0–13", "normal multi-layer formatting");
  assert(formatLayerRange([0, 1]) === "layer 0", "single layer formatting");
  assert(formatLayerRange([14, 28]) === "layers 14–27", "peer range formatting");
  // Host edge cases: lo >= hi
  assert(formatLayerRange([0, 0], true) === "embed/head only", "host zero layers should show embed/head only");
  assert(formatLayerRange([0, -1], true) === "embed/head only", "host negative range should not format as 0–-1");
  // Worker edge cases: lo >= hi
  assert(formatLayerRange([0, 0], false) === "0 layers", "worker zero layers should show 0 layers");
  assert(formatLayerRange([5, 4], false) === "0 layers", "worker inverted range should show 0 layers");
  // Invalid inputs
  assert(formatLayerRange(null) === "", "null range should return empty string");
  assert(formatLayerRange([]) === "", "empty range should return empty string");
});

Deno.test("allocation: allocateLayers prevents 0 layers when L >= peers even with low host memory", () => {
  const L = 28;
  const layerBytes = 150 * 1024 * 1024; // 150 MB
  const embedBytes = 600 * 1024 * 1024; // 600 MB
  // Host pledges 0.5 GB, workers pledge 8 GB each
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

  // Verify contiguous ranges
  assert(ranges[0][0] === 0, "first range must start at layer 0");
  assert(ranges[ranges.length - 1][1] === L, `last range must end at layer ${L}`);
  for (let i = 1; i < ranges.length; i++) {
    assert(ranges[i][0] === ranges[i - 1][1], `range gap or overlap at index ${i}`);
  }
});

Deno.test("allocation: pledgeOf and calculateClusterPledge exclude non-WebGPU devices", () => {
  const webgpuPeer = { contribGB: 4, webgpu: true };
  const cpuPeer = { contribGB: 8, webgpu: false };
  const host = { contribGB: 2, webgpu: true };

  assert(pledgeOf(webgpuPeer) === 4 * (2 ** 30), "WebGPU peer should pledge memory");
  assert(pledgeOf(cpuPeer) === 0, "non-WebGPU peer should have 0 pledge");
  assert(pledgeOf(null) === 0, "null peer should have 0 pledge");

  const clusterPledged = calculateClusterPledge(host, [webgpuPeer, cpuPeer]);
  assert(clusterPledged === 6, `Cluster pledged should be 2 + 4 = 6 GB, got ${clusterPledged} GB`);
});

