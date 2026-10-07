// Model catalogue for the room: URLs, layer counts, memory needs, context length.

export const NEED_GB = {
  "qwen3-0.6b": 0.8,
  "qwen3-1.7b": 2.0,
  "qwen3-4b": 4.6,
  "qwen2.5-coder-1.5b": 1.8,
  "qwen2.5-coder-7b": 5.0,
  "deepseek-r1-distill-qwen-14b": 9.5,
  "qwq-32b": 21.0,
  "qwen3.8-27b": 16.5,
  "phi-4-mini": 3.0,
  "smollm-135m": 0.3,
};

// Cloud choices are model mappings, not the local weights named in the picker.
// Keep the actual Groq model IDs visible in the UI and send those IDs unchanged.
export const GROQ_MODEL_MAP = {
  "qwen3.8-27b": "qwen/qwen3.8-27b",
  "qwen2.5-coder-7b": "openai/gpt-oss-120b",
  "qwen2.5-coder-1.5b": "openai/gpt-oss-20b",
  "deepseek-r1-distill-qwen-14b": "openai/gpt-oss-120b",
  "qwq-32b": "openai/gpt-oss-120b",
  "qwen3-4b": "openai/gpt-oss-20b",
  "qwen3-1.7b": "openai/gpt-oss-20b",
  "qwen3-0.6b": "openai/gpt-oss-20b",
  "phi-4-mini": "openai/gpt-oss-20b",
  "smollm-135m": "openai/gpt-oss-20b",
};

export function getGroqModelId(modelKey) {
  return GROQ_MODEL_MAP[modelKey] || "openai/gpt-oss-120b";
}

export const MODELS = {
  "qwen3-0.6b": { label: "Qwen3 0.6B · Q8", kind: "gguf", thinking: true,
    gguf: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    ggufFallback: "https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
  "qwen3-1.7b": { label: "Qwen3 1.7B · Q8", kind: "gguf", thinking: true,
    gguf: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    ggufFallback: "https://hf-mirror.com/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json" },
  "qwen3-4b": { label: "Qwen3 4B · Q8", kind: "gguf", thinking: true,
    gguf: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    ggufFallback: "https://hf-mirror.com/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json" },
  "qwen2.5-coder-1.5b": { label: "Qwen2.5 Coder 1.5B · Q4", kind: "gguf", thinking: false,
    gguf: "https://hf-mirror.com/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_0.gguf",
    ggufFallback: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/tokenizer.json" },
  "qwen2.5-coder-7b": { label: "Qwen2.5 Coder 7B · Q4 (Code & Web)", kind: "gguf", thinking: false,
    gguf: "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_0.gguf",
    ggufFallback: "https://hf-mirror.com/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/tokenizer.json" },
  "deepseek-r1-distill-qwen-14b": { label: "DeepSeek-R1 Distill Qwen 14B · Q4 (Reasoning)", kind: "gguf", thinking: true,
    gguf: "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf",
    ggufFallback: "https://hf-mirror.com/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf",
    cfg: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/config.json",
    tok: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/tokenizer.json" },
  "qwq-32b": { label: "Qwen QwQ 32B · Q4 (Deep Reasoning)", kind: "gguf", thinking: true,
    gguf: "https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF/resolve/main/Qwen_QwQ-32B-Q4_0.gguf",
    ggufFallback: "https://hf-mirror.com/bartowski/Qwen_QwQ-32B-GGUF/resolve/main/Qwen_QwQ-32B-Q4_0.gguf",
    cfg: "https://huggingface.co/Qwen/QwQ-32B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/QwQ-32B/resolve/main/tokenizer.json" },
  "phi-4-mini": { label: "Phi-4 mini · Q4", kind: "gguf", arch: "phi3", thinking: false,
    gguf: "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_0.gguf",
    ggufFallback: "https://hf-mirror.com/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_0.gguf",
    cfg: "https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/config.json" },
  "smollm-135m": { label: "SmolLM 135M · bf16", kind: "st", thinking: false,
    st: "https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/resolve/main/model.safetensors",
    cfg: "https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/resolve/main/tokenizer.json" },
  "qwen3.8-27b": { label: "Qwen 3.8 27B · Q4", kind: "qwen35", thinking: false,
    gguf: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf",
    ggufFallback: "https://hf-mirror.com/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf", cfg: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/main/config.json", tok: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/main/tokenizer.json" },
};

// Context window per room, in tokens: prompt + answer. Each full-attention layer keeps K and V
// for this many positions (4 KB per position each for the 27B, so 16 MiB per attention layer at
// 2048); the kernels only use it as a stride. Generation stops before the cache would overflow.
export const MAX_SEQ = 2048;
export const MAX_NEW = 2048;   // longest answer, tokens (utilizes full remaining context)
export const MIN_ROOM = 32;    // a prompt must leave at least this many tokens for the answer

// Local model candidates for bypassing downloads during testing.
// Place downloaded .gguf or .safetensors files in ./models/ to serve directly from local disk.
export const LOCAL_CANDIDATES = {
  "qwen3-0.6b": ["/models/qwen3-0.6b.gguf", "/models/Qwen3-0.6B-Q8_0.gguf"],
  "qwen3-1.7b": ["/models/qwen3-1.7b.gguf", "/models/Qwen3-1.7B-Q8_0.gguf", "/models/qwen17/model.gguf"],
  "qwen3-4b": ["/models/qwen3-4b.gguf", "/models/Qwen3-4B-Q8_0.gguf", "/models/qwen4/model.gguf"],
  "qwen2.5-coder-1.5b": ["/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf", "/models/Qwen2.5-Coder-1.5B-Instruct-Q4_0.gguf"],
  "qwen2.5-coder-7b": ["/models/qwen25coder/model.gguf", "/models/qwen2.5-coder-7b.gguf", "/models/Qwen2.5-Coder-7B-Instruct-Q4_0.gguf", "/models/qwen7b/model.gguf"],
  "deepseek-r1-distill-qwen-14b": ["/models/deepseek-r1-distill-qwen-14b.gguf", "/models/DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf", "/models/r1-14b/model.gguf"],
  "qwq-32b": ["/models/qwq-32b.gguf", "/models/Qwen_QwQ-32B-Q4_0.gguf", "/models/qwq32b/model.gguf"],
  "phi-4-mini": ["/models/phi4mini/model.gguf", "/models/microsoft_Phi-4-mini-instruct-Q4_0.gguf", "/models/Phi-4-mini-instruct-Q4_0.gguf", "/models/phi-4-mini-instruct-Q4_0.gguf", "/models/phi-4-mini.gguf"],
  "qwen3.8-27b": ["/models/qwen3.8-27b.gguf", "/models/Qwen3.8-27B-Q4_0.gguf", "/models/q38/model.gguf"],
  "smollm-135m": ["/models/smollm-135m.safetensors", "/models/model.safetensors", "/models/model/model.safetensors"],
};

export async function detectLocalModel(modelKey) {
  const m = MODELS[modelKey];
  if (!m) return null;
  // Preserve original remote URLs so fetch fallbacks always know the real endpoints
  if (!m.originalGguf && m.gguf) m.originalGguf = m.gguf;
  if (!m.originalSt && m.st) m.originalSt = m.st;

  const candidates = LOCAL_CANDIDATES[modelKey] || [];
  const expectedMinBytes = ((NEED_GB[modelKey] || 0.5) * 0.45) * 1024 * 1024 * 1024;
  for (const p of candidates) {
    try {
      const resp = await fetch(p, { method: "HEAD", headers: { "ngrok-skip-browser-warning": "1" } });
      if (resp.ok) {
        const cl = resp.headers.get("content-length");
        if (cl) {
          const size = parseInt(cl, 10);
          if (size < expectedMinBytes) {
            console.warn(`[webslice] Local model file ${p} is incomplete (${(size / 2 ** 30).toFixed(2)} GB < ${(expectedMinBytes / 2 ** 30).toFixed(2)} GB). Skipping.`);
            continue;
          }
        }
        console.log(`[webslice] Using local model file: ${p}`);
        if (m.gguf) m.gguf = p;
        if (m.st) m.st = p;
        const dir = p.substring(0, p.lastIndexOf("/") + 1);
        if (dir !== "/" && dir !== "/models/") {
          try {
            const cfgResp = await fetch(dir + "config.json", { method: "HEAD" });
            if (cfgResp.ok) m.cfg = dir + "config.json";
          } catch {}
          try {
            const tokResp = await fetch(dir + "tokenizer.json", { method: "HEAD" });
            if (tokResp.ok) m.tok = dir + "tokenizer.json";
          } catch {}
        }
        return p;
      }
    } catch {}
  }
  return null;
}

