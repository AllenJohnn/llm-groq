# LLM ShardX Project Progress & Execution Plan

**Last Updated:** 2026-09-24  
**Target Repository:** https://github.com/AllenJohnn/llm  

---

## 1. Project Mission & Architecture Overview
**LLM ShardX** is a decentralized, browser-native LLM inference engine. Multiple client devices (laptops, phones, desktops) join a shared WebRTC room, divide the transformer layers according to their available WebGPU memory (e.g. Host gets layers 0–13 + embeddings/head, Worker gets layers 14–27), and stream activations in a pipeline to execute inference cooperatively.

---

## 2. Complete Chronological Progress (What Has Been Accomplished)

### Milestone 1: Routing & Local Server Setup
- Created [`serve.json`](file:///D:/New%20folder/webslice/serve.json) configuring clean rewrites from `/room` to `/room/index.html`.
- Created [`room/index.html`](file:///D:/New%20folder/webslice/room/index.html) as a standalone entrypoint matching [`p2p.html`](file:///D:/New%20folder/webslice/p2p.html).

### Milestone 2: UI Button Interactivity & Diagnostic Loader
- Added direct inline event handlers `window.swarmStart(create)` and `window.stepGB(delta)`.
- Created a pre-flight diagnostic module loader that verifies network access, checks MIME types, and imports submodules cleanly with visible error banners if any script fails.

### Milestone 3: Ad-Blocker False Positive Fix
- Renamed `engine/quant.js` to [`engine/quantize.js`](file:///D:/New%20folder/webslice/engine/quantize.js) to prevent Brave Shields / EasyPrivacy from blocking it as Quantcast tracker (`/quant.js`).
- Updated all import statements across [`engine/engine.js`](file:///D:/New%20folder/webslice/engine/engine.js) and [`engine/selftest.js`](file:///D:/New%20folder/webslice/engine/selftest.js).

### Milestone 4: Fix `pipeline timeout (batch prefill)`
- Switched default `WIRE` to `"slice"` instead of `stripe4` to avoid uncoordinated channel packet drops.
- Added active peer pruning in `conn.on("close")` and at the start of `aiGenerate()`.
- Wrapped worker activation chunks in bounds clamping and error reporting.
- Wrapped batched prefill in a 15s timeout with automatic graceful fallback to sequential token prefill via `aiPipeToken`.

### Milestone 5: WebRTC ICE & Relay Traversal Hardening
- Added redundant STUN servers (Google, Cloudflare, Twilio) and free OpenRelay TURN servers across UDP and TCP ports 80/443 in `ICE.iceServers`.
- Increased negotiation window to 35s with real-time state tracking on `conn.peerConnection.iceConnectionState`.
- Resolved `.open` race conditions on both host and joiner links.

### Milestone 6: Clean Project Identity & Single-Author Repository
- Rebranded author metadata, LICENSE copyright, AUTHORS, CITATION, package.json, and documentation links.
- Reset git commit graph into a clean single initial root commit authored exclusively by `AllenJohnn <allenjohnjoy2004@gmail.com>`.
- Pointed repository remote strictly to `https://github.com/AllenJohnn/llm.git`.

### Milestone 7: Syntax Error & Link Fixes
- Fixed missing closing brace in `room.js` line 1565 in `aiStart()` that previously triggered `Unexpected token 'catch'`.
- Verified and updated all frontend UI buttons and footers to point to `https://github.com/AllenJohnn/llm`.

### Milestone 8: High-Speed Mirror Routing for All 10 Models
- Audited all 10 models in [`room/models.js`](file:///D:/New%20folder/webslice/room/models.js).
- Replaced slow Hugging Face direct links with `https://hf-mirror.com/` as primary endpoints while preserving `huggingface.co` as fallback.
- Verified HTTP 206 Range responses and CORS across all endpoints.

### Milestone 9: Fixed Corrupted / Garbled Output (Qwen2.5 Bias & Tokenizer Alignment)
- **Problem:** Running Qwen2.5 Coder 1.5B produced garbled tokens (`30ESS mexicoA1 tempList60PS0 optargak helf.HowessPOess9_letter...`).
- **Root Cause 1:** Missing Q, K, and V attention projection biases (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`) in `DenseEngine`. Qwen2.5 has no QK RMSNorm and relies on additive biases with values up to $\pm 28.8$. Missing biases distorted attention dot products across 28 layers.
- **Root Cause 2:** Remote `tokenizer.json` had 151,665 tokens, while the GGUF model embeddings have 151,936 tokens.
- **Fixes Applied:**
  1. Added `@compute fn add_bias` WGSL compute shader kernel in [`engine/wgsl/base.js`](file:///D:/New%20folder/webslice/engine/wgsl/base.js#L240-L250).
  2. Mapped and loaded `qBias`, `kBias`, `vBias` in [`engine/gguf.js`](file:///D:/New%20folder/webslice/engine/gguf.js#L175-L287).
  3. Integrated `add_bias` dispatch in [`engine/dense.js`](file:///D:/New%20folder/webslice/engine/dense.js#L60-L465) for single-token decode and batched prefill passes.
  4. Built the authoritative tokenizer directly from `tokenizer.ggml.tokens` in [`room.js`](file:///D:/New%20folder/webslice/room.js#L1410-L1435).
  5. Verified with golden reference test [`tests/test_qwen.js`](file:///D:/New%20folder/webslice/tests/test_qwen.js) on WebGPU (QWEN Q8 PASS ✓, 0.096% diff, generated `" Paris. The capital of France is also"`).

### Milestone 10: Multi-Device Split & WebGPU Buffer Size Fix
- **Clarified Multi-Device Model Sharing:**
  - Only the host laptop needs the model `.gguf` in `./models/`.
  - When Device B connects to the host server, it streams only its assigned layer slice via HTTP 206 Range requests over local Wi-Fi without downloading the whole file.
- **Fixed `createBuffer` Size Error (`Value is not of type 'unsigned long long'`):**
  - **Root Cause:** When worker devices joined without a local `config.json`, remote `config.json` fetch failed (CORS/offline), falling back to `{ num_hidden_layers: L }` without `hidden_size`. This caused `dim * 4` to evaluate to `NaN` in `device.createBuffer({ size: NaN })`.
  - **Fixes Applied:**
    1. Implemented [`cfgFromGGUF(G)`](file:///D:/New%20folder/webslice/engine/gguf.js#L432-L465) to extract all architecture parameters (`embedding_length`, `head_count`, `head_count_kv`, `feed_forward_length`, `block_count`, etc.) directly from GGUF metadata, making inference completely self-contained and offline-capable.
    2. Host now bundles `cfg: ai.cfg` in the `ai-load` WebRTC message sent to workers so workers inherit the exact configuration without remote network fetches.
    3. Added defensive dimension fallbacks and `Math.max(4, ...)` guards on all buffer allocations in [`engine/dense.js`](file:///D:/New%20folder/webslice/engine/dense.js).

### Milestone 11: Comprehensive Test Suite Validation
- **Node Unit Tests:** All 35 tests in [`tests/unit/run_node_tests.mjs`](file:///D:/New%20folder/webslice/tests/unit/run_node_tests.mjs) pass.
- **Deno Unit Tests:** All 11 tests in [`tests/unit/models_test.js`](file:///D:/New%20folder/webslice/tests/unit/models_test.js) pass.
- **Topology Tests:** [`tests/test_topologies.js`](file:///D:/New%20folder/webslice/tests/test_topologies.js) passes for Solo, 2-Device Mesh, and Multi-Device Chain.

---

### Milestone 12: Zero-Config Local Signaling Server (`PeerServer`)
- Created [`scripts/signal-server.mjs`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/scripts/signal-server.mjs) running a local WebSocket PeerServer on port 9000.
- Added `npm run signal` script to `package.json`.
- Eliminates any external dependency on `0.peerjs.com`, enabling 100% offline local demonstrations.

### Milestone 13: Universal Model Downloader & Local Weight Verification
- Created [`scripts/download_model.mjs`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/scripts/download_model.mjs) supporting all 10 models in the LLM ShardX catalog.
- Features chunked streaming download with progress bar, download resumption (`Range` requests), high-speed mirror routing (`hf-mirror.com`), and automatic fallback to `huggingface.co`.
- Downloads model weights, `config.json`, and `tokenizer.json` into primary directories (`models/<dir>/model.gguf`) and root candidate paths (`models/<model-name>.gguf`).
- Downloaded and verified local offline models: `smollm-135m` (256.6 MB safetensors) and `qwen3-0.6b` (609.8 MB GGUF).

### Milestone 14: Single-System Split-Tab Demonstration Workflow
- Enhanced [`room.js`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/room.js), [`p2p.html`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/p2p.html), and [`room/index.html`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/room/index.html).
- Added `+ Split Demo Tab` button in host room screen: clicking it automatically opens a second browser tab with the room code and local signaling pre-filled.
- Updated `copyRoomLink()` to preserve `&signal=localhost:9000` query parameter.
- Enhanced signaling timeout alert to provide a 1-click fallback link to local PeerServer.

### Milestone 15: Unified Demonstration Suite (`npm run demo`)
- Created [`scripts/demo.mjs`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/scripts/demo.mjs) launching both the static server (port 8080) and local signaling server (port 9000) concurrently.
- Automatically detects local LAN IP to provide QR/URL access for mobile devices on the same Wi-Fi.
- Created comprehensive guide in [`docs/demo-guide.md`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/docs/demo-guide.md).

### Milestone 16: Fallback Mode with Groq API & Qwen 27B Model
- Created [`room/groq.js`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/room/groq.js) implementing streaming and non-streaming Groq API integration using model `qwen/qwen3.8-27b` (Qwen 27B).
- Added `var fallbackmode` in [`room.js`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/room.js) and [`tests/test_new_models.js`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/tests/test_new_models.js).
- When `fallbackmode` is enabled (via UI button, URL parameter `?fallbackmode=1`, `window.fallbackmode = true`, or `FALLBACKMODE=true` in Node), inference immediately routes to Groq API with the Qwen 27B model and streams tokens into the chat interface.
- Added Fallback Mode button (`#mode-fallback`) in [`p2p.html`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/p2p.html) and [`room/index.html`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/room/index.html).
- Added comprehensive unit tests in [`tests/unit/run_node_tests.mjs`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/tests/unit/run_node_tests.mjs) (all 39 tests passing) and [`tests/test_fallbackmode.js`](file:///c:/Users/NEEHA%20NAZER/Documents/LLM/llm/tests/test_fallbackmode.js).

---

## 3. Current Working State

- **Demonstration Suite:** `npm run demo` starts static web server + local PeerServer.
- **Active Servers:**
  - Static Web Server: `http://localhost:8080/room`
  - Local PeerServer: `ws://localhost:9000`
- **Models Downloaded & Ready for Offline Demo:**
  - `smollm-135m`: `models/model/model.safetensors` & `models/smollm-135m.safetensors` (256.6 MB).
  - `qwen3-0.6b`: `models/qwen/model.gguf` & `models/qwen3-0.6b.gguf` (609.8 MB).
  - `qwen2.5-coder-1.5b`: `models/qwen25coder15/model.gguf` & `models/qwen2.5-coder-1.5b-instruct-q4_0.gguf` (1016.8 MB).
  - Any of the 10 models can be downloaded on-demand with `npm run download <model-id>`.
- **Fallback Mode:** `fallbackmode` variable routes requests to Groq Cloud running `qwen/qwen3.8-27b` (Qwen 27B).
- **Test Suite:**
  - `npm run test:node`: All 39 tests pass.
  - `node tests/test_fallbackmode.js`: Fallback Mode validation passes.
  - `node tests/test_topologies.js`: All Solo, 2-Device Mesh, 3-Device Chain, and 16-Device Swarm tests pass.

---

## 4. How to Run the Demonstration

1. Run:
   ```bash
   npm run demo
   ```
2. **Single-System Demo (Split Screen):**
   - Tab 1 (Host): Open `http://localhost:8080/room?signal=localhost:9000`, click **Create room**.
   - Click **+ Split Demo Tab** to launch Tab 2 (Worker), then click **Join room**.
   - Select **SmolLM 135M** or **Qwen3 0.6B**, click **Start Swarm**, and send prompts!
3. **Multi-Device Demo (Wi-Fi):**
   - Open `http://<your-lan-ip>:8080/room?signal=<your-lan-ip>:9000` on your phone, join room, and run inference across phone + laptop.

---

## 5. How to Resume Later
To resume work seamlessly in a new session, simply prompt:
> **"Continue where we stopped by reading PLAN.md"**
