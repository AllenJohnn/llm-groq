# LLM ShardX Demonstration & System Setup Guide

This guide provides end-to-end instructions for running and demonstrating **LLM ShardX** on a **single system** (via split browser tabs) or across **multiple devices** (laptop + phone/tablet over local Wi-Fi) with zero cloud dependency.

---

## 1. Quick Start Demonstration (Single Command)

To launch the complete demonstration environment:

```bash
npm run demo
```

This starts:
1. **Static Web & Weight Server:** `http://localhost:8080` (serves the WebGPU engine and local models with HTTP 206 range streaming).
2. **Local PeerServer Signaling Server:** `ws://localhost:9000` (provides high-speed, local WebRTC signaling without external cloud servers).

---

## 2. Model Management (Download & Test)

LLM ShardX supports 10 distinct models. Any model can be downloaded locally:

### View Model Catalog & Download Status
```bash
npm run models
# or: node scripts/download_model.mjs list
```

### Download Models
- **Fastest Demo Models (SmolLM2 135M & Qwen3 0.6B):**
  ```bash
  node scripts/download_model.mjs demo
  ```
- **Download a Specific Model:**
  ```bash
  npm run download smollm-135m
  npm run download qwen3-0.6b
  npm run download qwen2.5-coder-1.5b
  npm run download qwen2.5-coder-7b
  npm run download phi-4-mini
  npm run download deepseek-r1-distill-qwen-14b
  npm run download qwq-32b
  npm run download qwen3.8-27b
  ```
- **Download All Models:**
  ```bash
  npm run download all
  ```

Downloads automatically use high-speed mirrors (`hf-mirror.com`), support download resumption (`Range` headers), download accompanying `config.json` and `tokenizer.json`, and place weights into both engine test paths and candidate paths for instant recognition.

---

## 3. Demonstration Mode A: Single-System (Split Screen)

You can demonstrate distributed pipeline parallel inference on a single laptop/desktop by running two browser tabs:

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│        TAB 1: HOST (L0 - L13)         │       TAB 2: WORKER (L14 - L27)       │
│                                       │                                       │
│  [Room: ABCD]  [+ Split Demo Tab]     │  [Room: ABCD]                         │
│  Pledge: 1 GB GPU                     │  Pledge: 1 GB GPU                     │
│  Model: Qwen3 0.6B                    │                                       │
│                                       │                                       │
│  Prompt: "Write a poem about space."  │  [Status: Processing Layers 14-27]    │
│  > Token generation streaming...      │  > WebRTC activations streaming...    │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

### Step-by-Step Instructions:
1. Run `npm run demo` in your terminal.
2. Open **Tab 1 (Host):**
   - Navigate to `http://localhost:8080/room?signal=localhost:9000`
   - Enter name (e.g. `Host Laptop`).
   - Click **Create room**.
   - Note the 4-letter room code (e.g. `ABCD`).
3. Click the **+ Split Demo Tab** button in the left sidebar:
   - A second tab opens automatically with the room code and local signaling pre-filled!
4. In **Tab 2 (Worker):**
   - Click **Join room**.
   - Tab 1 and Tab 2 instantly recognize each other and appear in the device list.
5. In **Tab 1 (Host):**
   - Select **SmolLM 135M** or **Qwen3 0.6B** from the model dropdown.
   - Click **Start Swarm**.
   - Notice the layer distribution: each tab downloads/streams only its assigned slice!
6. Type a prompt (e.g., *"Explain quantum computing in one sentence."*) and click Send.
7. **Demonstration Highlight:** Point out that Tab 1 processes the early layers, passes the hidden state activation vector via WebRTC to Tab 2, Tab 2 processes the remaining layers, and streams the output back to the host!

---

## 4. Demonstration Mode B: Multi-Device (Laptop + Mobile Phone)

Demonstrate cross-device inference where an iPhone, Android phone, or secondary laptop on the same Wi-Fi participates in inference:

1. In the `npm run demo` terminal, copy the **Mobile / LAN PEER URL**:
   ```
   http://<YOUR-LAN-IP>:8080/room?signal=<YOUR-LAN-IP>:9000
   ```
2. Open this URL on your phone's browser (Safari or Chrome).
3. On your laptop, create a room (e.g. `XYZW`).
4. On your phone, enter `XYZW` and tap **Join room**.
5. The phone's GPU appears in the cluster overview on the laptop!
6. Start the swarm: the phone runs a slice of layers on its mobile GPU while the laptop runs the rest.

---

## 5. Verification & Test Suite

Verify that all algorithms, mappings, and topologies pass:

```bash
# Run all unit tests (35/35 passing)
npm run test:node

# Run multi-device topology planner tests (Solo, 2-device mesh, 3-device chain, 16-device swarm)
node tests/test_topologies.js
```
