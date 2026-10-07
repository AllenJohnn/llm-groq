# LLM ShardX Multiplayer Setup Guide

This guide explains how to host a LLM ShardX session on one laptop and allow another laptop over the internet to connect, join your room, and share the AI processing load.

## Step 1: Start the LLM ShardX Host
1. Open a terminal (Command Prompt or PowerShell) on the **Host Laptop**.
2. Navigate to your LLM ShardX directory:
   ```bash
   cd "D:\New folder\llm-shardx"
   ```
3. Start the local server:
   ```bash
   npm run demo
   ```
   *(Keep this terminal window open!)*

## Step 2: Start the Secure Tunnel (ngrok)
To allow the second laptop to connect without dealing with Windows Firewall or router settings, we use an ngrok tunnel.
1. Open a **second** terminal window on the Host Laptop.
2. Navigate to the project directory again:
   ```bash
   cd "D:\New folder\llm-shardx"
   ```
3. Start the tunnel on port 8080:
   ```bash
   npx ngrok http 8080
   ```
4. Ngrok will start up and show a **Forwarding** URL that looks something like `https://some-random-words.ngrok-free.dev`. Copy this link!

## Step 3: Connect the Laptops
1. **Host Laptop**: 
   - Open your browser and go to your local dashboard: `http://localhost:8080/room`
   - Type in a name (like "Laptop 1"), click **Create Room**, and note the 4-letter **Room Code**.
2. **Worker Laptop (The second system)**:
   - Open a browser and go to the ngrok URL you copied in Step 2, making sure to add `/room` to the end.
   - Example: `https://some-random-words.ngrok-free.dev/room`
   - Type in a name (like "Laptop 2"), type in the **Room Code**, and click **Join Room**.

## Step 4: Start the Swarm
Once both laptops appear in the lobby on your screen:
1. On the **Host Laptop**, select your AI model (e.g., SmolLM 135M or Qwen3 0.6B).
2. Click **Start Swarm**.
3. The engine will automatically divide the model layers between the two laptops (e.g., a 50/50 split). Once loaded, you can type a prompt and watch them stream tokens collaboratively!

---

### Troubleshooting Tips:
- **"EADDRINUSE: address already in use 8080" Error**: You have a background node process that didn't close. Open Task Manager and end `node.exe` processes, then restart.
- **"ERR_NGROK_334"**: This means you already have an ngrok terminal open somewhere routing to your laptop. Find it and use that URL, or close it and run the command again.
