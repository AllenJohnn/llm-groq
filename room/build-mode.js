const STORAGE_KEY = "shardx-build-project-v1";
const DEFAULT_FILES = { "index.html": "<!-- Ask Build to create a small web app, or start editing here. -->\n<h1>Your preview starts here</h1>\n<p>Describe an app below to build it with the room model.</p>" };

const esc = (value) => String(value ?? "").replace(/[&<>\"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[ch]));

function parseFiles(text) {
  const files = {};
  const pattern = /```(?:html|css|javascript|js|json)?\s*(?:file\s*[:=]\s*["']?([^\s"'`]+)["']?)?\s*\n([\s\S]*?)```/gi;
  for (const match of text.matchAll(pattern)) {
    let name = match[1];
    const body = match[2].trim();
    if (!name) {
      const header = body.match(/^(?:<!--\s*FILE:\s*([^\s]+)\s*-->|\/\/\s*FILE:\s*([^\s]+))\s*\n/i);
      if (header) {
        name = header[1] || header[2];
        files[name] = body.slice(header[0].length);
        continue;
      }
      if (/^\s*<!doctype html|^\s*<html|^\s*<(?:main|body|div|h1|section|article)\b/i.test(body)) name = "index.html";
      else if (/<\/style>|\{[^}]*:[^}]*\}/.test(body) && !/function\s|=>|document\./.test(body)) name = "style.css";
      else if (/function\s|=>|document\.|addEventListener/.test(body)) name = "app.js";
    }
    if (name && ["index.html", "style.css", "app.js"].includes(name)) files[name] = body;
  }
  if (!files["index.html"]) {
    const first = text.match(/```(?:html)?\s*\n([\s\S]*?)```/i);
    const html = (first?.[1] || text).trim();
    if (/<(?:!doctype|html|main|body|div|h1|section)\b/i.test(html)) files["index.html"] = html;
  }
  return Object.keys(files).length ? files : null;
}

function previewDocument(files) {
  const html = files["index.html"] || "<main></main>";
  const css = files["style.css"] || "";
  const js = files["app.js"] || "";
  const telemetry = `<script>(function(){const send=(kind,args)=>parent.postMessage({source:'shardx-build-preview',kind,args:Array.from(args||[]).map(x=>{try{return typeof x==='string'?x:JSON.stringify(x)}catch{return String(x)}})},'*');for(const k of ['log','warn','error']){const original=console[k];console[k]=function(){send(k,arguments);return original.apply(this,arguments)}}addEventListener('error',e=>send('error',[e.message+' at '+e.filename+':'+e.lineno]));addEventListener('unhandledrejection',e=>send('error',[e.reason?.message||e.reason||'Unhandled promise rejection']))})()</script>`;
  const style = `<style>${css.replace(/<\/style/gi, "<\\/style")}</style>`;
  const script = `<script>${js.replace(/<\/script/gi, "<\\/script")}</script>`;
  let document = html;
  if (!/<html[\s>]/i.test(document)) document = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${document}</body></html>`;
  const csp = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: blob:; media-src data: blob:; style-src 'unsafe-inline' data:; script-src 'unsafe-inline' blob:; font-src data:; connect-src 'none'; frame-src 'none'; form-action 'none';">`;
  if (/<head[\s>]/i.test(document)) document = document.replace(/<head([^>]*)>/i, `<head$1>${csp}${telemetry}${style}`);
  else document = document.replace(/<html([^>]*)>/i, `<html$1><head>${csp}${telemetry}${style}</head>`);
  if (/<\/body>/i.test(document)) document = document.replace(/<\/body>/i, `${script}</body>`);
  else document = document.replace(/<\/html>/i, `${script}</html>`);
  return document;
}

class BuildMode {
  constructor() {
    this.files = this.load();
    this.activeFile = "index.html";
    this.revision = 0;
    this.previousFiles = null;
    this.generating = false;
    this.repairing = false;
    this.allowEdits = false;
    this.lastPrompt = "";
    this.renderQueued = false;
  }

  load() {
    try {
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (stored?.files && typeof stored.files["index.html"] === "string") return stored.files;
    } catch {}
    return { ...DEFAULT_FILES };
  }

  init() {
    if (document.getElementById("build-pane")) return;
    const pane = document.createElement("section");
    pane.id = "build-pane";
    pane.setAttribute("aria-label", "Build workspace");
    pane.innerHTML = `<div class="build-topline"><div><div class="build-eyebrow">ROOM WORKSPACE <span>·</span> BUILD</div><h1>Build a small web app</h1><p class="build-deck">Describe what you want. The host edits the project; everyone in the room sees the same preview.</p></div><div class="build-status" id="build-status" role="status" aria-live="polite"><i></i><span>Ready</span></div></div>
      <div class="build-toolbar"><div class="build-files" id="build-files" role="tablist" aria-label="Project files"></div><div class="build-tools"><button type="button" class="build-tool" id="build-new" title="Clear the current project. Undo restores it.">New project</button><button type="button" class="build-tool" id="build-undo" title="Restore the previous project version" disabled>Undo</button><button type="button" class="build-tool" id="build-run">Run preview</button><button type="button" class="build-tool build-stop" id="build-stop" hidden>Stop</button></div></div>
      <div class="build-workspace"><section class="build-editor-panel" aria-label="Project source"><div class="build-panel-label"><span id="build-file-name">index.html</span><span class="build-owner" id="build-owner">HOST EDITOR</span></div><textarea id="build-editor" spellcheck="false" aria-label="Source editor"></textarea><div class="build-log" id="build-log" role="log" aria-live="polite"><span class="build-log-dot"></span><span id="build-log-text">Preview is isolated from room devices and the network.</span></div></section>
      <section class="build-preview-panel" aria-label="Live app preview"><div class="build-panel-label"><span>PREVIEW</span><span class="build-preview-state"><i></i> LIVE</span></div><iframe id="build-preview" title="Generated app preview" sandbox="allow-scripts" referrerpolicy="no-referrer"></iframe><div class="build-console" id="build-console" aria-label="Preview console"></div></section></div>
      <form class="build-prompt-bar" id="build-form"><div class="build-examples" id="build-examples" aria-label="Example prompts"><span>TRY</span><button type="button" data-example="Create a polished Kerala travel guide with a map-inspired hero, three destination highlights, local food suggestions, and a simple itinerary. Use vivid but calm colors and make it work on mobile.">Kerala guide</button><button type="button" data-example="Create a clean focus timer with start, pause, reset, and a short break mode. Make the timer large, keyboard friendly, and easy to use on a phone.">Focus timer</button></div><label for="build-prompt">PROMPT</label><textarea id="build-prompt" rows="2" placeholder="Build a polished one-page app…" maxlength="1200"></textarea><div class="build-prompt-actions"><span id="build-permission">Host controls edits · Changes stay in this browser</span><button class="build-generate" id="build-generate" type="submit">Build app <span aria-hidden="true">↗</span></button></div></form>`;
    document.getElementById("room-screen")?.append(pane);
    this.pane = pane;
    this.editor = pane.querySelector("#build-editor");
    this.preview = pane.querySelector("#build-preview");
    this.console = pane.querySelector("#build-console");
    this.status = pane.querySelector("#build-status");
    this.renderFiles();
    this.editor.addEventListener("input", () => {
      if (!this.allowEdits || this.generating) return;
      this.files[this.activeFile] = this.editor.value;
      this.scheduleSaveAndSync();
      this.schedulePreview();
    });
    pane.querySelector("#build-run").addEventListener("click", () => this.runPreview());
    pane.querySelector("#build-undo").addEventListener("click", () => this.undo());
    pane.querySelector("#build-new").addEventListener("click", () => this.newProject());
    pane.querySelector("#build-stop").addEventListener("click", () => window.roomBuildStop?.());
    pane.querySelector("#build-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const prompt = pane.querySelector("#build-prompt").value.trim();
      if (prompt) this.generate(prompt);
    });
    pane.querySelector("#build-examples").addEventListener("click", (event) => {
      const button = event.target.closest("[data-example]");
      if (!button || !this.allowEdits) return;
      pane.querySelector("#build-prompt").value = button.dataset.example;
      pane.querySelector("#build-prompt").focus();
    });
    pane.querySelector("#build-prompt").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); pane.querySelector("#build-form").requestSubmit(); }
    });
    pane.querySelector("#build-files").addEventListener("click", (event) => {
      const button = event.target.closest("[data-file]");
      if (!button) return;
      this.files[this.activeFile] = this.editor.value;
      this.activeFile = button.dataset.file;
      this.renderFiles();
    });
    window.addEventListener("message", (event) => this.onPreviewMessage(event));
    const buildTab = document.getElementById("room-build-tab");
    buildTab?.addEventListener("click", () => { this.show(); window.roomBuildRequestState?.(); });
    document.getElementById("room-chat-tab")?.addEventListener("click", () => this.hide());
    document.getElementById("room-performance-tab")?.addEventListener("click", () => this.hide());
    this.setHost(Boolean(window.roomBuildIsHost?.()));
    this.renderPreview();
  }

  show() {
    const isHost = Boolean(window.roomBuildIsHost?.());
    this.setHost(isHost);
    if (!isHost) {
      this.files = { ...DEFAULT_FILES };
      this.activeFile = "index.html";
      this.renderFiles();
      this.renderPreview();
    }
    this.pane?.classList.add("visible");
    document.getElementById("room-screen")?.classList.add("room-build-view");
    for (const id of ["room-chat-tab", "room-performance-tab", "room-build-tab"]) {
      const button = document.getElementById(id);
      if (!button) continue;
      const active = id === "room-build-tab";
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
  }

  hide() {
    this.pane?.classList.remove("visible");
    document.getElementById("room-screen")?.classList.remove("room-build-view");
  }

  setHost(isHost) {
    this.allowEdits = isHost;
    if (!this.pane) return;
    this.editor.readOnly = !isHost || this.generating;
    this.pane.querySelector("#build-prompt").disabled = !isHost || this.generating;
    this.pane.querySelector("#build-generate").disabled = !isHost || this.generating;
    this.pane.querySelector("#build-owner").textContent = isHost ? "HOST EDITOR" : "VIEW ONLY";
    this.pane.querySelector("#build-new").disabled = !isHost || this.generating;
    this.pane.querySelectorAll("#build-examples button").forEach((button) => { button.disabled = !isHost || this.generating; });
    this.pane.querySelector("#build-permission").textContent = isHost ? "Host controls edits · Changes stay in this browser" : "Following the host · Preview updates live";
    this.renderFiles();
  }

  renderFiles() {
    if (!this.pane) return;
    const tabs = this.pane.querySelector("#build-files");
    tabs.innerHTML = Object.keys(this.files).map((file) => `<button type="button" class="build-file-tab${file === this.activeFile ? " active" : ""}" role="tab" aria-selected="${file === this.activeFile}" data-file="${esc(file)}">${esc(file)}</button>`).join("");
    this.pane.querySelector("#build-file-name").textContent = this.activeFile;
    this.editor.value = this.files[this.activeFile] || "";
    this.editor.readOnly = !this.allowEdits || this.generating;
  }

  setStatus(text, mode = "ready") {
    if (!this.status) return;
    this.status.className = `build-status ${mode}`;
    this.status.querySelector("span").textContent = text;
    this.pane.querySelector("#build-stop").hidden = !(mode === "working" && this.allowEdits);
    this.pane.querySelector("#build-undo").disabled = !this.previousFiles || !this.allowEdits;
    this.pane.querySelector("#build-new").disabled = !this.allowEdits || this.generating;
    this.pane.querySelectorAll("#build-examples button").forEach((button) => { button.disabled = !this.allowEdits || this.generating; });
    this.pane.querySelector("#build-generate").disabled = !this.allowEdits || this.generating;
    this.editor.readOnly = !this.allowEdits || this.generating;
    this.pane.querySelector("#build-prompt").disabled = !this.allowEdits || this.generating;
  }

  setLog(text, error = false) {
    const label = this.pane.querySelector("#build-log-text");
    label.textContent = text;
    label.parentElement.classList.toggle("error", error);
  }

  async generate(prompt, repair = false) {
    if (!this.allowEdits || this.generating) return;
    this.generating = true;
    this.repairing = repair;
    if (!repair) { this.originalPrompt = prompt; this.autoRepairUsed = false; }
    this.lastPrompt = prompt;
    if (!repair) this.previousFiles = structuredClone(this.files);
    this.console.innerHTML = "";
    this.setStatus(repair ? "Repairing preview" : "Building app", "working");
    this.setLog(repair ? "Applying one automatic repair from the preview error…" : "Generating project files with the room model…");
    this.pane.querySelector("#build-prompt").value = repair ? this.pane.querySelector("#build-prompt").value : prompt;
    this.renderFiles();
    const systemPrompt = `You build small, polished web apps for a browser-only workspace. Return only project files as fenced code blocks, with the filename on the opening fence, exactly like \`\`\`html file="index.html". You may return index.html, style.css, and app.js. Include complete working code, no dependencies or external network requests, and make the app responsive and accessible. The preview runs in a sandbox. ${repair ? "Fix the reported preview error while keeping the app's design and behavior. Return the full corrected files." : "Create the app described below. Always include a complete index.html."}`;
    const currentFiles = Object.entries(this.files).map(([name, content]) => {
      const lang = name.endsWith(".html") ? "html" : name.endsWith(".css") ? "css" : "js";
      return `\`\`\`${lang} file="${name}"\n${content}\n\`\`\``;
    }).join("\n\n");
    try {
      await window.roomBuildGenerate?.(`${systemPrompt}\n\n${repair ? "Original request: " + this.originalPrompt + "\n" : ""}Request: ${prompt}\n\nCurrent files:\n${currentFiles}`, { repair,
        onStart: (data) => this.receiveStart(data),
        onToken: (text) => this.receiveToken(text),
        onDone: (text, data) => this.receiveDone(text, data),
        onError: (message) => this.receiveError(message),
      });
    } catch (error) {
      this.receiveError(error.message || "Build generation failed.");
    }
  }

  receiveStart(data = {}) {
    this.generating = true;
    this.setStatus(data.repair ? "Repairing preview" : "Building app", "working");
    this.setLog(data.model ? `Generating with ${data.model}…` : "Generating with the room model…");
  }

  receiveToken(text) {
    if (!this.generating) return;
    this.streamText = (this.streamText || "") + text;
    this.setLog("Writing project files…");
  }

  receiveDone(text, data = {}) {
    const files = parseFiles(text || this.streamText || "");
    this.generating = false;
    this.streamText = "";
    if (!files) {
      this.setStatus("Needs another try", "error");
      this.setLog("The model response did not contain recognizable project files. Try a shorter, specific prompt.", true);
      this.renderFiles();
      return;
    }
    this.files = files;
    this.activeFile = "index.html";
    this.renderFiles();
    this.saveAndSync();
    this.renderPreview();
    this.setStatus("Preview ready", "ready");
    this.setLog(data.stats ? `Project updated · ${data.stats}` : "Project updated. The room preview is in sync.");
  }

  receiveError(message) {
    this.generating = false;
    this.streamText = "";
    this.renderFiles();
    this.setStatus("Build stopped", "error");
    this.setLog(String(message || "Generation failed."), true);
  }

  undo() {
    if (!this.allowEdits || !this.previousFiles) return;
    this.files = this.previousFiles;
    this.previousFiles = null;
    this.activeFile = "index.html";
    this.renderFiles();
    this.saveAndSync();
    this.renderPreview();
    this.setStatus("Previous version restored", "ready");
    this.setLog("Undo restored the project from before the last build.");
  }

  newProject() {
    if (!this.allowEdits || this.generating) return;
    this.previousFiles = structuredClone(this.files);
    this.files = { ...DEFAULT_FILES };
    this.activeFile = "index.html";
    this.console.innerHTML = "";
    this.renderFiles();
    this.saveAndSync();
    this.renderPreview();
    this.setStatus("New project ready", "ready");
    this.setLog("The starter project is ready. Undo restores the cleared files.");
  }

  schedulePreview() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    clearTimeout(this.previewTimer);
    this.previewTimer = setTimeout(() => { this.renderQueued = false; this.renderPreview(); }, 500);
  }

  scheduleSaveAndSync() {
    clearTimeout(this.syncTimer);
    this.syncTimer = setTimeout(() => this.saveAndSync(), 140);
  }

  runPreview() {
    this.files[this.activeFile] = this.editor.value;
    this.autoRepairUsed = false;
    this.console.innerHTML = "";
    this.saveAndSync();
    this.renderPreview();
    this.setLog("Preview refreshed in an isolated browser frame.");
  }

  renderPreview() {
    if (!this.preview) return;
    this.preview.srcdoc = previewDocument(this.files);
  }

  onPreviewMessage(event) {
    if (event.source !== this.preview?.contentWindow || event.data?.source !== "shardx-build-preview") return;
    const kind = event.data.kind || "log";
    const line = document.createElement("div");
    line.className = `build-console-line ${kind === "error" ? "error" : kind === "warn" ? "warn" : ""}`;
    line.textContent = `${kind}: ${(event.data.args || []).join(" ")}`;
    this.console.append(line);
    this.console.scrollTop = this.console.scrollHeight;
    if (kind === "error") {
      this.setLog(`Preview error: ${(event.data.args || []).join(" ")}`, true);
      if (this.allowEdits && !this.generating && !this.autoRepairUsed) {
        this.autoRepairUsed = true;
        setTimeout(() => this.generate(`Fix this preview error: ${(event.data.args || []).join(" ")}`, true), 250);
      }
    }
  }

  saveAndSync() {
    if (!this.allowEdits) return;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ files: this.files, updatedAt: Date.now() })); } catch {}
    window.roomBuildBroadcast?.({ files: this.files, revision: ++this.revision });
  }

  receiveState(state) {
    if (this.allowEdits || !state?.files?.["index.html"]) return;
    this.files = state.files;
    this.revision = state.revision || this.revision;
    if (!this.files[this.activeFile]) this.activeFile = "index.html";
    this.renderFiles();
    this.renderPreview();
  }

  receiveHostStart(data) { this.streamText = ""; this.receiveStart(data); }
  receiveHostToken(text) { this.receiveToken(text); }
  receiveHostDone(text, data) { this.receiveDone(text, data); }
  receiveHostError(message) { this.receiveError(message); }
}

export const buildMode = new BuildMode();
