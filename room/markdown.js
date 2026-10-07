// Ant Design X-inspired streaming Markdown renderer for webslice.
import { marked } from "../vendor/marked.esm.js";

export function esc(s) {
  if (typeof s !== "string") return "";
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Custom renderer for code blocks with header, language badge, and copy button
const renderer = new marked.Renderer();
renderer.code = function({ text, lang }) {
  const language = (lang || "").trim().toLowerCase() || "code";
  const cleanCode = esc(text);
  
  if (language === 'html') {
    return `<div class="code-block has-tabs" data-lang="html">
  <div class="code-header">
    <div class="code-tabs" role="tablist" aria-label="HTML result">
      <button type="button" class="code-tab-btn active" role="tab" aria-selected="true" onclick="window.switchCodeTab(this, 'code')">Code</button>
      <button type="button" class="code-tab-btn" role="tab" aria-selected="false" onclick="window.switchCodeTab(this, 'preview')">Preview</button>
    </div>
    <button class="code-copy-btn" onclick="copyCode(this)" title="Copy code" aria-label="Copy code">
      <svg class="copy-icon" width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
        <path d="M4 2a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a2 2 0 0 0 2 2h6a1 1 0 0 0 1-1v-1H3a2 2 0 0 1-2-2V5H2z"/>
      </svg>
      <span class="copy-text">Copy</span>
    </button>
  </div>
  <div class="code-tab-content code-view active" role="tabpanel">
    <pre><code class="language-html">${cleanCode}</code></pre>
  </div>
  <div class="code-tab-content preview-view" role="tabpanel" hidden>
    <div class="preview-toolbar">
      <span>Sandboxed preview</span>
      <button type="button" class="code-preview-refresh" onclick="window.refreshCodePreview(this)">Refresh preview</button>
    </div>
    <iframe sandbox="allow-scripts" title="HTML code preview" loading="lazy" referrerpolicy="no-referrer"></iframe>
  </div>
  <template class="preview-source">${cleanCode}</template>
</div>`;
  }

  return `<div class="code-block" data-lang="${esc(language)}">
  <div class="code-header">
    <span class="code-lang">${esc(language)}</span>
    <button class="code-copy-btn" onclick="copyCode(this)" title="Copy code" aria-label="Copy code">
      <svg class="copy-icon" width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
        <path d="M4 2a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a2 2 0 0 0 2 2h6a1 1 0 0 0 1-1v-1H3a2 2 0 0 1-2-2V5H2z"/>
      </svg>
      <span class="copy-text">Copy</span>
    </button>
  </div>
  <pre><code class="language-${esc(language)}">${cleanCode}</code></pre>
</div>`;
};

// Open links in new tab safely
renderer.link = function({ href, title, text }) {
  const t = title ? ` title="${esc(title)}"` : "";
  return `<a href="${esc(href)}"${t} target="_blank" rel="noopener noreferrer">${text}</a>`;
};

marked.use({
  renderer,
  gfm: true,
  breaks: true,
});

export function md(src, isStreaming = false) {
  if (!src) return "";
  let content = src;

  // Ant Design X ThoughtChain pattern: render <think> tags into collapsible thought accordion
  if (content.includes("<think>")) {
    content = content.replace(/<think>([\s\S]*?)<\/think>/gi, (_, inner) => {
      const trimmed = inner.trim();
      return `\n\n<details class="thought-chain" data-status="completed"><summary class="thought-header"><svg class="thought-icon thought-done" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a6 6 0 0 0-6 6c0 2.2 1.2 4.1 3 5.1V17a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-3.9c1.8-1 3-2.9 3-5.1a6 6 0 0 0-6-6z"/><path d="M9 22h6"/></svg><span class="thought-title">Thought Process</span><span class="thought-badge done">completed</span><svg class="thought-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg></summary><div class="thought-content">\n\n${trimmed}\n\n</div></details>\n\n`;
    });
    if (isStreaming && content.includes("<think>")) {
      content = content.replace(/<think>([\s\S]*?)$/gi, (_, inner) => {
        const trimmed = inner.trim();
        return `\n\n<details class="thought-chain" data-status="streaming" open><summary class="thought-header"><svg class="thought-icon thinking-pulse" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a6 6 0 0 0-6 6c0 2.2 1.2 4.1 3 5.1V17a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-3.9c1.8-1 3-2.9 3-5.1a6 6 0 0 0-6-6z"/><path d="M9 22h6"/></svg><span class="thought-title">Thinking…</span><span class="thought-badge streaming">in progress</span><svg class="thought-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg></summary><div class="thought-content">\n\n${trimmed}\n\n</div></details>\n\n`;
      });
    }
  }

  // In streaming mode, handle incomplete code fences (Ant Design X x-markdown pattern)
  if (isStreaming) {
    const fenceMatches = content.match(/```/g);
    if (fenceMatches && fenceMatches.length % 2 !== 0) {
      content += "\n```"; // Temporarily complete the code block for streaming render
    }
  }

  try {
    return marked.parse(content);
  } catch (err) {
    console.warn("Markdown parse fallback:", err);
    return `<p>${esc(content)}</p>`;
  }
}
