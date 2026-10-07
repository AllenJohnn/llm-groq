// Groq API client for streaming chat completions in LLM ShardX.
// Calls the server-side proxy endpoint (/api/groq) by default so GROQ_API_KEY is never exposed in client JS.
// Can also call Groq directly if an explicit apiKey is supplied.

export const GROQ_PROXY_URL = "/api/groq";
export const GROQ_DIRECT_URL = "https://api.groq.com/openai/v1/chat/completions";

/**
 * Format Groq error statuses into clear, helpful user-facing messages.
 */
export function formatGroqError(status, data, model) {
  let rawMsg = "";
  if (typeof data === "string") {
    try {
      const parsed = JSON.parse(data);
      rawMsg = parsed.error?.message || data;
    } catch {
      rawMsg = data;
    }
  } else if (data && typeof data === "object") {
    rawMsg = data.error?.message || JSON.stringify(data);
  }

  if (typeof rawMsg === "string") {
    if (rawMsg.includes("<html") || rawMsg.includes("<!DOCTYPE")) {
      const match = rawMsg.match(/<title>(.*?)<\/title>/i);
      rawMsg = match ? match[1].trim() : "Server returned an HTML error page";
    }
    if (rawMsg.length > 300) {
      rawMsg = rawMsg.slice(0, 300) + "…";
    }
  }

  if (status === 401) {
    return `Invalid Groq API Key (401). Please check GROQ_API_KEY in .env. (${rawMsg})`;
  }
  if (status === 404) {
    if (typeof rawMsg === "string" && (rawMsg.includes("HTML") || rawMsg.includes("Cannot POST") || rawMsg.includes("404") || rawMsg.includes("not found on this server"))) {
      return `Proxy endpoint (/api/groq) returned 404 (Not Found). Please ensure the LLM ShardX server is running via "node scripts/server.mjs" (or "npm run serve") on http://localhost:8080. (${rawMsg})`;
    }
    return `Model "${model}" not found on Groq (404). This model id may not be supported by Groq. (${rawMsg})`;
  }
  if (status === 429) {
    return `Groq rate limit exceeded (429). Please wait a moment before sending another prompt. (${rawMsg})`;
  }
  if (status === 400) {
    return `Groq request error (400) for "${model}": ${rawMsg}`;
  }
  if (status === 503) {
    return `Groq service is over capacity or temporarily unavailable (503). (${rawMsg})`;
  }
  return `Groq API error (${status}): ${rawMsg || "Unknown error"}`;
}

/**
 * Parses an SSE text chunk and extracts token deltas.
 */
export function parseGroqSSEChunk(chunk, onToken) {
  let textExtracted = "";
  let isDone = false;
  const lines = chunk.split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(":")) continue;
    if (trimmed === "data: [DONE]") {
      isDone = true;
      continue;
    }
    if (trimmed.startsWith("data: ")) {
      try {
        const payload = JSON.parse(trimmed.slice(6));
        const delta = payload.choices?.[0]?.delta?.content || "";
        if (delta) {
          textExtracted += delta;
          if (onToken) onToken(delta);
        }
      } catch {
        // partial chunk, buffered in stream handler
      }
    }
  }

  return { text: textExtracted, isDone };
}

/**
 * Streams chat completion from Groq API or proxy endpoint.
 * Acts as an async generator yielding token strings incrementally,
 * while also invoking onToken(token, fullText) callback if provided.
 */
export async function* streamGroqChat({
  model,
  messages,
  prompt,
  endpoint,
  apiKey,
  signal,
  temperature = 0.6,
  max_tokens = 4096,
  meta,
  onToken,
  onFinish,
}) {
  const msgs = messages && messages.length
    ? messages
    : [{ role: "user", content: prompt || "" }];

  // If apiKey is provided explicitly, default to direct Groq URL; otherwise use proxy endpoint
  let targetUrl = endpoint || (apiKey ? GROQ_DIRECT_URL : GROQ_PROXY_URL);
  if (!endpoint && !apiKey && typeof window !== "undefined" && window.location?.protocol === "file:") {
    targetUrl = "http://localhost:8080/api/groq";
  }

  const headers = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey.trim()}`;
  }

  let response;
  try {
    response = await fetch(targetUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model,
        messages: msgs,
        stream: true,
        temperature,
        max_tokens,
      }),
      signal,
    });
  } catch (netErr) {
    if (!endpoint && !apiKey && typeof window !== "undefined" && window.location?.origin !== "http://localhost:8080") {
      try {
        targetUrl = "http://localhost:8080/api/groq";
        response = await fetch(targetUrl, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model,
            messages: msgs,
            stream: true,
            temperature,
            max_tokens,
          }),
          signal,
        });
      } catch {
        throw netErr;
      }
    } else {
      throw netErr;
    }
  }

  // If local static server returned 404 HTML, attempt fallback to http://localhost:8080/api/groq
  if (!response.ok && response.status === 404 && !endpoint && !apiKey && typeof window !== "undefined" && window.location?.origin !== "http://localhost:8080") {
    try {
      const fallbackUrl = "http://localhost:8080/api/groq";
      const fallbackResp = await fetch(fallbackUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model,
          messages: msgs,
          stream: true,
          temperature,
          max_tokens,
        }),
        signal,
      });
      if (fallbackResp.ok || fallbackResp.status !== 404) {
        response = fallbackResp;
      }
    } catch {}
  }

  if (!response.ok) {
    let rawText = "";
    try {
      rawText = await response.text();
    } catch {}
    let errBody = rawText;
    try {
      errBody = JSON.parse(rawText);
    } catch {}
    const formatted = formatGroqError(response.status, errBody, model);
    throw new Error(formatted);
  }

  if (!response.body) {
    throw new Error("ReadableStream not supported on this fetch response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let fullText = "";
  let inThinking = false;
  let lastFinishReason = null;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep partial line for next iteration

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(":")) continue;
        if (trimmed === "data: [DONE]") {
          if (inThinking) {
            const closeTag = "\n</think>\n\n";
            fullText += closeTag;
            if (onToken) onToken(closeTag, fullText);
            yield closeTag;
            inThinking = false;
          }
          if (meta && typeof meta === "object") {
            meta.finishReason = lastFinishReason;
            meta.capped = lastFinishReason === "length";
            meta.fullText = fullText;
          }
          if (typeof onFinish === "function") {
            onFinish({
              finishReason: lastFinishReason,
              capped: lastFinishReason === "length",
              fullText,
            });
          }
          return fullText;
        }

        if (trimmed.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            const choice = parsed.choices?.[0];
            if (choice?.finish_reason) {
              lastFinishReason = choice.finish_reason;
              if (meta && typeof meta === "object") {
                meta.finishReason = lastFinishReason;
                meta.capped = lastFinishReason === "length";
              }
            }
            const delta = choice?.delta;
            if (!delta) continue;

            // Handle Groq reasoning streams (e.g. deepseek-r1, gpt-oss)
            if (delta.reasoning) {
              if (!inThinking) {
                const openTag = "<think>\n";
                fullText += openTag;
                if (onToken) onToken(openTag, fullText);
                yield openTag;
                inThinking = true;
              }
              fullText += delta.reasoning;
              if (onToken) onToken(delta.reasoning, fullText);
              yield delta.reasoning;
            }

            // Handle standard content tokens
            if (delta.content) {
              if (inThinking) {
                const closeTag = "\n</think>\n\n";
                fullText += closeTag;
                if (onToken) onToken(closeTag, fullText);
                yield closeTag;
                inThinking = false;
              }
              fullText += delta.content;
              if (onToken) onToken(delta.content, fullText);
              yield delta.content;
            }
          } catch {
            // ignore partial JSON chunk
          }
        }
      }
    }

    if (inThinking) {
      const closeTag = "\n</think>\n\n";
      fullText += closeTag;
      if (onToken) onToken(closeTag, fullText);
      yield closeTag;
      inThinking = false;
    }
  } finally {
    if (meta && typeof meta === "object") {
      meta.finishReason = lastFinishReason;
      meta.capped = lastFinishReason === "length";
      meta.fullText = fullText;
    }
    if (typeof onFinish === "function") {
      onFinish({
        finishReason: lastFinishReason,
        capped: lastFinishReason === "length",
        fullText,
      });
    }
    try { reader.releaseLock(); } catch {}
  }

  return fullText;
}

/**
 * Non-streaming chat completion helper.
 */
export async function completeGroqChat({
  model,
  messages,
  prompt,
  endpoint,
  apiKey,
  signal,
  temperature = 0.6,
  max_tokens = 4096,
}) {
  let fullText = "";
  for await (const token of streamGroqChat({
    model,
    messages,
    prompt,
    endpoint,
    apiKey,
    signal,
    temperature,
    max_tokens,
  })) {
    fullText += token;
  }
  return fullText;
}
