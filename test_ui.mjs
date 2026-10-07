import { spawn } from "node:child_process";
import { chromium } from "playwright";

const port = 8081;
const origin = `http://127.0.0.1:${port}`;
const server = spawn(process.execPath, ["scripts/server.mjs", "--port", String(port)], {
  stdio: "ignore",
});
let browser;

try {
  let ready = false;
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      const response = await fetch(`${origin}/room`);
      if (response.ok) { ready = true; break; }
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  if (!ready) throw new Error("Local room server did not become ready on port 8081");

  browser = await chromium.launch();
  const context = await browser.newContext();
  await context.addInitScript(() => localStorage.setItem("webslice_fallbackmode", "true"));
  const page = await context.newPage();
  await page.goto(`${origin}/room`);
  await page.waitForFunction(() => typeof window.fallbackmode === "boolean");
  const selectedModel = await page.evaluate(() => {
    const select = document.querySelector("#ai-model");
    select.value = "deepseek-r1-distill-qwen-14b";
    select.dispatchEvent(new Event("change", { bubbles: true }));
    return select.value;
  });
  if (selectedModel !== "deepseek-r1-distill-qwen-14b") {
    throw new Error("DeepSeek model option was not selected");
  }
  if (!(await page.evaluate(() => window.fallbackmode))) {
    throw new Error("Expected the test page to start in Groq mode");
  }
  await page.locator("#sidebar-fallback-toggle").evaluate((el) => el.click());
  await page.waitForFunction(() => window.fallbackmode === false);
  if (await page.evaluate(() => localStorage.getItem("webslice_fallbackmode")) !== "false") {
    throw new Error("Groq mode remained enabled after toggling it off");
  }
  console.log("UI smoke check passed: selected DeepSeek and switched Groq mode off.");
  console.log("The test does not start model loading or claim to verify generation.");
} finally {
  await browser?.close();
  server.kill();
}
