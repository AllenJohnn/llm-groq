import { makeTokenizer } from "./engine/tokenizer.js";
import { MODELS } from "./room/models.js";

const modelKey = "deepseek-r1-distill-qwen-14b";
const model = MODELS[modelKey];
if (!model?.tok?.startsWith("https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/")) {
  throw new Error(`Unexpected DeepSeek tokenizer URL: ${model?.tok}`);
}

const response = await fetch(model.tok);
if (!response.ok) throw new Error(`Tokenizer request failed: HTTP ${response.status}`);
const tokenizer = makeTokenizer(await response.json());

const expectedSpecials = new Map([
  ["<｜begin▁of▁sentence｜>", 151646],
  ["<｜end▁of▁sentence｜>", 151643],
  ["<｜User｜>", 151644],
  ["<｜Assistant｜>", 151645],
]);
for (const [token, expectedId] of expectedSpecials) {
  if (tokenizer.vocab[token] !== expectedId) {
    throw new Error(`${token} mapped to ${tokenizer.vocab[token]}, expected ${expectedId}`);
  }
}

const sample = "Who are you? Answer in one word.";
const encoded = tokenizer.encode(sample);
if (tokenizer.decode(encoded) !== sample) throw new Error("Ordinary-text tokenizer round trip failed");

console.log(`Tokenizer loaded from ${model.tok}`);
console.log(`Verified ${expectedSpecials.size} DeepSeek special-token IDs.`);
console.log(`Ordinary-text round trip passed (${encoded.length} tokens).`);
console.log("No model weights were loaded; this script does not verify generation.");
