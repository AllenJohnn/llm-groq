import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

const quick = [
  "test_qwen.js", "test_smollm.js", "test_stream.js", "test_batch.js",
  "test_qwen_split.js", "test_qwen_stream.js", "test_topologies.js",
];
const q38 = [
  "test_q38.js", "test_batch_q38.js", "test_mtp.js", "test_b4.js",
  "test_twins.js", "test_gemm.js", "test_q38_split.js", "test_mtp_split.js",
  "test_ctx.js", "test_reset.js", "test_batch_split.js",
];
const suites = { quick, q38, all: [...quick, ...q38] };
const suiteName = process.argv[2] || "quick";
const suite = suites[suiteName];
if (!suite) {
  console.error(`Unknown GPU suite: ${suiteName}. Choose quick, q38, or all.`);
  process.exit(2);
}

let failed = 0;
for (const test of suite) {
  console.log(`=== ${test}`);
  const result = spawnSync("deno", [
    "run", "--unstable-webgpu", "--allow-read", "--allow-env", test,
  ], {
    cwd: resolve("tests"),
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
  });

  const output = `${result.stdout || ""}${result.stderr || ""}`
    .split(/\r?\n/)
    .filter((line) => !/^TU:|^MESA/.test(line));
  const tail = result.error || result.status !== 0 ? 12 : 4;
  for (const line of output.slice(-tail)) console.log(line);

  if (result.error || result.status !== 0) {
    failed++;
    if (result.error) console.error(result.error.message);
    console.error(`${test} failed (exit ${result.status ?? "unavailable"}).`);
  }
}

console.log(`\n${suite.length - failed}/${suite.length} GPU tests passed; ${failed} failed.`);
if (failed) process.exitCode = 1;
