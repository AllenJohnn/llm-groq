import { readdirSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

const root = process.cwd();
const engineFiles = readdirSync(join(root, "engine"))
  .filter((name) => name.endsWith(".js"))
  .map((name) => join("engine", name));
const nodeFiles = ["tests", "benchmarks", "scripts"]
  .flatMap((dir) => readdirSync(join(root, dir))
    .filter((name) => name.endsWith(".js") || name.endsWith(".mjs"))
    .map((name) => join(dir, name)));
nodeFiles.push("test_deepseek.js", "test_ui.mjs");

let failed = false;
for (const file of engineFiles) {
  const result = spawnSync("deno", ["check", file], { stdio: "inherit" });
  if (result.status !== 0) failed = true;
}
for (const file of nodeFiles) {
  const result = spawnSync(process.execPath, ["--check", file], { stdio: "inherit" });
  if (result.status !== 0) failed = true;
}

if (failed) process.exitCode = 1;
else console.log(`Syntax and type checks passed (${engineFiles.length} engine files, ${nodeFiles.length} test/benchmark files).`);
