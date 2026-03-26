import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "adapters/vercel-ai": "src/adapters/vercel-ai.ts",
    "adapters/openai": "src/adapters/openai.ts",
    "adapters/anthropic": "src/adapters/anthropic.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: true,
  treeshake: true,
});
