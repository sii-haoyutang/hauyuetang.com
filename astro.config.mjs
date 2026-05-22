// @ts-check
import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import rehypeFigureCaptions from "./src/utils/rehype-figure-captions.mjs";

// https://astro.build/config
export default defineConfig({
  site: "https://hauyuetang.com",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeFigureCaptions, rehypeKatex],
  },
});
