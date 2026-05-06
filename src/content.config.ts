import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const essays = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/essays" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.date(),
    updated: z.date().optional(),
    draft: z.boolean().default(false),
    language: z.enum(["zh-CN", "en"]).default("zh-CN"),
    tags: z.array(z.string()).default([]),
  }),
});

export const collections = {
  essays,
};
