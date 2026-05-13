import type { CollectionEntry } from "astro:content";

export type Essay = CollectionEntry<"essays">;

export function compareEssaysByDateDesc(a: Essay, b: Essay) {
  return b.data.date.valueOf() - a.data.date.valueOf();
}

export function getEssaySlug(essay: Pick<Essay, "id">) {
  return essay.id.replace(/\.(md|mdx)$/i, "");
}

export function getEssayUrl(essay: Pick<Essay, "id">) {
  return `/essays/${getEssaySlug(essay)}/`;
}

export function getEssayKeywords(essay: Pick<Essay, "data">) {
  return [...new Set([...essay.data.keywords, ...essay.data.tags])];
}

export function formatEssayDate(date: Date, locale = "zh-CN") {
  return new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}
