import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import {
  compareEssaysByDateDesc,
  getEssayKeywords,
  getEssayUrl,
} from "../lib/essays";

const siteUrl = "https://hauyuetang.com";

function escapeXml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export const GET: APIRoute = async () => {
  const essays = (await getCollection("essays", ({ data }) => data.draft !== true)).sort(
    compareEssaysByDateDesc,
  );

  const items = essays
    .map((essay) => {
      const url = new URL(getEssayUrl(essay), siteUrl).href;
      const categories = getEssayKeywords(essay)
        .map((keyword) => `<category>${escapeXml(keyword)}</category>`)
        .join("");

      return [
        "<item>",
        `<title>${escapeXml(essay.data.title)}</title>`,
        `<description>${escapeXml(essay.data.description)}</description>`,
        `<link>${url}</link>`,
        `<guid>${url}</guid>`,
        `<pubDate>${essay.data.date.toUTCString()}</pubDate>`,
        categories,
        "</item>",
      ].join("");
    })
    .join("");

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Hauyue Tang</title>
    <description>关于政治、科学与现代世界的随笔与评论。</description>
    <link>${siteUrl}/</link>
    <language>zh-CN</language>
    ${items}
  </channel>
</rss>`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
    },
  });
};
