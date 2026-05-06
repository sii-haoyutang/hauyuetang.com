import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import {
  compareEssaysByDateDesc,
  getEssayUrl,
} from "../lib/essays";

const siteUrl = "https://hauyuetang.com";

export const GET: APIRoute = async () => {
  const essays = (await getCollection("essays", ({ data }) => data.draft !== true)).sort(
    compareEssaysByDateDesc,
  );

  const staticPages = ["/", "/essays/", "/about/"];
  const urls = [
    ...staticPages.map((path) => `<url><loc>${new URL(path, siteUrl).href}</loc></url>`),
    ...essays.map((essay) => {
      const loc = new URL(getEssayUrl(essay), siteUrl).href;
      const lastmod = (essay.data.updated ?? essay.data.date).toISOString().slice(0, 10);

      return `<url><loc>${loc}</loc><lastmod>${lastmod}</lastmod></url>`;
    }),
  ].join("");

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls}</urlset>`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
};
