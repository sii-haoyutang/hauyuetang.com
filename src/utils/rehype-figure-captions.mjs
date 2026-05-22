import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";

function isElement(node, tagName) {
  return node && node.type === "element" && node.tagName === tagName;
}

const captionProcessor = unified()
  .use(remarkParse)
  .use(remarkMath)
  .use(remarkRehype);

function getCaptionInfo(imgNode) {
  const title = imgNode?.properties?.title;
  const alt = imgNode?.properties?.alt;

  if (typeof title === "string" && title.trim()) {
    return { value: title.trim(), isMarkdown: true, stripTitle: true };
  }

  if (typeof alt === "string" && alt.trim()) {
    return { value: alt.trim(), isMarkdown: false };
  }

  return null;
}

function parseCaptionMarkdown(markdown) {
  const tree = captionProcessor.runSync(captionProcessor.parse(markdown));
  const children = tree.children ?? [];

  if (children.length === 1 && isElement(children[0], "p")) {
    return children[0].children;
  }

  return children;
}

function buildCaptionChildren(captionInfo) {
  if (!captionInfo) {
    return [];
  }

  if (captionInfo.isMarkdown) {
    return parseCaptionMarkdown(captionInfo.value);
  }

  return [{ type: "text", value: captionInfo.value }];
}

function normalizeImageNode(imgNode, captionInfo) {
  if (!captionInfo?.stripTitle || !imgNode?.properties) {
    return imgNode;
  }

  const { title, ...properties } = imgNode.properties;
  return {
    ...imgNode,
    properties,
  };
}

function transformNode(node) {
  if (!node || !Array.isArray(node.children)) {
    return;
  }

  for (let i = 0; i < node.children.length; i += 1) {
    const child = node.children[i];

    if (isElement(child, "p") && child.children.length === 1 && isElement(child.children[0], "img")) {
      const rawImg = child.children[0];
      const captionInfo = getCaptionInfo(rawImg);
      const img = normalizeImageNode(rawImg, captionInfo);

      if (captionInfo) {
        node.children[i] = {
          type: "element",
          tagName: "figure",
          properties: {},
          children: [
            img,
            {
              type: "element",
              tagName: "figcaption",
              properties: {},
              children: buildCaptionChildren(captionInfo),
            },
          ],
        };
        continue;
      }
    }

    transformNode(child);
  }
}

export default function rehypeFigureCaptions() {
  return function transformer(tree) {
    transformNode(tree);
  };
}
