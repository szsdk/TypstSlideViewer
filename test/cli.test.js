import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { zstdDecompressSync } from "node:zlib";
import { main, optimizeSvg } from "../lib/typstslideviewer.js";

test("html-embed writes the bundled helper", async () => {
  const directory = await mkdtemp(join(tmpdir(), "typstslideviewer-"));
  const output = join(directory, "html-embed.typ");
  await main(["html-embed", "--output", output]);
  assert.match(await readFile(output, "utf8"), /embed-html/);
});

test("generator emits a viewer from existing SVG files", async () => {
  const directory = await mkdtemp(join(tmpdir(), "typstslideviewer-"));
  const source = join(directory, "slides.typ"), svgs = join(directory, "svgs"), output = join(directory, "slides.html");
  await writeFile(source, "= A slide");
  await mkdir(svgs);
  await writeFile(join(svgs, "modified_slide_1.svg"), '<svg width="100" height="100"><text>Hello</text></svg>');
  await writeFile(join(svgs, "meta.json"), JSON.stringify({ pages: [{ idx: 0, label: "1", forcedOverlay: false, hidden: false }] }));
  await main([source, "--svg-folder", svgs, "--output-file", output, "--thumbnails", "false"]);
  const viewer = await readFile(output, "utf8");
  assert.match(viewer, /"label":1/);
  assert.match(viewer, /canvas\.toDataURL\("image\/webp", 0\.8\)/);
  assert.match(viewer, /xmlns:xlink="http:\/\/www\.w3\.org\/1999\/xlink"/);
  assert.match(viewer, /snapshotInteractiveRegions\(svgContent, canvas, context\)/);
  assert.match(viewer, /preserveDrawingBuffer: true/);
  assert.match(viewer, /drawIframeHtmlOverlay\(iframe, context, destination\)/);
  assert.match(viewer, /Math\.min\(4, jobs\.length\)/);
  assert.match(viewer, /requestIdleCallback\(resolve, \{ timeout: 200 \}\)/);
  assert.match(viewer, /setTimeout\(preload, 0\)/);
  assert.match(viewer, /const thumbnailCache = svgPackage\.thumbnails \|\| \{\}/);
  assert.match(viewer, /ensureThumbnails\(\)\.then\(highlightCurrentThumbnail\)/);
  assert.doesNotMatch(viewer, /\n\s*loadThumbnails\(\);\n/);
  const encoded = viewer.match(/const base64String = "([^"]+)"/)[1];
  const payload = JSON.parse(zstdDecompressSync(Buffer.from(encoded, "base64")).toString("utf8"));
  assert.match(payload.slides[0], /Hello/);
  assert.deepEqual(payload.thumbnails, {});
});

test("optimizer makes nested SVG video embeds playable", async () => {
  const directory = await mkdtemp(join(tmpdir(), "typstslideviewer-"));
  await writeFile(join(directory, "movie.mp4"), "video bytes");
  const inner = '<svg viewBox="0 0 400 300"><foreignObject width="400" height="300"><video><source src="movie.mp4" type="video/mp4" /></video></foreignObject></svg>';
  const outer = `<svg width="800" height="600"><image x="10" y="20" width="400" height="300" xlink:href="data:image/svg+xml;base64,${Buffer.from(inner).toString("base64")}" /></svg>`;
  const optimized = await optimizeSvg(outer, directory);
  assert.match(optimized, /<foreignObject/);
  assert.match(optimized, /data:video\/mp4;base64,dmlkZW8gYnl0ZXM=/);
  assert.doesNotMatch(optimized, /data:image\/svg\+xml/);
});
