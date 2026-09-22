import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { zstdDecompressSync } from "node:zlib";
import { buildPresentationTar, main, optimizeSvg } from "../lib/typstslideviewer.js";

function readTar(bytes) {
  const entries = new Map();
  for (let offset = 0; offset + 512 <= bytes.length;) {
    const header = bytes.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) break;
    const name = header.subarray(0, 100).toString("utf8").replace(/\0.*$/, "");
    const size = Number.parseInt(header.subarray(124, 136).toString("utf8").replace(/\0.*$/, "").trim() || "0", 8);
    entries.set(name, bytes.subarray(offset + 512, offset + 512 + size));
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  return entries;
}

const tinyPng = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAP+KeNJXAAAADElEQVR42mNgYGAAAAAEAAGjChM8AAAAAElFTkSuQmCC", "base64");
const pngUri = `data:image/png;base64,${tinyPng.toString("base64")}`;

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
  // Plain Typst documents have no <pdfpc-file> metadata, so `typst query`
  // returns an empty array and the generator must synthesize page metadata.
  await writeFile(join(svgs, "meta.json"), "[]");
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
  assert.match(viewer, /const thumbnailCache = Object\.fromEntries/);
  assert.match(viewer, /const presenterThumbnailCache = Object\.fromEntries/);
  assert.match(viewer, /URL\.createObjectURL\(new Blob/);
  assert.match(viewer, /ensureThumbnails\(\)\.then\(highlightCurrentThumbnail\)/);
  assert.doesNotMatch(viewer, /\n\s*loadThumbnails\(\);\n/);
  const encoded = viewer.match(/const base64String = "([^"]+)"/)[1];
  const entries = readTar(zstdDecompressSync(Buffer.from(encoded, "base64")));
  const manifest = JSON.parse(entries.get("manifest.json").toString("utf8"));
  assert.equal(manifest.format, "typst-slide-viewer");
  assert.equal(manifest.version, 2);
  assert.match(entries.get("slides/0.svg").toString("utf8"), /Hello/);
  assert.deepEqual(manifest.thumbnails, {});
  assert.deepEqual(manifest.presenterThumbnails, {});
});

test("presentation TAR deduplicates binary slide assets by decoded bytes", () => {
  const alternateEncoding = `data:image/png;base64,${tinyPng.toString("base64").replace(/=+$/, "")}`;
  const entries = readTar(buildPresentationTar({
    slides: { 0: `<svg><image href="${pngUri}" /></svg>`, 1: `<svg><image href="${alternateEncoding}" /></svg>` },
  }));
  const manifest = JSON.parse(entries.get("manifest.json").toString("utf8"));
  assert.equal(manifest.assets.length, 1);
  assert.equal([...entries.keys()].filter((name) => name.startsWith("assets/")).length, 1);
  assert.match(entries.get("slides/0.svg").toString(), /@@TSV_ASSET_0@@/);
  assert.match(entries.get("slides/1.svg").toString(), /@@TSV_ASSET_0@@/);
});

test("presentation TAR keeps distinct assets and thumbnail bytes out of the manifest", () => {
  const anotherUri = `data:image/png;base64,${Buffer.from("different image bytes").toString("base64")}`;
  const entries = readTar(buildPresentationTar({
    slides: { 0: `<svg><image href="${pngUri}"/><image href="${anotherUri}"/></svg>` },
    thumbnails: { 1: pngUri }, presenterThumbnails: { 0: pngUri },
  }));
  const manifestText = entries.get("manifest.json").toString("utf8");
  const manifest = JSON.parse(manifestText);
  assert.equal(manifest.assets.length, 2);
  assert.deepEqual(manifest.thumbnails, { 1: 0 });
  assert.deepEqual(manifest.presenterThumbnails, { 0: 0 });
  assert.doesNotMatch(manifestText, /data:image|base64,/);
  assert.deepEqual(entries.get("assets/0.png"), tinyPng);
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

test("optimizer converts embedded bitmap images to WebP", async () => {
  const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAADUExURf8AABniCTcAAAAHdElNRQfqCQ0QCyKuwxO3AAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDI2LTA5LTEzVDE2OjExOjM0KzAwOjAwZyZ3uwAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyNi0wOS0xM1QxNjoxMTozNCswMDowMBZ7zwcAAAAodEVYdGRhdGU6dGltZXN0YW1wADIwMjYtMDktMTNUMTY6MTE6MzQrMDA6MDBBbu7YAAAACklEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==";
  const svg = `<svg><image href="data:image/png;base64,${png}" /></svg>`;
  const optimized = await optimizeSvg(svg, process.cwd(), { imageFormat: "webp", quality: 50 });
  assert.match(optimized, /data:image\/webp;base64,/);
  assert.doesNotMatch(optimized, /data:image\/png;base64,/);

  const retained = await optimizeSvg(svg, process.cwd(), { optimizePng: false });
  assert.match(retained, /data:image\/png;base64,/);
});
