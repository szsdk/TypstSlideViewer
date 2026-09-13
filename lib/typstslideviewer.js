import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir, readdir, rm, stat, copyFile, mkdtemp } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { basename, dirname, extname, isAbsolute, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { zstdCompressSync, constants as zlibConstants } from "node:zlib";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ASSETS = join(ROOT, "assets");
const require = createRequire(import.meta.url);
const FZSTD_BUNDLE = join(dirname(require.resolve("fzstd")), "..", "umd", "index.js");
const TARTS_BUNDLE = require.resolve("tarts");
const DATA_URI = /data:[^;,]+;base64,[A-Za-z0-9+/=]+/g;

const HELP = `Usage:
gen-slide <typst-source> [options]
gen-slide html-embed [--output path] [--force]
gen-slide placeholders <typst-source> [options]

Options:
-o, --output-file <path>       HTML output path
-s, --svg-folder <path>        Generated SVG folder (default: svgs)
-t, --template-file <path>     Viewer template path
--optimize-png <boolean>   Optimize PNG images (currently retained as-is)
--optimize-jpg <boolean>   Optimize JPEG images (currently retained as-is)
--quality <number>         Image quality setting (default: 80)
--image-format <webp|avif> Image target format (default: webp)
--thumbnails <boolean>     Pre-render thumbnails during generation (default: true)
--note <right>             Speaker-note position
--no-transition            Disable slide transitions
-f, --force                    Rebuild generated files
-h, --help                     Show this help
`;

function parseArgs(argv) {
    const options = { svgFolder: "svgs", optimizePng: true, optimizeJpg: true, quality: 80,
        imageFormat: "webp", thumbnails: true, force: false, note: "", transition: true };
    const positional = [];
    for (let i = 0; i < argv.length; i += 1) {
        const arg = argv[i];
        if (!arg.startsWith("-")) { positional.push(arg); continue; }
        if (arg === "--help" || arg === "-h") options.help = true;
            else if (arg === "--force" || arg === "-f") options.force = true;
                else if (arg === "--no-transition") options.transition = false;
                    else {
                        const [name, inline] = arg.replace(/^--?/, "").split("=", 2);
                        const value = inline ?? argv[++i];
                        if (value === undefined) throw new Error(`Missing value for ${arg}`);
                        const keys = { o: "outputFile", "output-file": "outputFile", s: "svgFolder", "svg-folder": "svgFolder",
                            t: "templateFile", "template-file": "templateFile", note: "note", quality: "quality",
                            "image-format": "imageFormat", "optimize-png": "optimizePng", "optimize-jpg": "optimizeJpg",
                            thumbnails: "thumbnails",
                            browser: "browser", output: "output", "placeholder-folder": "placeholderFolder", width: "width",
                            height: "height", retries: "retries", "virtual-time-budget": "virtualTimeBudget" };
                        if (!keys[name]) throw new Error(`Unknown option --${name}`);
                        options[keys[name]] = ["quality", "width", "height", "retries", "virtualTimeBudget"].includes(keys[name])
                            ? Number(value) : (["true", "false"].includes(value) ? value === "true" : value);
                    }
    }
    return { positional, options };
}

function command(command, args, { input, timeout } = {}) {
    return new Promise((resolveCommand, reject) => {
        const child = spawn(command, args, { stdio: [input ? "pipe" : "ignore", "pipe", "pipe"] });
        let stdout = "", stderr = "", timedOut = false;
        const timer = timeout ? setTimeout(() => { timedOut = true; child.kill("SIGTERM"); }, timeout) : null;
        child.stdout.on("data", (chunk) => { stdout += chunk; });
        child.stderr.on("data", (chunk) => { stderr += chunk; });
        child.on("error", reject);
        child.on("close", (code) => {
            if (timer) clearTimeout(timer);
            if (timedOut) reject(new Error(`${command} timed out after ${timeout} ms`));
            else if (code === 0) resolveCommand({ stdout, stderr });
            else reject(new Error(`${command} exited with ${code}: ${stderr.trim()}`));
        });
        if (input) child.stdin.end(input);
    });
}

function formatSize(bytes) {
    for (const unit of ["B", "KB", "MB", "GB"]) { if (bytes < 1024) return `${bytes.toFixed(2)} ${unit}`; bytes /= 1024; }
    return `${bytes.toFixed(2)} TB`;
}

function placeholderKey(kind, source) {
    const key = `${kind}-${source}`.replace(/[\\/:*?"<>|#%&{}$!@+=` ]/g, "-");
    return key || createHash("sha1").update(`${kind}:${source}`).digest("hex");
}

function isUrl(value) { return /^(https?|file|data):/i.test(value); }
function localTarget(value, baseDir) { return isUrl(value) ? value : resolve(baseDir, value); }

function splitTypstCalls(source, functionName) {
    const calls = [], marker = `${functionName}(`;
    let start = 0;
    while ((start = source.indexOf(marker, start)) !== -1) {
        let pos = start + marker.length, depth = 1, quote = null, escaped = false;
        for (; pos < source.length && depth; pos += 1) {
            const char = source[pos];
            if (quote) { if (escaped) escaped = false; else if (char === "\\") escaped = true; else if (char === quote) quote = null; }
            else if (char === "\"" || char === "'") quote = char;
                else if ("([{".includes(char)) depth += 1;
                    else if (")] }".replace(" ", "").includes(char)) depth -= 1;
        }
        if (!depth) { calls.push(source.slice(start + marker.length, pos - 1)); start = pos; }
        else start += marker.length;
    }
    return calls;
}

function typstStringArg(call, name) {
    const match = call.match(new RegExp(`\\b${name}\\s*:\\s*("(?:\\\\.|[^\\"])*")`));
    if (!match) return null;
    try { return JSON.parse(match[1]); } catch { return match[1].slice(1, -1); }
}

function videoWrapper(src, baseDir, width, height) {
    let target = localTarget(src, baseDir);
    if (!isUrl(target)) target = pathToFileURL(target).href;
    return `<!doctype html><html><head><meta charset="utf-8"><style>html,body{width:${width}px;height:${height}px;margin:0;background:#111827;overflow:hidden}video{width:100vw;height:100vh;object-fit:contain;background:#111827}</style></head><body><video src="${target.replaceAll("&", "&amp;").replaceAll('"', "&quot;")}" controls muted autoplay playsinline></video></body></html>`;
}

async function discoverPlaceholderTargets(sourcePath, folder, width, height) {
    const source = await readFile(sourcePath, "utf8"), baseDir = dirname(sourcePath), targets = new Map();
    const add = (kind, value, target) => { const key = `${kind}\0${value}`; if (!targets.has(key)) targets.set(key, { kind, source: value, target, output: join(folder, `${placeholderKey(kind, value)}.png`) }); };
    for (const call of splitTypstCalls(source, "embed-html-file")) { const path = typstStringArg(call, "path"); if (path) add("path", path, localTarget(path, baseDir)); }
    for (const call of splitTypstCalls(source, "embed-html")) { const src = typstStringArg(call, "src"), srcdoc = typstStringArg(call, "srcdoc"); if (src) add("src", src, localTarget(src, baseDir)); else if (srcdoc) add("srcdoc", srcdoc, srcdoc); }
    for (const call of splitTypstCalls(source, "embed-video")) { const src = typstStringArg(call, "src"); if (src) add("video", src, videoWrapper(src, baseDir, width, height)); }
    return [...targets.values()];
}

function findBrowser(requested) {
    const names = requested ? [requested] : ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome", "microsoft-edge", "brave-browser", "firefox"];
    for (const name of names) { const found = spawnSync("which", [name], { encoding: "utf8" }); if (found.status === 0) return { name: basename(found.stdout.trim()), path: found.stdout.trim() }; }
    throw new Error(requested ? `Browser '${requested}' was not found` : "No supported browser found. Install Chromium/Chrome, or pass --browser.");
}

function findExecutable(names) {
    for (const name of names) {
        const found = spawnSync("which", [name], { encoding: "utf8" });
        if (found.status === 0) return found.stdout.trim();
    }
    return null;
}

async function placeholders(source, options) {
    const sourcePath = resolve(source), folder = resolve(dirname(sourcePath), options.placeholderFolder ?? ".typstslideviewer-placeholders");
    const width = options.width ?? 1200, height = options.height ?? 800, budget = options.virtualTimeBudget ?? 6000;
    const targets = await discoverPlaceholderTargets(sourcePath, folder, width, height);
    if (!targets.length) { console.warn("No literal html embed targets found"); return []; }
    const browser = findBrowser(options.browser); await mkdir(folder, { recursive: true });
    for (const target of targets) {
        try { if (!options.force) { await stat(target.output); console.log(`Skipping existing placeholder: ${target.output}`); continue; } } catch {}
        let browserTarget = target.target, temporary;
        if (/^\s*<(?:!doctype|html)/i.test(browserTarget)) { temporary = join(folder, `.typstslideviewer-${Date.now()}.html`); await writeFile(temporary, browserTarget); browserTarget = pathToFileURL(temporary).href; }
        else if (!isUrl(browserTarget)) browserTarget = pathToFileURL(browserTarget).href;
        const args = browser.name.startsWith("firefox") ? ["--headless", "--window-size", `${width},${height}`, "--screenshot", target.output, browserTarget] : ["--headless=new", "--no-sandbox", `--window-size=${width},${height}`, `--virtual-time-budget=${budget}`, `--screenshot=${target.output}`, browserTarget];
        try { await command(browser.path, args); } finally { if (temporary) await rm(temporary, { force: true }); }
    }
    const manifest = { placeholder_dir: folder, typst_input: options.placeholderFolder ? `html-placeholder-dir=${options.placeholderFolder}` : null, targets: targets.map(({ kind, source: targetSource, output }) => ({ kind, source: targetSource, output })) };
    await writeFile(join(folder, "manifest.json"), JSON.stringify(manifest, null, 2));
    return manifest.targets;
}

function initPages(rawPages) {
    const pages = Object.fromEntries(rawPages.map((page) => [page.idx, {
        ...page,
        idx: Number(page.idx),
        label: Number(page.label),
        note: page.note ?? "",
    }]));
    for (const [key, page] of Object.entries(pages)) if (page.note) for (let i = Number(key) - 1; i >= 0 && pages[i]?.label === page.label && !pages[i].note; i -= 1) pages[i].note = page.note;
    return pages;
}

async function initMetaInfo(file, slideCount) {
    let meta;
    try { const queried = JSON.parse(await readFile(file, "utf8")); meta = queried.pages ? queried : queried[0]; } catch { meta = { pages: Array.from({ length: slideCount }, (_, idx) => ({ idx, label: idx + 1, forcedOverlay: false, hidden: false })) }; }
    const pages = initPages(meta.pages), jumpMap = {}, thumbnailMap = {};
    for (const page of Object.values(pages)) { const label = Number(page.label); jumpMap[label] = Math.min(jumpMap[label] ?? Object.keys(pages).length + 1, page.idx); thumbnailMap[label] = Math.max(0, page.idx); }
    return { pages, jumpMap, thumbnailMap };
}

function attribute(markup, name) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const match = markup.match(new RegExp(`\\b${escaped}\\s*=\\s*(["'])(.*?)\\1`, "i"));
    return match?.[2];
}

function svgNumber(value, fallback = 0) {
    const match = String(value ?? "").match(/[+-]?(?:\d+(?:\.\d*)?|\.\d+)/);
    return match ? Number(match[0]) : fallback;
}

async function inlineForeignObjects(svg, baseDir) {
    let result = "", offset = 0;
    for (const image of svg.matchAll(/<image\b[^>]*\/?\s*>/gi)) {
        const href = attribute(image[0], "xlink:href") ?? attribute(image[0], "href");
        if (!href?.startsWith("data:image/svg+xml;base64,")) continue;
        let inner;
        try { inner = Buffer.from(href.slice(href.indexOf(",") + 1), "base64").toString("utf8"); } catch { continue; }
        const foreignObject = inner.match(/<foreignObject\b[\s\S]*?<\/foreignObject>/i)?.[0];
        if (!foreignObject) continue;

        let replacement = foreignObject;
        const source = replacement.match(/<source\b[^>]*\bsrc\s*=\s*(["'])(.*?)\1[^>]*>/i);
        if (source && /<video\b/i.test(replacement)) {
            const mimeType = attribute(source[0], "type") ?? "video/mp4";
            try {
                const video = await readFile(resolve(baseDir, source[2]));
                replacement = replacement.replace(source[0], source[0].replace(source[2], `data:${mimeType};base64,${video.toString("base64")}`));
            } catch (error) {
                console.warn(`Could not embed video '${source[2]}': ${error.message}`);
            }
        }

        const root = inner.match(/<svg\b[^>]*>/i)?.[0];
        const viewBox = attribute(root ?? "", "viewBox")?.trim().split(/[\s,]+/).map(Number);
        const width = svgNumber(attribute(image[0], "width")), height = svgNumber(attribute(image[0], "height"));
        if (viewBox?.length === 4 && viewBox[2] && viewBox[3] && width && height) {
            const transforms = [attribute(image[0], "transform"),
                ...(attribute(image[0], "x") || attribute(image[0], "y") ? [`translate(${svgNumber(attribute(image[0], "x"))} ${svgNumber(attribute(image[0], "y"))})`] : []),
                `scale(${width / viewBox[2]} ${height / viewBox[3]})`,
                ...(viewBox[0] || viewBox[1] ? [`translate(${-viewBox[0]} ${-viewBox[1]})`] : [])].filter(Boolean);
            replacement = `<g transform="${transforms.join(" ")}">${replacement}</g>`;
        }
        result += svg.slice(offset, image.index) + replacement;
        offset = image.index + image[0].length;
    }
    return result ? result + svg.slice(offset) : svg;
}

export async function optimizeSvg(svg, baseDir = process.cwd()) {
    svg = await inlineForeignObjects(svg, baseDir);
    return svg.replace(/<svg\b([^>]*)>/i, (_, attributes) => `<svg${attributes.replace(/\s(?:width|height)="[^"]*"/g, "")}>`)
        .replace(/(<\/?)(?:[\w-]+:)/g, "$1")
        .replace(/\sxmlns(?::\w+)?="[^"]*"/g, "");
}

function packSvgs(slides) {
    const counts = new Map();
    for (const svg of Object.values(slides)) for (const uri of svg.match(DATA_URI) ?? []) counts.set(uri, (counts.get(uri) ?? 0) + 1);
    const assets = {}, tokens = new Map(); let index = 0;
    for (const [uri, count] of [...counts].sort((a, b) => b[0].length - a[0].length)) if (count > 1 && Buffer.byteLength(uri) >= 1024) { const token = `@@TSV_ASSET_${index++}@@`; assets[token] = uri; tokens.set(uri, token); }
    const packed = Object.fromEntries(Object.entries(slides).map(([id, svg]) => [id, [...tokens].reduce((text, [uri, token]) => text.replaceAll(uri, token), svg)]));
    return { slides: packed, assets };
}

function cropSvgForThumbnail(svg, note) {
    if (!note) return svg;
    return svg.replace(/<svg\b[^>]*>/i, (root) => {
        const values = attribute(root, "viewBox")?.trim().split(/[\s,]+/).map(Number);
        if (values?.length !== 4) return root;
        const [minX, minY, width, height] = values;
        return root.replace(/\bviewBox\s*=\s*(["']).*?\1/i, `viewBox="${minX} ${minY} ${width / 2.001} ${height}"`);
    });
}

function thumbnailGeometry(svg, width) {
    const root = svg.match(/<svg\b[^>]*>/i)?.[0] ?? "";
    const values = attribute(root, "viewBox")?.trim().split(/[\s,]+/).map(Number);
    const ratio = values?.length === 4 && values[2] > 0 && values[3] > 0 ? values[3] / values[2] : 9 / 16;
    return { width, height: Math.max(1, Math.round(width * ratio)) };
}

async function generateThumbnails(slides, meta, note, options) {
    const converter = findExecutable(["magick", "convert"]);
    const svgRenderer = findExecutable(["rsvg-convert"]);
    if (!converter) throw new Error("ImageMagick was not found");

    const width = 300;
    const items = Object.entries(meta.thumbnailMap).map(([page, index]) => {
        const svg = cropSvgForThumbnail(slides[index], note);
        const geometry = thumbnailGeometry(svg, width);
        return { page, index, svg, ...geometry };
    });
    if (!items.length) return {};
    const browser = items.some((item) => /<foreignObject\b/i.test(item.svg)) ? findBrowser(options.browser) : null;

    const folder = await mkdtemp(join(tmpdir(), "typstslideviewer-thumbnails-"));
    try {
        const budget = options.virtualTimeBudget ?? 6000;
        const thumbnails = {};
        for (const item of items) {
            try {
                const screenshot = join(folder, `thumbnail-${item.page}.png`);
                const output = join(folder, `thumbnail-${item.page}.webp`);
                if (/<foreignObject\b/i.test(item.svg)) {
                    const source = join(folder, `thumbnail-${item.page}.html`);
                    const html = `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;width:${item.width}px;height:${item.height}px;overflow:hidden;background:#000}svg{display:block;width:100%;height:100%}</style></head><body>${item.svg}</body></html>`;
                    await writeFile(source, html);
                    const target = pathToFileURL(source).href;
                    const browserArgs = browser.name.startsWith("firefox")
                        ? ["--headless", "--window-size", `${item.width},${item.height}`, "--screenshot", screenshot, target]
                        : ["--headless=new", "--no-sandbox", "--allow-file-access-from-files", "--hide-scrollbars", "--force-device-scale-factor=1", "--enable-webgl", "--ignore-gpu-blocklist", `--window-size=${item.width},${item.height}`, `--timeout=${budget}`, `--screenshot=${screenshot}`, target];
                    await command(browser.path, browserArgs, { timeout: budget + 10000 });
                } else {
                    const source = join(folder, `thumbnail-${item.page}.svg`);
                    const standaloneSvg = item.svg.replace(/<svg\b([^>]*)>/i, (_, attributes) => `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"${attributes}>`);
                    await writeFile(source, standaloneSvg);
                    if (svgRenderer) await command(svgRenderer, ["--width", String(item.width), "--height", String(item.height), "--output", screenshot, source]);
                    else await command(converter, [source, "-resize", `${item.width}x${item.height}!`, screenshot]);
                }
                const convertArgs = [screenshot, "-crop", `${item.width}x${item.height}+0+0`, "+repage", "-quality", String(options.quality), output];
                await command(converter, convertArgs);
                thumbnails[item.page] = `data:image/webp;base64,${(await readFile(output)).toString("base64")}`;
            } catch (error) {
                console.warn(`Could not pre-render thumbnail ${item.page}; using browser fallback: ${error.message}`);
            }
        }
        console.log(`Pre-rendered ${Object.keys(thumbnails).length} WebP thumbnails`);
        return thumbnails;
    } finally {
        await rm(folder, { recursive: true, force: true });
    }
}

async function renderViewer({ slides, thumbnails, meta, note, transition, templateFile }) {
    const [template, fzstd, tar] = await Promise.all([readFile(templateFile ?? join(ASSETS, "viewer.template.html"), "utf8"), readFile(FZSTD_BUNDLE, "utf8"), readFile(TARTS_BUNDLE, "utf8")]);
    const packed = { ...packSvgs(slides), thumbnails }, source = Buffer.from(JSON.stringify(packed)), compressed = zstdCompressSync(source, { params: { [zlibConstants.ZSTD_c_compressionLevel]: 4, [zlibConstants.ZSTD_c_windowLog]: 29 } }).toString("base64");
    console.log(`Compressed data size: ${formatSize(Buffer.byteLength(compressed, "base64"))}`);
    const replacements = { fzstd_js: fzstd, tar_js: `${tar}\nconst Tar = globalThis.tarts;`, total_files: Object.keys(slides).length, jump_map: JSON.stringify(meta.jumpMap), thumbnail_map: JSON.stringify(meta.thumbnailMap), pages: JSON.stringify(meta.pages), no_animation: String(!transition), slide_64: JSON.stringify(compressed), note: JSON.stringify(note) };
    return template.replace(/\{\{\s*(fzstd_js|tar_js|total_files|jump_map \| tojson|thumbnail_map \| tojson|pages \| tojson|no_animation \| lower|slide_64 \| tojson|note \| tojson)\s*\}\}/g, (_, token) => replacements[token.replace(/ \|.*/, "")]);
}

async function generate(source, options) {
    const sourcePath = resolve(source), svgFolder = resolve(options.svgFolder), output = resolve(options.outputFile ?? sourcePath.replace(/\.[^.]+$/, ".html"));
    let rebuild = options.force;
    try { rebuild ||= (await stat(sourcePath)).mtimeMs > (await stat(join(svgFolder, "meta.json"))).mtimeMs; } catch { rebuild = true; }
    if (rebuild) {
        await mkdir(svgFolder, { recursive: true });
        for (const entry of await readdir(svgFolder, { withFileTypes: true })) if (entry.isFile()) await rm(join(svgFolder, entry.name));
        await command("typst", ["compile", "--input", "html-embed-mode=iframe", sourcePath, join(svgFolder, "slide_{0p}.svg")]);
        let query = "[]"; try { ({ stdout: query } = await command("typst", ["query", "--input", "html-embed-mode=iframe", sourcePath, "--field", "value", "<pdfpc-file>"])); } catch (error) { console.warn(`Typst metadata query failed; using sequential pages: ${error.message}`); }
        const files = (await readdir(svgFolder)).filter((name) => /^slide_.*\.svg$/.test(name)).sort();
        await Promise.all(files.map(async (file) => writeFile(join(svgFolder, `modified_${file}`), await optimizeSvg(await readFile(join(svgFolder, file), "utf8"), dirname(sourcePath)))));
        await writeFile(join(svgFolder, "meta.json"), query);
    }
    const files = (await readdir(svgFolder)).filter((name) => /^modified_.*\.svg$/.test(name)).sort();
    if (!files.length) throw new Error(`No SVG files found in the '${options.svgFolder}' folder`);
    const slides = Object.fromEntries(await Promise.all(files.map(async (file) => [Number(file.match(/(\d+)\.svg$/)[1]) - 1, await readFile(join(svgFolder, file), "utf8")] )));
    const meta = await initMetaInfo(join(svgFolder, "meta.json"), files.length);
    let thumbnails = {};
    if (options.thumbnails) {
        try { thumbnails = await generateThumbnails(slides, meta, options.note, options); }
        catch (error) { console.warn(`Could not pre-render thumbnails; using browser fallback: ${error.message}`); }
    }
    await writeFile(output, await renderViewer({ slides, thumbnails, meta, note: options.note, transition: options.transition, templateFile: options.templateFile }));
    console.log(`Wrote ${output}`);
}

async function writeHtmlEmbed(options) {
    let output = resolve(options.output ?? "html-embed.typ");
    try { if ((await stat(output)).isDirectory()) output = join(output, "html-embed.typ"); } catch {}
    try { await stat(output); if (!options.force) throw new Error(`Refusing to overwrite '${output}'. Pass --force to replace it.`); } catch (error) { if (error.code !== "ENOENT" && !error.message.startsWith("Refusing")) throw error; if (error.message.startsWith("Refusing")) throw error; }
    await mkdir(dirname(output), { recursive: true }); await copyFile(join(ASSETS, "html-embed.typ"), output); console.log(`Wrote HTML embed helper: ${output}`);
}

export async function main(argv) {
    const { positional, options } = parseArgs(argv);
    if (options.help || !positional.length) { console.log(HELP); return; }
    const [subcommand, source] = positional;
    if (subcommand === "html-embed") return writeHtmlEmbed(options);
    if (subcommand === "placeholders") { if (!source) throw new Error("placeholders requires a Typst source path"); return placeholders(source, options); }
    return generate(subcommand, options);
}
