import base64
import concurrent.futures
import hashlib
import html
import importlib.resources
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import urlparse

import fire
import jinja2
import zstandard as zstd
from loguru import logger
from PIL import Image, ImageStat
from pydantic import BaseModel, ConfigDict


class Page(BaseModel):
    model_config = ConfigDict(extra="ignore")
    idx: int
    label: int
    forcedOverlay: bool
    hidden: bool
    note: str = ""


class MetaInfo(BaseModel):
    jump_map: dict[int, int]
    thumbnail_map: dict[int, int]
    pages: dict[int, Page]


class PlaceholderTarget(BaseModel):
    kind: Literal["src", "path", "srcdoc", "video"]
    source: str
    target: str
    output: Path


def format_size(size: Union[int, float]) -> str:
    # Convert bytes to a human-readable format
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def placeholder_image_key(kind: str, source: str) -> str:
    key = f"{kind}-{source}"
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '#', '%', '&', '{', '}', '$', '!', '@', '+', '=', '`', ' ']:
        key = key.replace(char, "-")
    return key or hashlib.sha1(f"{kind}:{source}".encode("utf-8")).hexdigest()


def find_browser(browser: Optional[str] = None) -> tuple[str, str]:
    if browser is not None:
        browser_path = shutil.which(browser) or browser
        if Path(browser_path).exists():
            browser_name = Path(browser_path).name
            return browser_name, browser_path
        raise FileNotFoundError(f"Browser '{browser}' was not found")

    for browser_name in [
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
        "microsoft-edge",
        "microsoft-edge-stable",
        "brave-browser",
        "firefox",
    ]:
        browser_path = shutil.which(browser_name)
        if browser_path is not None:
            return browser_name, browser_path

    raise FileNotFoundError(
        "No supported browser found. Install Chromium/Chrome, or pass --browser."
    )


def split_typst_calls(source: str, function_name: str) -> list[str]:
    calls = []
    pattern = f"{function_name}("
    start = 0
    while True:
        index = source.find(pattern, start)
        if index < 0:
            break
        pos = index + len(pattern)
        depth = 1
        quote = None
        escaped = False
        while pos < len(source) and depth > 0:
            char = source[pos]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            pos += 1
        if depth == 0:
            calls.append(source[index + len(pattern): pos - 1])
            start = pos
        else:
            start = index + len(pattern)
    return calls


def typst_string_arg(call: str, name: str) -> Optional[str]:
    match = re.search(rf"\b{name}\s*:\s*(\"(?:\\.|[^\"])*\")", call)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return match.group(1)[1:-1]


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "file", "data"}


def local_target(value: str, base_dir: Path) -> str:
    if is_url(value):
        return value
    return str((base_dir / value).resolve())


def video_wrapper(src: str, base_dir: Path, width: int, height: int) -> str:
    target = local_target(src, base_dir)
    if not is_url(target):
        target = Path(target).as_uri()
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      html, body {{
        width: {width}px;
        height: {height}px;
        margin: 0;
        background: #111827;
        overflow: hidden;
      }}
      video {{
        width: 100vw;
        height: 100vh;
        object-fit: contain;
        background: #111827;
      }}
    </style>
  </head>
  <body>
    <video src="{html.escape(target, quote=True)}" controls muted autoplay playsinline></video>
  </body>
</html>
"""


def discover_placeholder_targets(
    typst_src: Union[str, Path],
    placeholder_folder: Union[str, Path],
    width: int,
    height: int,
) -> list[PlaceholderTarget]:
    typst_src_path = Path(typst_src)
    base_dir = typst_src_path.parent
    placeholder_folder_path = Path(placeholder_folder)
    source = typst_src_path.read_text()
    targets: dict[tuple[str, str], PlaceholderTarget] = {}

    def add(kind: Literal["src", "path", "srcdoc", "video"], value: str, target: str):
        key = (kind, value)
        if key in targets:
            return
        output = placeholder_folder_path / f"{placeholder_image_key(kind, value)}.png"
        targets[key] = PlaceholderTarget(
            kind=kind,
            source=value,
            target=target,
            output=output,
        )

    for call in split_typst_calls(source, "embed-html-file"):
        path = typst_string_arg(call, "path")
        if path is not None:
            add("path", path, local_target(path, base_dir))

    for call in split_typst_calls(source, "embed-html"):
        src = typst_string_arg(call, "src")
        srcdoc = typst_string_arg(call, "srcdoc")
        if src is not None:
            add("src", src, local_target(src, base_dir))
        elif srcdoc is not None:
            add("srcdoc", srcdoc, srcdoc)

    for call in split_typst_calls(source, "embed-video"):
        src = typst_string_arg(call, "src")
        if src is not None:
            add("video", src, video_wrapper(src, base_dir, width, height))

    return list(targets.values())


def screenshot_with_browser(
    browser_name: str,
    browser_path: str,
    target: str,
    output: Union[str, Path],
    width: int,
    height: int,
    virtual_time_budget: int,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = None
    browser_target = target
    if target.lstrip().lower().startswith("<!doctype") or target.lstrip().lower().startswith("<html"):
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", encoding="utf-8", delete=False
        )
        temp_file.write(target)
        temp_file.close()
        browser_target = Path(temp_file.name).as_uri()
    elif not is_url(target):
        browser_target = Path(target).resolve().as_uri()

    try:
        if browser_name.startswith("firefox"):
            command = [
                browser_path,
                "--headless",
                "--window-size",
                f"{width},{height}",
                "--screenshot",
                str(output_path),
                browser_target,
            ]
        else:
            command = [
                browser_path,
                "--headless=new",
                "--no-sandbox",
                "--use-gl=swiftshader",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
                "--run-all-compositor-stages-before-draw",
                f"--window-size={width},{height}",
                f"--virtual-time-budget={virtual_time_budget}",
                f"--screenshot={output_path}",
                browser_target,
            ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Browser screenshot failed for {browser_target}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    finally:
        if temp_file is not None:
            Path(temp_file.name).unlink(missing_ok=True)


def is_blank_image(path: Union[str, Path], threshold: float = 1.0) -> bool:
    with Image.open(path) as image:
        stat = ImageStat.Stat(image.convert("RGB"))
    return max(stat.stddev) < threshold


def screenshot_with_retries(
    browser_name: str,
    browser_path: str,
    target: str,
    output: Union[str, Path],
    width: int,
    height: int,
    virtual_time_budget: int,
    retries: int,
) -> None:
    for attempt in range(retries + 1):
        budget = virtual_time_budget * (attempt + 1)
        screenshot_with_browser(
            browser_name,
            browser_path,
            target,
            output,
            width,
            height,
            budget,
        )
        if not is_blank_image(output):
            return
        logger.warning(
            f"Blank placeholder screenshot for {output}; retrying"
            if attempt < retries
            else f"Blank placeholder screenshot for {output}"
        )


def generate_placeholder_images(
    typst_src: str,
    placeholder_folder: str = ".typstslideviewer-placeholders",
    browser: Optional[str] = None,
    width: int = 1200,
    height: int = 800,
    virtual_time_budget: int = 6000,
    retries: int = 2,
    force: bool = False,
):
    """
    Generate browser screenshots for html-embed placeholders.

    The default placeholder folder is picked up by html-embed.typ automatically.
    If you use a custom folder, compile Typst with:
    --input html-placeholder-dir=<placeholder_folder>
    """
    typst_src_path = Path(typst_src)
    placeholder_folder_path = Path(placeholder_folder)
    if not placeholder_folder_path.is_absolute():
        placeholder_folder_path = typst_src_path.parent / placeholder_folder_path

    browser_name, browser_path = find_browser(browser)
    targets = discover_placeholder_targets(
        typst_src_path,
        placeholder_folder_path,
        width,
        height,
    )
    if not targets:
        logger.warning("No literal html embed targets found")
        return []

    logger.info(f"Using browser: {browser_path}")
    for target in targets:
        if target.output.exists() and not force:
            logger.info(f"Skipping existing placeholder: {target.output}")
            continue
        logger.info(f"Generating placeholder for {target.kind}: {target.source}")
        screenshot_with_retries(
            browser_name,
            browser_path,
            target.target,
            target.output,
            width,
            height,
            virtual_time_budget,
            retries,
        )

    manifest = {
        "placeholder_dir": str(placeholder_folder_path),
        "typst_input": (
            None
            if placeholder_folder == ".typstslideviewer-placeholders"
            else f"html-placeholder-dir={placeholder_folder}"
        ),
        "targets": [
            {
                "kind": target.kind,
                "source": target.source,
                "output": str(target.output),
            }
            for target in targets
        ],
    }
    manifest_path = placeholder_folder_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote placeholder manifest: {manifest_path}")
    if placeholder_folder == ".typstslideviewer-placeholders":
        logger.info(f"Use with Typst fallback output: typst compile {typst_src_path}")
    else:
        logger.info(
            "Use with Typst fallback output: "
            f"typst compile --input html-placeholder-dir={placeholder_folder} "
            f"{typst_src_path}"
        )
    return manifest["targets"]


def init_pages(raw_pages):
    pages = {i["idx"]: Page.model_validate(i) for i in raw_pages}
    for k in pages.keys():
        p = pages[k]
        if len(p.note) > 0:
            for i in range(k - 1, -1, -1):
                if pages[i].label != p.label:
                    break
                if len(pages[i].note) > 0:
                    break
                pages[i].note = p.note
    return pages


def init_meta_info(fn):
    with open(fn) as fp:
        meta = json.load(fp)

    pages = init_pages(meta["pages"])
    jump_map = {}
    thumbnail_map = {}

    # Iterate over the pages to populate the maps
    for page in pages.values():
        page_label = int(page.label)
        jump_map[page_label] = min(jump_map.get(page_label, len(pages) + 1), page.idx)
        thumbnail_map[page_label] = max(0, page.idx)

    return MetaInfo(jump_map=jump_map, thumbnail_map=thumbnail_map, pages=pages)


def decode_base64_data(data):
    # Remove the prefix 'data:image/svg+xml;base64,' and decode the base64 data
    base64_data = data.split(",")[1]
    decoded_data = base64.b64decode(base64_data)
    return decoded_data


def strip_namespace(element):
    """
    Remove namespace prefixes from tags and attributes in the element and its children.
    """
    for elem in element.iter():
        # Remove the namespace prefix from tag
        if isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

        # Handle attributes with namespaces
        new_attrs = {}
        for key, value in elem.attrib.items():
            if isinstance(key, str) and "}" in key:
                new_key = key.split("}", 1)[1]
                new_attrs[new_key] = value
            else:
                new_attrs[key] = value
        elem.attrib = new_attrs


def parse_svg_number(value):
    if value is None:
        return None
    match = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", str(value))
    if match is None:
        return None
    return float(match.group(1))


def parse_view_box(value):
    if value is None:
        return None
    parts = re.split(r"[\s,]+", value.strip())
    if len(parts) != 4:
        return None
    try:
        return tuple(float(part) for part in parts)
    except ValueError:
        return None


@lru_cache
def convert_to_webp(img_base64, quality=90):
    data = decode_base64_data(img_base64)
    image = Image.open(io.BytesIO(data))
    webp_io = io.BytesIO()
    image.save(webp_io, format="WEBP", optimize=True, quality=quality)
    return (
        f"data:image/webp;base64,{base64.b64encode(webp_io.getvalue()).decode('utf-8')}"
    )


@lru_cache
def convert_to_avif(img_base64, quality=90):
    data = decode_base64_data(img_base64)
    image = Image.open(io.BytesIO(data))
    avif_io = io.BytesIO()
    image.save(avif_io, format="AVIF", optimize=True, quality=quality)
    return (
        f"data:image/avif;base64,{base64.b64encode(avif_io.getvalue()).decode('utf-8')}"
    )


class SVGOptimizer(BaseModel):
    optimize_png: bool
    optimize_jpg: bool
    output_format: Literal["webp", "avif"] = "webp"
    quality: int = 80

    def inline_foreignObject(self, tree):
        """
        Detects SVG-in-SVG data URIs used for foreignObject content (like video or xhtml)
        and inlines them as true foreignObject elements while preserving positioning.
        """
        parent_map = {c: p for p in tree.iter() for c in p}
        root = tree.getroot()
        namespaces = {"xlink": "http://www.w3.org/1999/xlink"}
        svg_ns = "http://www.w3.org/2000/svg"

        for image in root.findall(".//{http://www.w3.org/2000/svg}image", namespaces):
            href = image.get("{http://www.w3.org/1999/xlink}href")
            if href and href.startswith("data:image/svg+xml;base64,"):
                decoded_data = decode_base64_data(href)
                # Parse decoded SVG
                inner_tree = ET.fromstring(decoded_data.decode("utf-8"))
                inner_view_box = parse_view_box(inner_tree.get("viewBox"))

                # Find foreignObject in the decoded SVG
                foreign_object = inner_tree.find(
                    ".//{http://www.w3.org/2000/svg}foreignObject"
                )
                if foreign_object is None:
                    foreign_object = inner_tree.find(".//foreignObject")

                if foreign_object is not None:
                    # Specific handling for video elements
                    children = list(foreign_object)
                    if (
                        len(children) == 1
                        and (children[0].tag == "{http://www.w3.org/1999/xhtml}video" or children[0].tag.endswith("video"))
                    ):
                        video_element = children[0]
                        video_element.set("width", "100%")
                        video_element.set("height", "100%")
                        video_element.set("style", "width: 100%; height: 100%;")
                        
                        source_element = video_element.find("{http://www.w3.org/1999/xhtml}source")
                        if source_element is None:
                             source_element = video_element.find("source")
                             
                        if source_element is not None:
                            src = source_element.attrib.get("src")
                            if src and src.endswith(".mp4"):
                                video_path = Path(src)
                                if video_path.exists():
                                    with open(video_path, "rb") as video_file:
                                        base64_video = base64.b64encode(video_file.read()).decode("utf-8")
                                        source_element.attrib["src"] = f"data:video/mp4;base64,{base64_video}"

                    parent = parent_map[image]
                    image_index = list(parent).index(image)

                    image_width = parse_svg_number(image.get("width"))
                    image_height = parse_svg_number(image.get("height"))
                    if (
                        inner_view_box is not None
                        and image_width
                        and image_height
                        and inner_view_box[2]
                        and inner_view_box[3]
                    ):
                        min_x, min_y, view_width, view_height = inner_view_box
                        scale_x = image_width / view_width
                        scale_y = image_height / view_height
                        transforms = []

                        image_transform = image.get("transform")
                        if image_transform:
                            transforms.append(image_transform)

                        image_x = parse_svg_number(image.get("x")) or 0
                        image_y = parse_svg_number(image.get("y")) or 0
                        if image_x != 0 or image_y != 0:
                            transforms.append(f"translate({image_x:g} {image_y:g})")

                        transforms.append(f"scale({scale_x:g} {scale_y:g})")
                        if min_x != 0 or min_y != 0:
                            transforms.append(f"translate({-min_x:g} {-min_y:g})")

                        wrapper = ET.Element(f"{{{svg_ns}}}g")
                        wrapper.set("transform", " ".join(transforms))
                        wrapper.append(foreign_object)
                        replacement = wrapper
                    else:
                        # Fall back to the old behavior when the embedded SVG has no usable viewBox.
                        foreign_object.attrib.clear()
                        for attr_name, attr_value in image.attrib.items():
                            if attr_name != "{http://www.w3.org/1999/xlink}href":
                                foreign_object.set(attr_name, attr_value)
                        replacement = foreign_object

                    parent.remove(image)
                    parent.insert(image_index, replacement)

    def _optimize_bitmap(self, tree):
        root = tree.getroot()
        namespaces = {"xlink": "http://www.w3.org/1999/xlink"}

        if self.output_format == "webp":
            convert_func = convert_to_webp
        elif self.output_format == "avif":
            convert_func = convert_to_avif
        else:
            raise ValueError(f"Invalid output format: {self.output_format}")
            
        # Find all <image> elements with xlink:href attribute
        for image in root.findall(".//{http://www.w3.org/2000/svg}image", namespaces):
            href = image.get("{http://www.w3.org/1999/xlink}href")
            if href is None:
                continue
            elif self.optimize_png and href.startswith("data:image/png;base64,"):
                image.attrib["{http://www.w3.org/1999/xlink}href"] = convert_func(href, self.quality)
            elif self.optimize_jpg and href.startswith("data:image/jpeg;base64,"):
                image.attrib["{http://www.w3.org/1999/xlink}href"] = convert_func(href, self.quality)
            elif href.startswith("data:image/svg+xml;base64,"):
                t = ET.ElementTree(ET.fromstring(decode_base64_data(href).decode("utf-8")))
                self._optimize_bitmap(t)
                data = ET.tostring(t.getroot(), encoding="unicode", short_empty_elements=False)
                image.attrib["{http://www.w3.org/1999/xlink}href"] = (
                    f"data:image/svg+xml;base64,{base64.b64encode(data.encode('utf-8')).decode('utf-8')}"
                )

    def optimize(self, svgstring):
        tree = ET.ElementTree(ET.fromstring(svgstring))
        root = tree.getroot()
        self._optimize_bitmap(tree)
        self.inline_foreignObject(tree)

        # Remove namespaces for cleaner output
        strip_namespace(root)
        
        # Remove width and height from the root <svg> element to let it be responsive
        if "width" in root.attrib:
            del root.attrib["width"]
        if "height" in root.attrib:
            del root.attrib["height"]
            
        return ET.tostring(root, encoding="unicode", short_empty_elements=False)


def process_file(f, svg_folder_path, optimizer):
    svgstring = f.read_text()
    with (svg_folder_path / f"modified_{f.name}").open("w") as fp:
        fp.write(optimizer.optimize(svgstring))


def init_svg_folder(typst_src, svg_folder, optimizer: SVGOptimizer):
    logger.info("Initializing SVG folder")
    svg_folder_path = Path(svg_folder)
    typst_src_path = Path(typst_src)
    if not svg_folder_path.exists():
        svg_folder_path.mkdir(parents=True)
    for file in svg_folder_path.glob("*"):
        if file.is_file():
            file.unlink()

    logger.info(f"Compiling Typst source file: {typst_src_path}")
    compile_process = subprocess.Popen(
        [
            "typst",
            "compile",
            "--input",
            "html-embed-mode=iframe",
            typst_src_path,
            f"{svg_folder_path}/slide_{{0p}}.svg",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info(f"Querying Typst source file: {typst_src_path}")
    query_process = subprocess.Popen(
        [
            "typst",
            "query",
            "--input",
            "html-embed-mode=iframe",
            typst_src_path,
            "--field",
            "value",
            "<pdfpc-file>",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    compile_process.communicate()
    if compile_process.returncode != 0:
        logger.error("Compilation failed")
        raise Exception(compile_process.stderr.read())

    logger.info("Compilation successful, processing SVG files")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    slide_fns = list(svg_folder_path.rglob("slide_*.svg"))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, f, svg_folder_path, optimizer)
            for f in slide_fns
        ]
        concurrent.futures.wait(futures)

    query_stdout, query_stderr = query_process.communicate()

    meta_path = svg_folder_path / "meta.json"
    meta = []
    if query_process.returncode == 0:
        meta = json.loads(query_stdout)
    if len(meta) > 0:
        meta = meta[0]
        logger.info("Query successful, processing metadata")
    else:
        logger.warning("Query failed")
        logger.error(query_stderr)
        meta = {
            "pages": [
                Page(idx=i, label=i + 1, forcedOverlay=False, hidden=False).model_dump()
                for i in range(len(slide_fns))
            ]
        }

    with meta_path.open("w") as fp:
        json.dump(meta, fp, indent=2)


class Compiler:
    def __init__(
        self,
        note,
        meta_info: MetaInfo,
        template_file: Optional[Path],
        transition: bool = True,
    ):
        if template_file is None:
            with importlib.resources.path("typstslideviewer", "template.j2.html") as p:
                template = jinja2.Template(p.read_text())
        else:
            template = jinja2.Template(template_file.read_text())

        self._js_libs = {}
        for lib_name, file_name in [
            (
                "fzstd_js",
                "fzstd.js",
            ),
            (
                "tar_js",
                "tarts.min.js",
            ),
        ]:
            with importlib.resources.path("typstslideviewer", file_name) as p:
                self._js_libs[lib_name] = p.read_text()

        self.template = template
        self.window_log = 29
        self.compression_level = 4
        self.note = note
        self.meta_info = meta_info
        self.transition = transition

    def __call__(self, svgs: dict[int, str]):
        compressor = zstd.ZstdCompressor(
            compression_params=zstd.ZstdCompressionParameters.from_level(
                self.compression_level, window_log=self.window_log
            )
        )
        logger.info("Start compressing JSON data")
        compressed_data = compressor.compress(json.dumps(svgs).encode("utf-8"))
        logger.info(f"Compressed data size: {format_size(len(compressed_data))}")

        base64_encoded = base64.b64encode(compressed_data).decode("utf-8")
        logger.info(f"Base64 encoded data size: {format_size(len(base64_encoded))}")

        return self.template.render(
            total_files=len(svgs),
            slide_64=base64_encoded,
            no_animation=not self.transition,
            note=self.note,
            **self._js_libs,
            **self.meta_info.model_dump(mode="json"),
        )


def main(
    typst_src: str,
    output_file=None,
    svg_folder="svgs",
    template_file=None,
    optimize_png: bool = True,
    optimize_jpg: bool = True,
    quality: int = 80,
    image_format: Literal["webp", "avif"] = "webp",
    force: bool = False,
    note: Literal["", "right"] = "",
    transition: bool = True,
):
    """
    Process Typst source files and generate HTML output.

    Args:
        typst_src (str): Path to the Typst source file.
        output_file (str, optional): Path to the output HTML file.
        svg_folder (str, optional): Folder containing SVG files.
        template_file (str, optional): Path to the template file.
        optimize_png (bool, optional): Whether to optimize PNG images.
        optimize_jpg (bool, optional): Whether to optimize JPG images.
        quality (int, optional): Quality of the optimized images.
        image_format (Literal["webp", "avif"], optional): Format of the optimized images.
        force (bool, optional): Force re-initialization of the SVG folder.
        note (Literal["", "right"], optional): Note position.
        transition (bool, optional): Whether to enable slide transitions.
    """
    svg_folder_path = Path(svg_folder)
    typst_src_path = Path(typst_src)
    if template_file is not None:
        template_file = Path(template_file)
        if not template_file.exists():
            raise Exception(f"Template file '{template_file}' not found")
    if output_file is None:
        output_file = typst_src_path.with_suffix(".html")
    else:
        output_file = Path(output_file)

    if (
        (not (svg_folder_path / "meta.json").exists())
        or (
            typst_src_path.stat().st_mtime
            > (svg_folder_path / "meta.json").stat().st_mtime
        )
        or force
    ):
        init_svg_folder(
            typst_src_path,
            svg_folder_path,
            SVGOptimizer(
                optimize_png=optimize_png,
                optimize_jpg=optimize_jpg,
                quality=quality,
                output_format=image_format,
            ),
        )

    svg_files = list(sorted(svg_folder_path.glob("modified_*.svg")))
    if not svg_files:
        raise Exception(f"No SVG files found in the '{svg_folder}' folder")

    meta_info = init_meta_info(svg_folder_path / "meta.json")
    compiler = Compiler(note, meta_info, template_file, transition=transition)

    svgs = {}
    for f in svg_files:
        idx = int(f.stem.split("_")[-1])
        svgs[idx - 1] = f.read_text()

    with open(output_file, "w") as fp:
        fp.write(compiler(svgs))


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == "placeholders":
        sys.argv.pop(1)
        fire.Fire(generate_placeholder_images)
    else:
        fire.Fire(main)
