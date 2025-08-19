import base64
import concurrent.futures
import importlib.resources
import io
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

import fire
import jinja2
import zstandard as zstd
from loguru import logger
from PIL import Image
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


def format_size(size: Union[int, float]) -> str:
    # Convert bytes to a human-readable format
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


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
    for elem in element.iter():
        # Remove the namespace prefix
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


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
        parent_map = {c: p for p in tree.iter() for c in p}
        root = tree.getroot()
        namespaces = {"xlink": "http://www.w3.org/1999/xlink"}

        for image in root.findall(".//{http://www.w3.org/2000/svg}image", namespaces):
            href = image.get("{http://www.w3.org/1999/xlink}href")
            if href and href.startswith("data:image/svg+xml;base64,"):
                decoded_data = decode_base64_data(href)
                decoded_svg = ET.fromstring(decoded_data.decode("utf-8"))

                foreign_object = decoded_svg.find(
                    ".//{http://www.w3.org/2000/svg}foreignObject"
                )
                if foreign_object is not None:
                    children = list(foreign_object)
                    if (
                        len(children) == 1
                        and children[0].tag == "{http://www.w3.org/1999/xhtml}video"
                    ):
                        source_element = children[0].find(
                            "{http://www.w3.org/1999/xhtml}source"
                        )
                        if source_element is not None:
                            src = source_element.attrib.get("src")
                            if src and src.endswith(".mp4"):
                                video_path = Path(src)
                                if video_path.exists():
                                    with open(video_path, "rb") as video_file:
                                        base64_video = base64.b64encode(
                                            video_file.read()
                                        ).decode("utf-8")
                                        source_element.attrib["src"] = (
                                            f"data:video/mp4;base64,{base64_video}"
                                        )

                    parent = parent_map[image]
                    parent.remove(image)
                    parent.append(foreign_object)

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
                pass
            elif self.optimize_png and href.startswith("data:image/png;base64,"):
                image.attrib["{http://www.w3.org/1999/xlink}href"] = convert_func(
                    href, self.quality
                )
            elif self.optimize_jpg and href.startswith("data:image/jpeg;base64,"):
                image.attrib["{http://www.w3.org/1999/xlink}href"] = convert_func(
                    href, self.quality
                )
            elif href and href.startswith("data:image/svg+xml;base64,"):
                t = ET.ElementTree(
                    ET.fromstring(decode_base64_data(href).decode("utf-8"))
                )
                self._optimize_bitmap(t)
                data = ET.tostring(
                    t.getroot(), encoding="unicode", short_empty_elements=False
                )
                image.attrib["{http://www.w3.org/1999/xlink}href"] = (
                    f"data:image/svg+xml;base64,{base64.b64encode(data.encode('utf-8')).decode('utf-8')}"
                )

    def optimize(self, svgstring):
        # tree = ET.parse(svg_file)
        tree = ET.ElementTree(ET.fromstring(svgstring))
        root = tree.getroot()
        self._optimize_bitmap(tree)
        self.inline_foreignObject(tree)

        # Write the modified SVG back to a file
        strip_namespace(root)
        img = ET.tostring(root, encoding="unicode", short_empty_elements=False)
        return re.sub(r'width="([0-9\.]+)pt" height="([0-9\.]+)pt"', "", img)


def process_file(f, svg_folder_path, optimizer):
    svgstring = f.read_text()
    with (svg_folder_path / f"modified_{f.name}").open("w") as fp:
        print(optimizer.optimize(svgstring), file=fp)


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
        ["typst", "compile", typst_src_path, f"{svg_folder_path}/slide_{{0p}}.svg"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info(f"Querying Typst source file: {typst_src_path}")
    query_process = subprocess.Popen(
        ["typst", "query", typst_src_path, "--field", "value", "<pdfpc-file>"],
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
    # for f in slide_fns:
    #     svgstring = f.read_text()
    #     with (svg_folder_path / f"modified_{f.name}").open("w") as fp:
    #         print(optimizer.optimize(svgstring), file=fp)

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
            ),  # https://cdn.jsdelivr.net/npm/fzstd@0.1.1/umd/index.js
            (
                "tar_js",
                "tarts.min.js",
            ),  # https://cdn.jsdelivr.net/npm/tarts@1.0.0/dist/tarts.min.js
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

        # Step 4: Encode the compressed data to Base64
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


def mian(
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
        note (Literal["", "right"], optional): Note position. If not specified, assume that the slides are not compiled in the speaker mode and the note will be displayed as text in the control window. Only use "right" if the slides are compiled in the speaker mode with `config-common(show-notes-on-second-screen: right)`.

    Raises:
        Exception: If the template file is not found.
        Exception: If no SVG files are found in the specified folder.
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
        logger.info("initializing")

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
        print(compiler(svgs), file=fp)


def cli():
    fire.Fire(mian)
