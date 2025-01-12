import base64
import importlib.resources
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import fire
import jinja2
import zstandard as zstd
from loguru import logger
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


def format_size(size):
    # Convert bytes to a human-readable format
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024


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
        # if page_label not in jump_map:
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


def find_and_replace_images(svg_file, output_file):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Create a parent map
    parent_map = {c: p for p in tree.iter() for c in p}

    # Define the namespace
    namespaces = {"xlink": "http://www.w3.org/1999/xlink"}

    # Find all <image> elements with xlink:href attribute
    for image in root.findall(".//{http://www.w3.org/2000/svg}image", namespaces):
        href = image.get("{http://www.w3.org/1999/xlink}href")
        if href and href.startswith("data:image/svg+xml;base64,"):
            decoded_data = decode_base64_data(href)
            decoded_svg = ET.fromstring(decoded_data.decode("utf-8"))

            # Find the <foreignObject> element in the decoded SVG
            foreign_object = decoded_svg.find(
                ".//{http://www.w3.org/2000/svg}foreignObject"
            )
            if foreign_object is not None:
                # Replace the <image> node with the <foreignObject> node

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

    # Write the modified SVG back to a file
    strip_namespace(root)
    with open(output_file, "w") as fp:
        img = ET.tostring(root, encoding="unicode", short_empty_elements=False)
        print(
            re.sub(
                r'width="([0-9\.]+)pt" height="([0-9\.]+)pt"',
                # 'width="100%" height="100%"',
                "",
                img,
            ),
            file=fp,
        )


def init_svg_folder(typst_src, svg_folder):
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

    query_stdout, query_stderr = query_process.communicate()

    if query_process.returncode == 0:
        logger.info("Query successful, processing metadata")
        meta = json.loads(query_stdout)[0]
        with open(svg_folder_path / "meta.json", "w") as fp:
            json.dump(meta, fp, indent=2)
    else:
        logger.error(query_stderr)
        raise Exception("Failed to query the Typst file")

    compile_process.communicate()
    if compile_process.returncode == 0:
        logger.info("Compilation successful, processing SVG files")
        for f in svg_folder_path.rglob("slide_*.svg"):
            find_and_replace_images(f, svg_folder_path / f"modified_{f.name}")


def mian(
    typst_src,
    output_file=None,
    svg_folder="svgs",
    template_file="",
    note="",
):
    svg_folder_path = Path(svg_folder)
    typst_src_path = Path(typst_src)
    if template_file != "":
        template_file = Path(template_file)
    if output_file is None:
        output_file = typst_src_path.with_suffix(".html")
    else:
        output_file = Path(output_file)

    if (not (svg_folder_path / "meta.json").exists()) or (
        typst_src_path.stat().st_mtime > (svg_folder_path / "meta.json").stat().st_mtime
    ):
        logger.info("SVG folder is outdated or missing, initializing")
        init_svg_folder(typst_src_path, svg_folder_path)

    # Get list of SVG files
    svg_files = [f.name for f in sorted(svg_folder_path.glob("modified_*.svg"))]
    if not svg_files:
        raise Exception(f"No SVG files found in the '{svg_folder}' folder")

    meta_info = init_meta_info(svg_folder_path / "meta.json")

    if isinstance(template_file, Path):
        template = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path())
        ).get_template(template_file.name)
    elif template_file == "":
        with importlib.resources.path("typstslideviewer", "template.j2.html") as p:
            template = jinja2.Template(p.read_text())

    svgs = {}
    for f in map(Path, svg_files):
        idx = int(f.stem.split("_")[-1])
        with open(svg_folder_path / f) as svg_fp:
            svgs[idx - 1] = svg_fp.read()
    json_string = json.dumps(svgs)

    # Step 3: Compress the JSON string using Zstd
    compressor = zstd.ZstdCompressor(
        compression_params=zstd.ZstdCompressionParameters.from_level(4, window_log=29)
    )
    logger.info("Start compressing JSON data")
    compressed_data = compressor.compress(json_string.encode("utf-8"))
    logger.info(f"Compressed data size: {format_size(len(compressed_data))}")

    # Step 4: Encode the compressed data to Base64
    base64_encoded = base64.b64encode(compressed_data).decode("utf-8")
    logger.info(f"Base64 encoded data size: {format_size(len(base64_encoded))}")

    with open(output_file, "w") as fp:
        print(
            template.render(
                total_files=len(svg_files),
                slide_64=base64_encoded,
                note=note,
                # slides=svgs,
                **meta_info.model_dump(mode="json"),
            ),
            file=fp,
        )


def cli():
    fire.Fire(mian)
