"""
#!/usr/bin/env python3
Crop regions from a page image using bounding boxes returned by NVIDIA nemotron-parse.

Inputs:
  1) A JSON file containing the chat completion response with
     choices[*].message.tool_calls[*].function.arguments (JSON-encoded string).
  2) The path to the page image that was parsed (PNG/JPEG).

Output:
  Cropped PNGs saved under --out-dir (default: ./crops).

Usage example:
  Imagecropper % python3 main.py \  
      --input-json ./nemotron_output_page5.json \
      --image ../out_png/out_png-05.png \
      --out-dir ../crops \
      --padding 8 \
      --exclude-classes Text Formula

../crops/crop_001_Page-header_1425-142-1456-178.png
../crops/crop_002_Picture_240-200-1456-1039.png
../crops/crop_003_Caption_240-1056-1456-1199.png
Done. Wrote 3 crops to: ../crops

Notes:
- By default, we EXCLUDE "Text" and "Formula" and keep everything else.
- If you want only specific classes (e.g., Picture, Table), use --include-classes.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Set

from PIL import Image

from logging_config import get_logger


logger = get_logger(__name__)

def parse_args():
    ap = argparse.ArgumentParser(description="Crop image regions from nemotron-parse output")
    ap.add_argument("--input-json", required=True, help="Path to nemotron-parse chat completion JSON")
    ap.add_argument("--image", required=True, help="Path to the corresponding page image (PNG/JPEG)")
    ap.add_argument("--out-dir", default="crops", help="Directory to write cropped images")
    ap.add_argument("--padding", type=int, default=8, help="Padding (pixels) around each crop")
    ap.add_argument(
        "--include-classes",
        nargs="*",
        default=["Picture", "Table", "Figure"],
        help="Only keep these classes (e.g., Picture Table Figure). If omitted, all classes are included except excluded ones."
    )
    ap.add_argument(
        "--exclude-classes",
        nargs="*",
        default=["Text", "Formula", "Caption"],
        help="Exclude these classes (defaults to Text, Formula, and Caption)"
    )
    ap.add_argument("--prefix", default="crop", help="Filename prefix for crops")
    return ap.parse_args()

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def is_normalized_box(b: Dict[str, float]) -> bool:
    # Treat as normalized if coordinates are within a small epsilon of [0,1]
    vals = [b.get("xmin", 0), b.get("ymin", 0), b.get("xmax", 0), b.get("ymax", 0)]
    return all(0.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in vals)

def to_xyxy_pixels(b: Dict[str, float], W: int, H: int) -> Tuple[int, int, int, int]:
    xmin = b.get("xmin", 0.0)
    ymin = b.get("ymin", 0.0)
    xmax = b.get("xmax", 0.0)
    ymax = b.get("ymax", 0.0)

    if is_normalized_box(b):
        x1 = int(round(xmin * W))
        y1 = int(round(ymin * H))
        x2 = int(round(xmax * W))
        y2 = int(round(ymax * H))
    else:
        x1 = int(round(xmin))
        y1 = int(round(ymin))
        x2 = int(round(xmax))
        y2 = int(round(ymax))

    # Ensure ordering is valid
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def expand_and_clip(x1: int, y1: int, x2: int, y2: int, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1p = clamp(x1 - pad, 0, W)
    y1p = clamp(y1 - pad, 0, H)
    x2p = clamp(x2 + pad, 0, W)
    y2p = clamp(y2 + pad, 0, H)
    return x1p, y1p, x2p, y2p

def load_all_markdown_bbox_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and flatten all elements from choices[*].message.tool_calls[*]
    where function.name == 'markdown_bbox'. The 'arguments' field is a
    JSON-encoded string, which typically decodes to a list of lists of items.

    Returns a flat list of element dicts like:
      { "bbox": {...}, "type": "...", "text": "...", ... }
    """
    elements: List[Dict[str, Any]] = []

    choices = payload.get("choices", [])
    for ch in choices:
        message = ch.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        for tc in tool_calls:
            func = (tc or {}).get("function") or {}
            if func.get("name") != "markdown_bbox":
                continue

            args_str = func.get("arguments", "")
            if not isinstance(args_str, str):
                # sometimes it could already be a structure
                data = args_str
            else:
                # Parse the JSON-encoded string
                try:
                    data = json.loads(args_str)
                except json.JSONDecodeError as e:
                    # Some providers wrap the data in another string layer
                    # Try one more time if it looks doubly-encoded
                    try:
                        data = json.loads(json.loads(args_str))
                    except Exception:
                        raise ValueError(f"Failed to parse function.arguments as JSON: {e}") from e

            # Expected shapes: [ [ {item}, {item}, ... ] ] OR [ {item}, ... ]
            if isinstance(data, list):
                # If it's a list of lists, flatten
                for entry in data:
                    if isinstance(entry, list):
                        elements.extend(entry)
                    elif isinstance(entry, dict):
                        elements.append(entry)
                    else:
                        # ignore unrecognized nodes
                        pass
            elif isinstance(data, dict):
                elements.append(data)
            # else: ignore other node types silently

    return elements

def filter_elements(
    elements: Iterable[Dict[str, Any]],
    include: Set[str] = None,
    exclude: Set[str] = None
) -> List[Dict[str, Any]]:
    inc = {c.strip() for c in include} if include else None
    exc = {c.strip() for c in exclude} if exclude else set()

    kept = []
    for el in elements:
        cls = (el.get("type") or "").strip()
        if inc is not None:
            if cls not in inc:
                continue
        else:
            # default path: keep all except excluded
            if cls in exc:
                continue
        kept.append(el)
    return kept

def main():
    args = parse_args()

    logger.info(
        f"image_cropper CLI invoked | input_json={args.input_json} | image={args.image} | out_dir={args.out_dir}",
    )

    # Load JSON payload from file for CLI usage
    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cropped_paths = crop_from_model_response(
        payload=payload,
        image_path=args.image,
        out_dir=args.out_dir,
        padding=args.padding,
        include_classes=args.include_classes,
        exclude_classes=args.exclude_classes,
        prefix=args.prefix,
    )

    logger.info(
        f"image_cropper CLI finished | image={args.image} | crops={len(cropped_paths)} | out_dir={args.out_dir}",
    )

    for p in cropped_paths:
        print(p)

    print(f"Done. Wrote {len(cropped_paths)} crops to: {args.out_dir}")


def crop_from_model_response(
    payload: Dict[str, Any],
    image_path: str,
    out_dir: str,
    padding: int = 8,
    include_classes: Iterable[str] | None = None,
    exclude_classes: Iterable[str] | None = None,
    prefix: str = "crop",
) -> List[str]:
    """Crop regions from a page image using a nemotron-parse-style payload.

    This function is suitable for importing and calling from other Python
    code (e.g., main.py) and mirrors the behavior of the CLI.

    Args:
        payload: Parsed JSON response from the model (dict).
        image_path: Path to the corresponding page image (PNG/JPEG).
        out_dir: Directory where cropped PNGs will be written.
        padding: Padding in pixels around each crop.
        include_classes: Iterable of class names to include (e.g., ["Picture", "Table"]).
            If provided and non-empty, only these classes are kept.
        exclude_classes: Iterable of class names to exclude.
            Used only when include_classes is None or empty.
        prefix: Filename prefix for crops.

    Returns:
        List of full file paths to the written crop images.
    """

    logger.info(
        f"Cropping from model response | image={image_path} | out_dir={out_dir} | padding={padding}",
    )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Default include/exclude lists if caller does not specify any
    if include_classes is None:
        include_classes = ["Picture", "Table", "Figure"]
    if exclude_classes is None:
        exclude_classes = ["Text", "Formula", "Caption"]

    # Derive structured output directory based on the PDF/page naming
    # Convention for page images (from pdf_processor):
    #   pdf_images_dir/{pdf_name}/{pdf_name}_page{N}.png
    # We map this to cropped components under:
    #   out_dir/{pdf_name}/{pdf_name}_page{N}/
    image_path_obj = Path(image_path)
    pdf_name = image_path_obj.parent.name  # e.g., "mydoc"
    page_label = image_path_obj.stem       # e.g., "mydoc_page1"

    page_out_dir = Path(out_dir) / pdf_name / page_label
    page_out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Extract all markdown_bbox elements
    elements = load_all_markdown_bbox_items(payload)

    # Filter classes according to include/exclude configuration
    elements = filter_elements(
        elements,
        include=set(include_classes) if include_classes else None,
        exclude=set(exclude_classes) if exclude_classes else None,
    )

    cropped_paths: List[str] = []
    count = 0

    for el in elements:
        bbox = el.get("bbox") or {}
        cls = (el.get("type") or "Unknown").replace(" ", "_")
        text_snippet = (el.get("text") or "").strip()

        x1, y1, x2, y2 = to_xyxy_pixels(bbox, W, H)
        x1, y1, x2, y2 = expand_and_clip(x1, y1, x2, y2, padding, W, H)

        # skip degenerate
        if x2 <= x1 or y2 <= y1:
            logger.debug(
                f"Skipping degenerate bbox | image={image_path} | bbox={bbox}"
            )
            continue

        crop = img.crop((x1, y1, x2, y2))
        count += 1

        # Shorten text snippet for filename if present (optional, safe chars only)
        safe_txt = ""
        if text_snippet:
            safe_txt = text_snippet.replace("\n", " ").replace("/", "_").replace("\\", "_")
            if len(safe_txt) > 40:
                safe_txt = safe_txt[:40].rstrip() + "…"

        if safe_txt:
            out_name = f"{prefix}{count}.png"
        else:
            out_name = f"{prefix}{count}.png"

        out_path = page_out_dir / out_name
        crop.save(out_path, "PNG")
        cropped_paths.append(str(out_path))

    logger.info(
        f"Completed cropping | image={image_path} | crops={len(cropped_paths)} | out_dir={page_out_dir}"
    )

    return cropped_paths

if __name__ == "__main__":
    main()