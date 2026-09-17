#!/usr/bin/env python3
"""Convert a Mistral-compatible OCR JSON response into one Markdown document.

Usage:
    python ocr_to_markdown.py ocr-result.json -o result.md --assets-dir result-assets

The converter uses ``pages[].blocks`` so it also works when the response's
page-level ``markdown`` field is missing or is not useful to the caller.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
from pathlib import Path
from typing import Any


def _number(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_content(content: Any) -> str:
    if not isinstance(content, str):
        return ""
    # OCR services occasionally return CRLF or trailing whitespace-only lines.
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return value or "image"


def _write_image(image: dict[str, Any], destination: Path) -> str | None:
    encoded = image.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        header, payload = encoded.split(",", 1) if "," in encoded else ("", encoded)
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        return None
    extension = ".jpeg" if "jpeg" in header.lower() else ".png"
    destination = destination.with_suffix(extension)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return destination.name


def page_to_markdown(
    page: dict[str, Any],
    *,
    page_number: int,
    assets_dir: Path | None = None,
    markdown_dir: Path | None = None,
) -> str:
    """Restore one page from OCR blocks, preserving the block reading order."""
    blocks = page.get("blocks")
    if not isinstance(blocks, list):
        return _clean_content(page.get("markdown", ""))

    images = {
        item.get("id"): item
        for item in page.get("images", [])
        if isinstance(item, dict) and item.get("id")
    }
    ordered = sorted(
        (item for item in blocks if isinstance(item, dict)),
        key=lambda item: (
            _number(item.get("top_left_y")),
            _number(item.get("top_left_x")),
            _number(item.get("bottom_right_y")),
        ),
    )
    output: list[str] = []
    for block in ordered:
        block_type = str(block.get("type", "text")).lower()
        content = _clean_content(block.get("content", ""))
        if block_type == "image":
            image_id = block.get("image_id")
            image = images.get(image_id)
            if assets_dir is None or not image:
                continue
            stem = _safe_filename(Path(str(image_id or f"page-{page_number}-image")).stem)
            image_name = _write_image(image, assets_dir / f"page-{page_number}-{stem}.bin")
            if image_name:
                relative = Path(image_name)
                if markdown_dir is not None:
                    if assets_dir.is_relative_to(markdown_dir):
                        relative = assets_dir.relative_to(markdown_dir) / image_name
                    else:
                        relative = Path("..") / assets_dir.name / image_name
                output.append(f"![OCR image]({relative.as_posix()})")
            continue
        # Empty table/equation/image placeholders are common in this format.
        if content:
            output.append(content)
    return "\n\n".join(output)


def response_to_markdown(
    response: dict[str, Any],
    *,
    assets_dir: Path | None = None,
    markdown_path: Path | None = None,
    page_separator: str = "\n\n",
) -> str:
    pages = response.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("OCR response must contain a pages list")
    markdown_dir = markdown_path.parent if markdown_path else None
    converted = [
        page_to_markdown(
            page,
            page_number=_number(page.get("index"), index),
            assets_dir=assets_dir,
            markdown_dir=markdown_dir,
        )
        for index, page in enumerate(pages)
        if isinstance(page, dict)
    ]
    return page_separator.join(text for text in converted if text).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OCR response JSON file")
    parser.add_argument("-o", "--output", type=Path, help="Markdown output (default: stdout)")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help="Directory for image_base64 assets; image blocks are omitted when not supplied",
    )
    args = parser.parse_args()
    response = json.loads(args.input.read_text(encoding="utf-8"))
    result = response_to_markdown(response, assets_dir=args.assets_dir, markdown_path=args.output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
