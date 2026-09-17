"""Call the local OCR adapter with a PDF and print the page results.

Usage:
    python test_pdf_ocr.py
    python test_pdf_ocr.py --pages 0,2-4 --output ocr-result.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PDF OCR through the local OCR adapter API.")
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=Path("test.pdf"),
        help="PDF file to recognize (default: test.pdf)",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("OCR_ADAPTER_URL", "http://127.0.0.1:8000/v1/ocr"),
        help="OCR endpoint URL (default: OCR_ADAPTER_URL or http://127.0.0.1:8000/v1/ocr)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OCR_ADAPTER_API_KEY", ""),
        help="Adapter API key (default: OCR_ADAPTER_API_KEY)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Pages to OCR, using zero-based indexes such as 0,2-4 (default: all pages)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr-result.json"),
        help="Path for the complete JSON response (default: ocr-result.json)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds (default: 300)",
    )
    return parser.parse_args()


def build_payload(pdf_path: Path, pages: str | None) -> dict[str, object]:
    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    payload: dict[str, object] = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded_pdf}",
        },
    }
    if pages is not None:
        payload["pages"] = pages
    return payload


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        print(f"错误：找不到 PDF 文件：{pdf_path}")
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"错误：输入文件不是 PDF：{pdf_path}")
        return 2

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print(f"正在提交：{pdf_path}")
    print(f"OCR 接口：{args.url}")
    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(args.url, headers=headers, json=build_payload(pdf_path, args.pages))
    except httpx.HTTPError as exc:
        print(f"请求失败：{exc}")
        print("请确认服务已启动，例如：python -m uvicorn app.main:app --port 8000")
        return 1

    try:
        result = response.json()
    except ValueError:
        print(f"服务返回了非 JSON 内容（HTTP {response.status_code}）：{response.text[:500]}")
        return 1

    if response.is_error:
        print(f"OCR 失败（HTTP {response.status_code}）：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pages_result = result.get("pages", [])
    print(f"OCR 成功，共识别 {len(pages_result)} 页。")
    for page in pages_result:
        print(f"\n===== 第 {page.get('index', '?')} 页 =====")
        print(page.get("markdown", ""))
    print(f"\n完整结果已保存到：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
