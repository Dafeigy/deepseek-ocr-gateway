import base64

from ocr_to_markdown import response_to_markdown


def test_reconstructs_sorted_blocks_and_falls_back_to_page_markdown() -> None:
    response = {
        "pages": [
            {
                "index": 0,
                "markdown": "unused",
                "blocks": [
                    {"type": "text", "top_left_x": 20, "top_left_y": 20, "content": "第二段"},
                    {"type": "title", "top_left_x": 10, "top_left_y": 10, "content": "# 标题"},
                    {"type": "table", "top_left_x": 0, "top_left_y": 30, "content": ""},
                    {"type": "caption", "top_left_x": 0, "top_left_y": 31, "content": "<table>表格</table>"},
                ],
            },
            {"index": 1, "markdown": "无 blocks 的页面"},
        ]
    }
    assert response_to_markdown(response) == "# 标题\n\n第二段\n\n<table>表格</table>\n\n无 blocks 的页面\n"


def test_extracts_base64_images_and_links_from_markdown_location(tmp_path) -> None:
    image = base64.b64encode(b"jpeg-data").decode()
    response = {
        "pages": [
            {
                "index": 4,
                "blocks": [{"type": "image", "top_left_y": 1, "image_id": "img-0.jpeg", "content": ""}],
                "images": [{"id": "img-0.jpeg", "image_base64": f"data:image/jpeg;base64,{image}"}],
            }
        ]
    }
    output = tmp_path / "doc.md"
    assets = tmp_path / "assets"
    result = response_to_markdown(response, assets_dir=assets, markdown_path=output)
    assert result == "![OCR image](assets/page-4-img-0.jpeg)\n"
    assert (assets / "page-4-img-0.jpeg").read_bytes() == b"jpeg-data"
