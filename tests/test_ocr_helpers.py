import io

import pytest
from PIL import Image

from app.core.errors import ServiceError
from app.schemas.ocr import OCRRequest
from app.services.ocr_service import OCRService
from app.services.pdf_renderer import RenderedPage
from app.services.siliconflow import OCRTextResult, SiliconFlowClient


def test_parse_page_ranges() -> None:
    assert OCRService._parse_pages("0,2-4", 6) == [0, 2, 3, 4]
    assert OCRService._parse_pages([2, 0], 3) == [0, 2]


def test_reject_invalid_pages() -> None:
    with pytest.raises(ServiceError) as error:
        OCRService._parse_pages("0,4", 3)
    assert error.value.code == "invalid_pages"


def test_clean_grounding_metadata() -> None:
    raw = "<|ref|>title<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n# 标题\n\n正文"
    assert SiliconFlowClient._clean_output(raw) == "# 标题\n\n正文"


def test_parse_grounding_blocks() -> None:
    raw = (
        "<|ref|>title<|/ref|><|det|>[[10,20,500,100]]<|/det|>\n# 标题\n"
        "<|ref|>image<|/ref|><|det|>[[100,200,600,800]]<|/det|>"
    )
    markdown, blocks = SiliconFlowClient._parse_output(raw)

    assert markdown == "# 标题"
    assert blocks[0].block_type == "title"
    assert blocks[0].content == "# 标题"
    assert blocks[0].boxes == ((10, 20, 500, 100),)
    assert blocks[1].block_type == "image"


def test_mistral_blocks_and_cropped_images_are_returned() -> None:
    source = io.BytesIO()
    Image.new("RGB", (100, 200), "white").save(source, format="PNG")
    raw = "<|ref|>image<|/ref|><|det|>[[100,200,600,800]]<|/det|>"
    markdown, grounding_blocks = SiliconFlowClient._parse_output(raw)
    result = OCRTextResult(
        text=markdown,
        total_tokens=10,
        trace_id=None,
        grounding_blocks=grounding_blocks,
    )
    request = OCRRequest.model_validate(
        {
            "model": "mistral-ocr-latest",
            "document": {"type": "image_url", "image_url": "data:image/png;base64,"},
            "include_image_base64": True,
        }
    )
    page = OCRService._page_response(
        0,
        RenderedPage(source.getvalue(), "image/png", 100, 200, 0),
        result,
        request,
    )

    assert len(page.blocks) == 1
    assert page.blocks[0].type == "image"
    assert page.blocks[0].image_id == "img-0.jpeg"
    assert (page.blocks[0].top_left_x, page.blocks[0].top_left_y) == (10, 40)
    assert (page.blocks[0].bottom_right_x, page.blocks[0].bottom_right_y) == (60, 160)
    assert len(page.images) == 1
    assert page.images[0].image_base64.startswith("data:image/jpeg;base64,")
