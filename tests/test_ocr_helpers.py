import pytest

from app.core.errors import ServiceError
from app.services.ocr_service import OCRService
from app.services.siliconflow import SiliconFlowClient


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
