import base64
from unittest.mock import AsyncMock

from app.core.config import Settings
from app.schemas.ocr import OCRRequest
from app.services.ocr_service import OCRService
from app.services.siliconflow import OCRTextResult
from tests.pdf_fixture import make_two_page_pdf


async def test_pdf_is_rendered_and_returned_in_page_order() -> None:
    settings = Settings(
        _env_file=None,
        SILICONFLOW_API_KEY="test-key",
        OCR_RENDER_DPI=72,
        OCR_MAX_UPSTREAM_CONCURRENCY=2,
        OCR_MAX_PAGE_CONCURRENCY_PER_DOCUMENT=2,
    )
    service = OCRService(settings)

    async def recognize(_data: bytes, _mime: str, request_id: str) -> OCRTextResult:
        page_index = int(request_id.rsplit("-p", 1)[1])
        return OCRTextResult(
            text=f"page-{page_index}",
            total_tokens=100 + 20 * page_index,
            trace_id=str(page_index),
        )

    service.upstream.recognize = AsyncMock(side_effect=recognize)
    pdf_data = make_two_page_pdf()
    encoded = base64.b64encode(pdf_data).decode()
    request = OCRRequest.model_validate(
        {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{encoded}",
            },
        }
    )
    try:
        response = await service.process(request)
    finally:
        await service.close()

    assert [page.index for page in response.pages] == [0, 1]
    assert [page.markdown for page in response.pages] == ["page-0", "page-1"]
    assert response.pages[0].dimensions.width == 612
    assert response.pages[0].dimensions.height == 792
    assert response.usage_info.pages_processed == 2
    assert response.usage_info.total_tokens == 220
