import asyncio
import io
from dataclasses import dataclass

import pypdfium2 as pdfium

from app.core.config import Settings
from app.core.errors import ServiceError


@dataclass(frozen=True, slots=True)
class RenderedPage:
    data: bytes
    mime_type: str
    width: int
    height: int
    dpi: int


class PDFRenderer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._render_slots = asyncio.Semaphore(settings.max_upstream_concurrency)

    async def page_count(self, pdf_data: bytes) -> int:
        try:
            count = await asyncio.to_thread(self._page_count_sync, pdf_data)
        except Exception as exc:
            raise ServiceError(422, "invalid_pdf", "PDF could not be opened") from exc
        if count <= 0:
            raise ServiceError(422, "empty_pdf", "PDF contains no pages")
        if count > self._settings.max_pdf_pages:
            raise ServiceError(
                413,
                "pdf_page_limit_exceeded",
                f"PDF has {count} pages; the configured limit is {self._settings.max_pdf_pages}",
            )
        return count

    @staticmethod
    def _page_count_sync(pdf_data: bytes) -> int:
        document = pdfium.PdfDocument(pdf_data)
        try:
            return len(document)
        finally:
            document.close()

    async def render(self, pdf_data: bytes, page_index: int) -> RenderedPage:
        async with self._render_slots:
            try:
                return await asyncio.to_thread(self._render_sync, pdf_data, page_index)
            except Exception as exc:
                raise ServiceError(
                    422,
                    "pdf_render_failed",
                    f"Failed to render PDF page {page_index}",
                    details={"page": page_index},
                ) from exc

    def _render_sync(self, pdf_data: bytes, page_index: int) -> RenderedPage:
        document = pdfium.PdfDocument(pdf_data)
        page = document.get_page(page_index)
        bitmap = None
        image = None
        try:
            bitmap = page.render(scale=self._settings.render_dpi / 72.0)
            image = bitmap.to_pil().convert("RGB")
            width, height = image.size
            output = io.BytesIO()
            if self._settings.render_format == "JPEG":
                image.save(output, format="JPEG", quality=94, optimize=True)
                mime_type = "image/jpeg"
            else:
                image.save(output, format="PNG", optimize=True)
                mime_type = "image/png"
            return RenderedPage(
                data=output.getvalue(),
                mime_type=mime_type,
                width=width,
                height=height,
                dpi=self._settings.render_dpi,
            )
        finally:
            if image is not None:
                image.close()
            if bitmap is not None:
                bitmap.close()
            page.close()
            document.close()
