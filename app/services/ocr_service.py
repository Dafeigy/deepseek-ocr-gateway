import asyncio
import base64
import io
import re
import uuid

from PIL import Image, UnidentifiedImageError

from app.core.admission import DocumentAdmissionGate
from app.core.config import Settings
from app.core.errors import ServiceError
from app.schemas.ocr import (
    OCRBlock,
    OCRImageObject,
    OCRPage,
    OCRPageDimensions,
    OCRRequest,
    OCRResponse,
    OCRUsageInfo,
    document_source,
)
from app.services.document_loader import DocumentLoader, LoadedDocument
from app.services.pdf_renderer import PDFRenderer, RenderedPage
from app.services.siliconflow import OCRTextResult, SiliconFlowClient


class OCRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = DocumentLoader(settings)
        self.renderer = PDFRenderer(settings)
        self.upstream = SiliconFlowClient(settings)
        self.admission = DocumentAdmissionGate(
            settings.max_active_documents,
            settings.max_queued_documents,
            settings.document_queue_timeout_seconds,
        )

    async def close(self) -> None:
        await self.loader.close()
        await self.upstream.close()

    async def process(self, request: OCRRequest, request_id: str | None = None) -> OCRResponse:
        trace_id = self._safe_trace_id(request_id)
        document = await self.loader.load(document_source(request.document))
        if document.is_pdf:
            pages, total_tokens = await self._process_pdf(document, request, trace_id)
        else:
            if request.pages is not None:
                selected = self._parse_pages(request.pages, 1)
                if selected != [0]:
                    raise ServiceError(422, "invalid_pages", "Images only contain page 0")
            page, total_tokens = await self._process_image(document, request, trace_id)
            pages = [page]
        return OCRResponse(
            pages=pages,
            model=self.settings.siliconflow_model,
            usage_info=OCRUsageInfo(
                pages_processed=len(pages),
                doc_size_bytes=document.size,
                total_tokens=total_tokens,
            ),
        )

    async def _process_pdf(
        self,
        document: LoadedDocument,
        request: OCRRequest,
        trace_id: str,
    ) -> tuple[list[OCRPage], int | None]:
        page_count = await self.renderer.page_count(document.data)
        indices = self._parse_pages(request.pages, page_count)
        per_document = asyncio.Semaphore(self.settings.max_page_concurrency_per_document)
        tasks: dict[int, asyncio.Task[tuple[OCRPage, int | None]]] = {}

        async def process_page(index: int) -> tuple[OCRPage, int | None]:
            async with per_document:
                try:
                    rendered = await self.renderer.render(document.data, index)
                    result = await self.upstream.recognize(
                        rendered.data,
                        rendered.mime_type,
                        f"{trace_id}-p{index}",
                    )
                    page = await asyncio.to_thread(
                        self._page_response,
                        index,
                        rendered,
                        result,
                        request,
                    )
                    return page, result.total_tokens
                except ServiceError as exc:
                    details = dict(exc.details or {})
                    details.setdefault("page", index)
                    raise ServiceError(
                        exc.status_code,
                        exc.code,
                        exc.message,
                        details=details,
                    ) from exc

        for index in indices:
            tasks[index] = asyncio.create_task(process_page(index))

        done, pending = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_EXCEPTION)
        failure = next((task.exception() for task in done if task.exception() is not None), None)
        if failure is not None:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise failure
        await asyncio.gather(*pending)

        pages: list[OCRPage] = []
        known_tokens: list[int] = []
        for index in indices:
            page, tokens = tasks[index].result()
            pages.append(page)
            if tokens is not None:
                known_tokens.append(tokens)
        return pages, sum(known_tokens) if known_tokens else None

    async def _process_image(
        self,
        document: LoadedDocument,
        request: OCRRequest,
        trace_id: str,
    ) -> tuple[OCRPage, int | None]:
        width, height = await asyncio.to_thread(self._image_dimensions, document.data)
        result = await self.upstream.recognize(document.data, document.mime_type, trace_id)
        rendered = RenderedPage(
            data=document.data,
            mime_type=document.mime_type,
            width=width,
            height=height,
            dpi=0,
        )
        page = await asyncio.to_thread(self._page_response, 0, rendered, result, request)
        page.dimensions.dpi = None
        return page, result.total_tokens

    @staticmethod
    def _image_dimensions(data: bytes) -> tuple[int, int]:
        try:
            with Image.open(io.BytesIO(data)) as image:
                return image.size
        except (UnidentifiedImageError, OSError) as exc:
            raise ServiceError(422, "invalid_image", "Image could not be decoded") from exc

    @classmethod
    def _page_response(
        cls,
        index: int,
        rendered: RenderedPage,
        result: OCRTextResult,
        request: OCRRequest,
    ) -> OCRPage:
        include_blocks = request.include_blocks is not False
        blocks: list[OCRBlock] = []
        images: list[OCRImageObject] = []
        source_image: Image.Image | None = None
        image_limit = request.image_limit
        min_size = request.image_min_size or 0

        try:
            for grounding in result.grounding_blocks:
                for normalized_box in grounding.boxes:
                    left, top, right, bottom = cls._pixel_box(
                        normalized_box,
                        rendered.width,
                        rendered.height,
                    )
                    image_id: str | None = None
                    eligible_image = (
                        grounding.block_type == "image"
                        and right - left >= min_size
                        and bottom - top >= min_size
                        and (image_limit is None or len(images) < image_limit)
                    )
                    if eligible_image:
                        image_id = f"img-{len(images)}.jpeg"
                        encoded_crop: str | None = None
                        if request.include_image_base64:
                            if source_image is None:
                                source_image = Image.open(io.BytesIO(rendered.data))
                            encoded_crop = cls._encode_crop(source_image, (left, top, right, bottom))
                        images.append(
                            OCRImageObject(
                                id=image_id,
                                top_left_x=left,
                                top_left_y=top,
                                bottom_right_x=right,
                                bottom_right_y=bottom,
                                image_base64=encoded_crop,
                            )
                        )
                    if include_blocks:
                        blocks.append(
                            OCRBlock(
                                type=grounding.block_type,
                                top_left_x=left,
                                top_left_y=top,
                                bottom_right_x=right,
                                bottom_right_y=bottom,
                                content=grounding.content,
                                image_id=image_id,
                            )
                        )
        finally:
            if source_image is not None:
                source_image.close()

        return OCRPage(
            index=index,
            markdown=result.text,
            images=images,
            blocks=blocks if include_blocks else None,
            dimensions=OCRPageDimensions(
                dpi=rendered.dpi or None,
                width=rendered.width,
                height=rendered.height,
            ),
        )

    @staticmethod
    def _pixel_box(
        box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        left = max(0, min(width - 1, round(x1 / 999 * width)))
        top = max(0, min(height - 1, round(y1 / 999 * height)))
        right = max(left + 1, min(width, round(x2 / 999 * width)))
        bottom = max(top + 1, min(height, round(y2 / 999 * height)))
        return left, top, right, bottom

    @staticmethod
    def _encode_crop(image: Image.Image, box: tuple[int, int, int, int]) -> str:
        with image.crop(box).convert("RGB") as crop:
            output = io.BytesIO()
            crop.save(output, format="JPEG", quality=92, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _safe_trace_id(value: str | None) -> str:
        if value and re.fullmatch(r"[A-Za-z0-9._:-]{1,96}", value):
            return value
        return str(uuid.uuid4())

    @staticmethod
    def _parse_pages(pages: list[int] | str | None, page_count: int) -> list[int]:
        if pages is None:
            return list(range(page_count))
        parsed: list[int] = []
        if isinstance(pages, list):
            parsed = pages
        else:
            if not pages.strip():
                raise ServiceError(422, "invalid_pages", "Page selection is empty")
            for part in pages.split(","):
                part = part.strip()
                if re.fullmatch(r"\d+", part):
                    parsed.append(int(part))
                    continue
                range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
                if not range_match:
                    raise ServiceError(422, "invalid_pages", f"Invalid page selection: {part}")
                start, end = map(int, range_match.groups())
                if end < start:
                    raise ServiceError(422, "invalid_pages", f"Invalid page range: {part}")
                parsed.extend(range(start, end + 1))
        if not parsed:
            raise ServiceError(422, "invalid_pages", "At least one page must be selected")
        if len(set(parsed)) != len(parsed):
            raise ServiceError(422, "invalid_pages", "Page selection contains duplicates")
        invalid = [index for index in parsed if index < 0 or index >= page_count]
        if invalid:
            raise ServiceError(
                422,
                "invalid_pages",
                "Page selection is outside the document",
                details={"invalid_pages": invalid, "page_count": page_count},
            )
        return sorted(parsed)
