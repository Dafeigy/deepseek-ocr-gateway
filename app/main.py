from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import ServiceError, service_error_handler
from app.services.ocr_service import OCRService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service = OCRService(get_settings())
    app.state.ocr_service = service
    try:
        yield
    finally:
        await service.close()


app = FastAPI(
    title="SiliconFlow Mistral OCR Adapter",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_exception_handler(ServiceError, service_error_handler)


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    settings = get_settings()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            too_large = int(content_length) > settings.max_request_body_bytes
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "object": "error",
                    "error": {
                        "type": "invalid_content_length",
                        "message": "Invalid Content-Length",
                    },
                },
            )
        if too_large:
            return JSONResponse(
                status_code=413,
                content={
                    "object": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": "Request body exceeds the configured limit",
                    },
                },
            )
    if request.method == "POST" and request.url.path in {"/ocr", "/v1/ocr"}:
        try:
            async with request.app.state.ocr_service.admission.slot():
                return await call_next(request)
        except ServiceError as exc:
            return await service_error_handler(request, exc)
    return await call_next(request)


app.include_router(router)
