import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.errors import ServiceError
from app.schemas.ocr import OCRRequest, OCRResponse
from app.services.ocr_service import OCRService

router = APIRouter()


async def authorize(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.adapter_api_key:
        return
    expected = f"Bearer {settings.adapter_api_key}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise ServiceError(401, "invalid_api_key", "Invalid adapter API key")


def get_service(request: Request) -> OCRService:
    return request.app.state.ocr_service


@router.post("/v1/ocr", response_model=OCRResponse, dependencies=[Depends(authorize)])
@router.post("/ocr", response_model=OCRResponse, include_in_schema=False, dependencies=[Depends(authorize)])
async def ocr(
    payload: OCRRequest,
    request: Request,
    service: Annotated[OCRService, Depends(get_service)],
) -> OCRResponse:
    return await service.process(payload, request.headers.get("x-request-id"))


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(
    service: Annotated[OCRService, Depends(get_service)],
) -> dict[str, object]:
    configured = bool(service.settings.siliconflow_api_key)
    return {
        "status": "ready" if configured else "not_ready",
        "siliconflow_api_key_configured": configured,
        "admission": await service.admission.snapshot(),
        "rate_limit": await service.upstream.rate_limiter.snapshot(),
    }
