from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
    error: dict[str, Any] = {
        "message": exc.message,
        "type": exc.code,
        "param": None,
        "code": exc.code,
    }
    if exc.details:
        error["details"] = exc.details
    return JSONResponse(
        status_code=exc.status_code,
        content={"object": "error", "error": error},
    )
