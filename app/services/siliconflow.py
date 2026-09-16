import asyncio
import base64
import random
import re
import uuid
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.rate_limit import AdaptiveSlidingWindowLimiter

_GROUNDING_METADATA = re.compile(r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", re.DOTALL)
_EOS_MARKERS = ("<｜end▁of▁sentence｜>", "<|end_of_sentence|>")
_SAFE_CONTEXT_FALLBACK_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class OCRTextResult:
    text: str
    total_tokens: int | None
    trace_id: str | None


class SiliconFlowClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._concurrency = asyncio.Semaphore(settings.max_upstream_concurrency)
        self.rate_limiter = AdaptiveSlidingWindowLimiter(
            rpm_limit=settings.siliconflow_rpm_limit,
            tpm_limit=settings.siliconflow_tpm_limit,
            utilization=settings.siliconflow_rate_utilization,
            window_seconds=settings.rate_window_seconds,
            initial_tokens=settings.initial_tokens_per_page,
            min_tokens=settings.min_tokens_per_page,
            max_tokens=settings.max_tokens_per_page,
            reservation_factor=settings.token_reservation_factor,
        )
        timeout = httpx.Timeout(
            connect=10,
            read=settings.upstream_timeout_seconds,
            write=settings.upstream_timeout_seconds,
            pool=settings.upstream_timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=settings.max_upstream_concurrency,
                max_keepalive_connections=settings.max_upstream_concurrency,
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _clean_output(text: str) -> str:
        cleaned = _GROUNDING_METADATA.sub("", text)
        for marker in _EOS_MARKERS:
            cleaned = cleaned.replace(marker, "")
        cleaned = cleaned.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
        cleaned = re.sub(r"\n{4,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _retry_after(response: httpx.Response | None) -> float | None:
        if response is None:
            return None
        value = response.headers.get("retry-after")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(
                    0.0,
                    (parsedate_to_datetime(value) - parsedate_to_datetime(response.headers["date"])).total_seconds(),
                )
            except (KeyError, TypeError, ValueError):
                return None

    @staticmethod
    def _is_context_limit_error(message: str) -> bool:
        normalized = message.lower()
        return "max_tokens" in normalized and "max_seq_len" in normalized

    async def recognize(self, image_data: bytes, mime_type: str, request_id: str) -> OCRTextResult:
        if not self._settings.siliconflow_api_key:
            raise ServiceError(503, "siliconflow_key_missing", "SILICONFLOW_API_KEY is not configured")
        data_uri = f"data:{mime_type};base64,{base64.b64encode(image_data).decode('ascii')}"
        payload = {
            "model": self._settings.siliconflow_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": self._settings.prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self._settings.max_output_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.siliconflow_api_key}",
            "Content-Type": "application/json",
            "X-Trace-Id": request_id or str(uuid.uuid4()),
        }
        last_error: Exception | None = None
        async with self._concurrency:
            for attempt in range(self._settings.max_retries + 1):
                reservation_id, _reserved = await self.rate_limiter.acquire()
                response: httpx.Response | None = None
                uncertain_usage = False
                try:
                    response = await self._client.post(
                        self._settings.siliconflow_api_url,
                        headers=headers,
                        json=payload,
                    )
                    if response.status_code in {429, 503, 504}:
                        await self.rate_limiter.settle(reservation_id, 0)
                        if attempt < self._settings.max_retries:
                            delay = self._retry_after(response)
                            if delay is None:
                                delay = (2**attempt) + random.uniform(0, 0.5)
                            await asyncio.sleep(min(delay, 30))
                            continue
                    if response.status_code >= 400:
                        await self.rate_limiter.settle(reservation_id, 0)
                        message = self._error_message(response)
                        if (
                            response.status_code == 400
                            and attempt < self._settings.max_retries
                            and payload["max_tokens"] > _SAFE_CONTEXT_FALLBACK_TOKENS
                            and self._is_context_limit_error(message)
                        ):
                            payload["max_tokens"] = _SAFE_CONTEXT_FALLBACK_TOKENS
                            continue
                        status = response.status_code if response.status_code < 500 else 502
                        raise ServiceError(
                            status,
                            "siliconflow_error",
                            message,
                            details={
                                "upstream_status": response.status_code,
                                "trace_id": response.headers.get("x-siliconcloud-trace-id"),
                            },
                        )
                    body = response.json()
                    if not isinstance(body, dict):
                        raise TypeError("upstream response is not an object")
                    usage = body.get("usage") or {}
                    total_tokens = self._as_int(usage.get("total_tokens"))
                    await self.rate_limiter.settle(reservation_id, total_tokens)
                    choices = body.get("choices") or []
                    if not choices:
                        raise ServiceError(502, "invalid_upstream_response", "SiliconFlow returned no OCR choices")
                    choice = choices[0]
                    if choice.get("finish_reason") == "length":
                        raise ServiceError(
                            502,
                            "ocr_output_truncated",
                            "OCR output reached the model context limit",
                        )
                    content = (choice.get("message") or {}).get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise ServiceError(502, "empty_ocr_output", "SiliconFlow returned empty OCR content")
                    return OCRTextResult(
                        text=self._clean_output(content),
                        total_tokens=total_tokens,
                        trace_id=response.headers.get("x-siliconcloud-trace-id"),
                    )
                except ServiceError:
                    raise
                except httpx.HTTPError as exc:
                    uncertain_usage = True
                    last_error = exc
                    if attempt < self._settings.max_retries:
                        await asyncio.sleep((2**attempt) + random.uniform(0, 0.5))
                        continue
                except (ValueError, TypeError) as exc:
                    await self.rate_limiter.settle(reservation_id, 0)
                    raise ServiceError(502, "invalid_upstream_response", "SiliconFlow returned invalid JSON") from exc
                finally:
                    if response is None and not uncertain_usage:
                        await self.rate_limiter.settle(reservation_id, 0)
        raise ServiceError(
            502, "siliconflow_unavailable", "SiliconFlow OCR request failed after retries"
        ) from last_error

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return value if isinstance(value, int) and value >= 0 else None

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict):
                if isinstance(body.get("message"), str):
                    return body["message"]
                error = body.get("error")
                if isinstance(error, dict) and isinstance(error.get("message"), str):
                    return error["message"]
        except ValueError:
            pass
        return f"SiliconFlow returned HTTP {response.status_code}"
