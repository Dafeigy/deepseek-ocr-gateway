import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.errors import ServiceError


class DocumentAdmissionGate:
    """Bounds active documents and the number of requests waiting for a slot."""

    def __init__(self, active_limit: int, queued_limit: int, timeout_seconds: float) -> None:
        self._active_limit = active_limit
        self._queued_limit = queued_limit
        self._timeout_seconds = timeout_seconds
        self._active = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    async def _acquire(self) -> None:
        async with self._condition:
            if self._active < self._active_limit:
                self._active += 1
                return
            if self._waiting >= self._queued_limit:
                raise ServiceError(503, "ocr_queue_full", "OCR document queue is full")
            self._waiting += 1
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    await self._condition.wait_for(lambda: self._active < self._active_limit)
                self._active += 1
            except TimeoutError as exc:
                raise ServiceError(503, "ocr_queue_timeout", "Timed out waiting for an OCR slot") from exc
            finally:
                self._waiting -= 1

    async def _release(self) -> None:
        async with self._condition:
            self._active -= 1
            self._condition.notify(1)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self._acquire()
        try:
            yield
        finally:
            await self._release()

    async def snapshot(self) -> dict[str, int]:
        async with self._condition:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "active_limit": self._active_limit,
                "queued_limit": self._queued_limit,
            }
