import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class _TokenEvent:
    reservation_id: int
    timestamp: float
    amount: int


class AdaptiveSlidingWindowLimiter:
    """RPM/TPM limiter with estimated admission and actual-usage reconciliation."""

    def __init__(
        self,
        *,
        rpm_limit: int,
        tpm_limit: int,
        utilization: float,
        window_seconds: float,
        initial_tokens: int,
        min_tokens: int,
        max_tokens: int,
        reservation_factor: float,
    ) -> None:
        self._rpm_limit = max(1, math.floor(rpm_limit * utilization))
        self._tpm_limit = max(1, math.floor(tpm_limit * utilization))
        self._window = window_seconds
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._factor = reservation_factor
        self._ewma_tokens = float(initial_tokens)
        self._requests: deque[float] = deque()
        self._tokens: deque[_TokenEvent] = deque()
        self._events_by_id: dict[int, _TokenEvent] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0].timestamp <= cutoff:
            event = self._tokens.popleft()
            self._events_by_id.pop(event.reservation_id, None)

    def _estimate(self) -> int:
        estimate = math.ceil(self._ewma_tokens * self._factor)
        return min(self._max_tokens, max(self._min_tokens, estimate))

    def _token_total(self) -> int:
        return sum(event.amount for event in self._tokens)

    def _wait_time(self, now: float, estimate: int, token_total: int) -> float:
        waits: list[float] = []
        if len(self._requests) >= self._rpm_limit:
            waits.append(self._requests[0] + self._window - now)
        excess = token_total + estimate - self._tpm_limit
        if excess > 0:
            released = 0
            for event in self._tokens:
                released += event.amount
                if released >= excess:
                    waits.append(event.timestamp + self._window - now)
                    break
        return max(0.05, min(waits) if waits else 0.25)

    async def acquire(self) -> tuple[int, int]:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                estimate = self._estimate()
                token_total = self._token_total()
                if len(self._requests) < self._rpm_limit and token_total + estimate <= self._tpm_limit:
                    reservation_id = self._next_id
                    self._next_id += 1
                    event = _TokenEvent(reservation_id, now, estimate)
                    self._requests.append(now)
                    self._tokens.append(event)
                    self._events_by_id[reservation_id] = event
                    return reservation_id, estimate
                delay = self._wait_time(now, estimate, token_total)
            await asyncio.sleep(min(delay, 1.0))

    async def settle(self, reservation_id: int, actual_tokens: int | None) -> None:
        if actual_tokens is None or actual_tokens < 0:
            return
        async with self._lock:
            event = self._events_by_id.get(reservation_id)
            if event is None:
                return
            event.amount = actual_tokens
            if actual_tokens > 0:
                self._ewma_tokens = 0.8 * self._ewma_tokens + 0.2 * actual_tokens

    async def snapshot(self) -> dict[str, int | float]:
        async with self._lock:
            now = time.monotonic()
            self._prune(now)
            return {
                "rpm_used": len(self._requests),
                "rpm_limit": self._rpm_limit,
                "tpm_used": self._token_total(),
                "tpm_limit": self._tpm_limit,
                "next_token_reservation": self._estimate(),
                "observed_tokens_ewma": round(self._ewma_tokens, 2),
            }
