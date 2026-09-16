from app.core.rate_limit import AdaptiveSlidingWindowLimiter


async def test_rate_limiter_reconciles_actual_usage() -> None:
    limiter = AdaptiveSlidingWindowLimiter(
        rpm_limit=1000,
        tpm_limit=80000,
        utilization=0.9,
        window_seconds=60,
        initial_tokens=3000,
        min_tokens=512,
        max_tokens=8192,
        reservation_factor=1.25,
    )
    reservation_id, reserved = await limiter.acquire()
    assert reserved == 3750
    await limiter.settle(reservation_id, 1000)
    snapshot = await limiter.snapshot()
    assert snapshot["tpm_used"] == 1000
    assert snapshot["rpm_limit"] == 900
    assert snapshot["tpm_limit"] == 72000
