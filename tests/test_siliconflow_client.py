import httpx
import respx

from app.core.config import Settings
from app.services.siliconflow import SiliconFlowClient


@respx.mock
async def test_upstream_retry_cleaning_and_usage_reconciliation() -> None:
    url = "https://api.siliconflow.cn/v1/chat/completions"
    route = respx.post(url).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"message": "rate limited"}),
            httpx.Response(
                200,
                headers={"x-siliconcloud-trace-id": "trace-upstream"},
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": ("<|ref|>title<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n# 标题\n正文")},
                        }
                    ],
                    "usage": {"total_tokens": 900},
                },
            ),
        ]
    )
    settings = Settings(
        _env_file=None,
        SILICONFLOW_API_KEY="test-key",
        OCR_MAX_RETRIES=1,
        SILICONFLOW_RATE_UTILIZATION=1,
    )
    client = SiliconFlowClient(settings)
    try:
        result = await client.recognize(b"fake-image", "image/png", "request-1")
        snapshot = await client.rate_limiter.snapshot()
    finally:
        await client.close()

    assert route.call_count == 2
    assert result.text == "# 标题\n正文"
    assert result.total_tokens == 900
    assert result.trace_id == "trace-upstream"
    assert snapshot["tpm_used"] == 900


@respx.mock
async def test_context_limit_error_retries_once_with_safe_max_tokens() -> None:
    url = "https://api.siliconflow.cn/v1/chat/completions"
    route = respx.post(url).mock(
        side_effect=[
            httpx.Response(
                400,
                json={"message": "max_tokens (8192) have exceeded max_seq_len (8192) limit."},
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": "正文"}}],
                    "usage": {"total_tokens": 1000},
                },
            ),
        ]
    )
    settings = Settings(
        _env_file=None,
        SILICONFLOW_API_KEY="test-key",
        OCR_MAX_OUTPUT_TOKENS=8192,
        OCR_MAX_RETRIES=1,
        SILICONFLOW_RATE_UTILIZATION=1,
    )
    client = SiliconFlowClient(settings)
    try:
        result = await client.recognize(b"fake-image", "image/png", "request-2")
    finally:
        await client.close()

    first_payload = route.calls[0].request.read().decode("utf-8")
    second_payload = route.calls[1].request.read().decode("utf-8")
    assert route.call_count == 2
    assert '"max_tokens":8192' in first_payload
    assert '"max_tokens":4096' in second_payload
    assert result.text == "正文"
