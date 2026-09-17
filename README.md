# SiliconFlow Mistral OCR Adapter

将 Mistral 兼容的 `POST /v1/ocr` 转换为硅基流动
`deepseek-ai/DeepSeek-OCR` 的多模态 `POST /v1/chat/completions` 请求。

- 图片：一次上游 OCR 请求。
- PDF：本地按页渲染，每页一次上游 OCR 请求。
- 同一 PDF 默认最多并发两页。
- 所有文档共享全局上游并发、RPM 和 TPM 限制。
- 提供逐页 Markdown，并将 DeepSeek grounding 信息转换成 Mistral `blocks` 像素坐标。
- 检测到图像区域时返回 `images`；`include_image_base64=true` 时附带裁剪图 data URI。
- 上游返回 `finish_reason=length` 时明确报错，不静默返回截断内容。

为兼容 LiteLLM/Mistral SDK，服务会接收 annotation、confidence 等高级请求字段，但当前不执行
annotation 和 confidence。响应提供 `pages[].markdown`、`blocks`、`images`、页面尺寸和基础 usage。
DeepSeek 坐标以 0–999 表示，服务会按照渲染后页面尺寸换算成 Mistral 使用的像素坐标。

## 运行

```powershell
# 仓库已经准备好 .env；填写 SILICONFLOW_API_KEY 和 OCR_ADAPTER_API_KEY

python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

也可以使用 Docker：

```powershell
# 先编辑 .env
docker compose up --build
```

健康检查：

```text
GET /health/live
GET /health/ready
```

OpenAPI：`http://localhost:8000/docs`

## 调用

图片 URL：

```bash
curl http://localhost:8000/v1/ocr \
  -H "Authorization: Bearer $OCR_ADAPTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-ocr-latest",
    "document": {
      "type": "image_url",
      "image_url": "https://example.com/document.png"
    }
  }'
```

PDF Base64 使用 Mistral 的标准形式：

```json
{
  "model": "mistral-ocr-latest",
  "document": {
    "type": "document_url",
    "document_url": "data:application/pdf;base64,..."
  },
  "pages": "0,2-4"
}
```

页码从 0 开始。`pages` 可以是整数数组或 `"0,2-4"` 字符串。

## LiteLLM

```yaml
model_list:
  - model_name: deepseek-ocr
    litellm_params:
      model: mistral/mistral-ocr-latest
      api_base: http://ocr-adapter:8000/v1
      api_key: os.environ/OCR_ADAPTER_API_KEY
      max_parallel_requests: 8
    model_info:
      mode: ocr
```

LiteLLM 的 Mistral provider 要求配置一个非空 `api_key`。因此即使适配器只部署在私有网络，
也建议给 `OCR_ADAPTER_API_KEY` 设置一个随机值，并在 LiteLLM 中使用同一个值。

LiteLLM 调用：

```bash
curl http://localhost:4000/v1/ocr \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "document": {
      "type": "document_url",
      "document_url": "https://example.com/document.pdf"
    }
  }'
```

全文可以通过以下方式合并：

```python
full_text = "\n\n".join(page.markdown for page in response.pages)
```

## RPM/TPM 策略

默认上游额度为 1000 RPM、80000 TPM，并只使用 90%：

```text
有效 RPM：900
有效 TPM：72000
```

每次页面请求先按自适应估算值预留 token。收到硅基流动的
`usage.total_tokens` 后，将预留值替换为实际值，并用 EWMA 调整后续预留。
这样不会因为 `max_tokens=8192` 就按每页固定消耗 8192 token，同时为突发和统计延迟保留余量。

限流数据可以在 `GET /health/ready` 中查看。

## 重要配置

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `SILICONFLOW_RPM_LIMIT` | `1000` | 上游 RPM 配额 |
| `SILICONFLOW_TPM_LIMIT` | `80000` | 上游 TPM 配额 |
| `SILICONFLOW_RATE_UTILIZATION` | `0.90` | 实际使用配额比例 |
| `OCR_MAX_UPSTREAM_CONCURRENCY` | `8` | 全局上游并发 |
| `OCR_MAX_PAGE_CONCURRENCY_PER_DOCUMENT` | `2` | 单 PDF 页面并发 |
| `OCR_INITIAL_TOKENS_PER_PAGE` | `3000` | 初始页面 token 预留估算 |
| `OCR_MAX_OUTPUT_TOKENS` | `4096` | 请求的最大输出 token；为图像和提示词预留约一半 8K 上下文 |
| `OCR_RENDER_DPI` | `144` | PDF 渲染 DPI |
| `OCR_MAX_PDF_PAGES` | `100` | 单 PDF 最大页数 |
| `OCR_PROMPT_MODE` | `grounding` | `grounding` 或 `free` |

请保持单个 Uvicorn worker。当前并发和 RPM/TPM 状态在进程内维护；如果需要多实例，
应在 LiteLLM/Redis 层实施共享限流，或者将额度按实例拆分。
