from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    siliconflow_api_key: str = Field(default="", validation_alias="SILICONFLOW_API_KEY")
    siliconflow_api_url: str = Field(
        default="https://api.siliconflow.cn/v1/chat/completions",
        validation_alias="SILICONFLOW_API_URL",
    )
    siliconflow_model: str = Field(default="deepseek-ai/DeepSeek-OCR", validation_alias="SILICONFLOW_MODEL")
    siliconflow_rpm_limit: int = Field(default=1000, validation_alias="SILICONFLOW_RPM_LIMIT")
    siliconflow_tpm_limit: int = Field(default=80000, validation_alias="SILICONFLOW_TPM_LIMIT")
    siliconflow_rate_utilization: float = Field(default=0.90, validation_alias="SILICONFLOW_RATE_UTILIZATION")

    max_upstream_concurrency: int = Field(default=8, validation_alias="OCR_MAX_UPSTREAM_CONCURRENCY")
    max_page_concurrency_per_document: int = Field(default=2, validation_alias="OCR_MAX_PAGE_CONCURRENCY_PER_DOCUMENT")
    max_active_documents: int = Field(default=8, validation_alias="OCR_MAX_ACTIVE_DOCUMENTS")
    max_queued_documents: int = Field(default=32, validation_alias="OCR_MAX_QUEUED_DOCUMENTS")
    document_queue_timeout_seconds: float = Field(default=30, validation_alias="OCR_DOCUMENT_QUEUE_TIMEOUT_SECONDS")

    initial_tokens_per_page: int = Field(default=3000, validation_alias="OCR_INITIAL_TOKENS_PER_PAGE")
    min_tokens_per_page: int = Field(default=512, validation_alias="OCR_MIN_TOKENS_PER_PAGE")
    max_tokens_per_page: int = Field(default=8192, validation_alias="OCR_MAX_TOKENS_PER_PAGE")
    token_reservation_factor: float = Field(default=1.25, validation_alias="OCR_TOKEN_RESERVATION_FACTOR")
    rate_window_seconds: float = Field(default=60, validation_alias="OCR_RATE_WINDOW_SECONDS")

    max_output_tokens: int = Field(default=4096, validation_alias="OCR_MAX_OUTPUT_TOKENS")
    max_retries: int = Field(default=2, validation_alias="OCR_MAX_RETRIES")
    upstream_timeout_seconds: float = Field(default=180, validation_alias="OCR_UPSTREAM_TIMEOUT_SECONDS")
    prompt_mode: Literal["grounding", "free"] = Field(default="grounding", validation_alias="OCR_PROMPT_MODE")

    max_document_bytes: int = Field(default=50 * 1024 * 1024, validation_alias="OCR_MAX_DOCUMENT_BYTES")
    max_request_body_bytes: int = Field(default=70 * 1024 * 1024, validation_alias="OCR_MAX_REQUEST_BODY_BYTES")
    max_pdf_pages: int = Field(default=100, validation_alias="OCR_MAX_PDF_PAGES")
    render_dpi: int = Field(default=144, validation_alias="OCR_RENDER_DPI")
    render_format: Literal["PNG", "JPEG"] = Field(default="PNG", validation_alias="OCR_RENDER_FORMAT")
    allow_private_urls: bool = Field(default=False, validation_alias="OCR_ALLOW_PRIVATE_URLS")
    download_timeout_seconds: float = Field(default=30, validation_alias="OCR_DOWNLOAD_TIMEOUT_SECONDS")
    max_redirects: int = Field(default=3, validation_alias="OCR_MAX_REDIRECTS")

    adapter_api_key: str = Field(default="", validation_alias="OCR_ADAPTER_API_KEY")

    @field_validator(
        "siliconflow_rpm_limit",
        "siliconflow_tpm_limit",
        "max_upstream_concurrency",
        "max_page_concurrency_per_document",
        "max_active_documents",
        "max_queued_documents",
        "initial_tokens_per_page",
        "min_tokens_per_page",
        "max_tokens_per_page",
        "max_output_tokens",
        "max_document_bytes",
        "max_request_body_bytes",
        "max_pdf_pages",
        "render_dpi",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @field_validator("siliconflow_rate_utilization")
    @classmethod
    def valid_utilization(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("must be in the interval (0, 1]")
        return value

    @property
    def prompt(self) -> str:
        if self.prompt_mode == "free":
            return "<image>\nFree OCR."
        return "<image>\n<|grounding|>Convert the document to markdown."


@lru_cache
def get_settings() -> Settings:
    return Settings()
