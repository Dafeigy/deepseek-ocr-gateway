from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentURL(BaseModel):
    type: Literal["document_url"]
    document_url: str


class ImageURL(BaseModel):
    type: Literal["image_url"]
    image_url: str


OCRDocument = Annotated[DocumentURL | ImageURL, Field(discriminator="type")]


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    document: OCRDocument
    pages: list[int] | str | None = None
    include_image_base64: bool | None = None
    image_limit: int | None = None
    image_min_size: int | None = None
    bbox_annotation_format: dict[str, Any] | None = None
    document_annotation_format: dict[str, Any] | None = None
    document_annotation_prompt: str | None = None
    extract_header: bool = False
    extract_footer: bool = False
    table_format: Literal["markdown", "html"] | None = None
    confidence_scores_granularity: Literal["word", "page", "block"] | None = None
    include_blocks: bool | None = None


class OCRPageDimensions(BaseModel):
    dpi: int | None = None
    height: int | None = None
    width: int | None = None


class OCRImageObject(BaseModel):
    id: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    image_base64: str | None = None


class OCRBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    content: str
    image_id: str | None = None


class OCRPage(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    markdown: str
    images: list[OCRImageObject] = Field(default_factory=list)
    blocks: list[OCRBlock] | None = None
    dimensions: OCRPageDimensions | None = None


class OCRUsageInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    pages_processed: int | None = None
    doc_size_bytes: int | None = None
    total_tokens: int | None = None


class OCRResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    pages: list[OCRPage]
    model: str
    document_annotation: Any | None = None
    usage_info: OCRUsageInfo | None = None
    object: Literal["ocr"] = "ocr"


def document_source(document: OCRDocument) -> str:
    if isinstance(document, DocumentURL):
        return document.document_url
    return document.image_url
