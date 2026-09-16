import asyncio
import base64
import binascii
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from app.core.config import Settings
from app.core.errors import ServiceError

_DATA_URI = re.compile(r"^data:([^;,]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)
_SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/tiff",
    "image/bmp",
}


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    data: bytes
    mime_type: str

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def is_pdf(self) -> bool:
        return self.mime_type == "application/pdf"


def _sniff_mime(data: bytes, declared: str | None) -> str:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image/tiff"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    normalized = (declared or "").split(";", 1)[0].strip().lower()
    if normalized in _SUPPORTED_MIME_TYPES:
        return normalized
    raise ServiceError(415, "unsupported_document_type", "Only PDF and common raster image formats are supported")


class DocumentLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.download_timeout_seconds),
            follow_redirects=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def load(self, source: str) -> LoadedDocument:
        if source.startswith("data:"):
            return self._load_data_uri(source)
        return await self._download(source)

    def _load_data_uri(self, source: str) -> LoadedDocument:
        match = _DATA_URI.match(source)
        if not match:
            raise ServiceError(422, "invalid_data_uri", "Document data URI must use base64 encoding")
        declared_mime, encoded = match.groups()
        estimated_size = len(encoded) * 3 // 4
        if estimated_size > self._settings.max_document_bytes + 3:
            raise ServiceError(413, "document_too_large", "Document exceeds the configured size limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ServiceError(422, "invalid_base64", "Document contains invalid base64 data") from exc
        if len(data) > self._settings.max_document_bytes:
            raise ServiceError(413, "document_too_large", "Document exceeds the configured size limit")
        return LoadedDocument(data=data, mime_type=_sniff_mime(data, declared_mime))

    async def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ServiceError(422, "invalid_document_url", "Only HTTP and HTTPS document URLs are supported")
        if self._settings.allow_private_urls:
            return
        try:
            addresses = await asyncio.to_thread(
                socket.getaddrinfo,
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ServiceError(422, "document_host_unresolvable", "Document host could not be resolved") from exc
        for entry in addresses:
            address = ipaddress.ip_address(entry[4][0])
            if any(
                (
                    address.is_private,
                    address.is_loopback,
                    address.is_link_local,
                    address.is_multicast,
                    address.is_reserved,
                    address.is_unspecified,
                )
            ):
                raise ServiceError(422, "private_document_url", "Private or local document URLs are not allowed")

    async def _download(self, initial_url: str) -> LoadedDocument:
        current_url = initial_url
        for redirect_count in range(self._settings.max_redirects + 1):
            await self._validate_public_url(current_url)
            try:
                async with self._client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirect_count >= self._settings.max_redirects:
                            raise ServiceError(
                                422,
                                "document_redirect_error",
                                "Document URL has too many or invalid redirects",
                            )
                        current_url = urljoin(current_url, location)
                        continue
                    if response.status_code >= 400:
                        raise ServiceError(
                            422,
                            "document_download_failed",
                            f"Document URL returned HTTP {response.status_code}",
                        )
                    declared_length = response.headers.get("content-length")
                    if declared_length:
                        try:
                            content_length = int(declared_length)
                        except ValueError as exc:
                            raise ServiceError(
                                422,
                                "invalid_content_length",
                                "Document server returned an invalid Content-Length",
                            ) from exc
                        if content_length > self._settings.max_document_bytes:
                            raise ServiceError(
                                413,
                                "document_too_large",
                                "Document exceeds the configured size limit",
                            )
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self._settings.max_document_bytes:
                            raise ServiceError(
                                413,
                                "document_too_large",
                                "Document exceeds the configured size limit",
                            )
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    if not data:
                        raise ServiceError(422, "empty_document", "Downloaded document is empty")
                    mime = _sniff_mime(data, response.headers.get("content-type"))
                    return LoadedDocument(data=data, mime_type=mime)
            except ServiceError:
                raise
            except httpx.HTTPError as exc:
                raise ServiceError(422, "document_download_failed", "Failed to download the document") from exc
        raise ServiceError(422, "document_redirect_error", "Document URL has too many redirects")
