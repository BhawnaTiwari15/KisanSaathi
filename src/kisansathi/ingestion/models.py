"""Data contracts for source manifests and extracted document content."""

from dataclasses import dataclass
from datetime import date
from pathlib import PurePosixPath, PureWindowsPath
import re
from urllib.parse import urlparse


class IngestionError(ValueError):
    """Raised when a source cannot be safely or reproducibly ingested."""


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_JURISDICTION = re.compile(r"^[A-Z]{2}(?:-[A-Z0-9]{2,3})*$")
_LANGUAGE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$|^mul$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_date(value: str | None, field_name: str) -> None:
    if value is None:
        return
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD)") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    source_id: str
    scheme: str
    jurisdiction: str
    language: str
    issuing_authority: str
    source_url: str
    title: str
    accessed_at: str
    staging_path: str
    publication_date: str | None = None
    publication_year: int | None = None
    effective_from: str | None = None
    effective_to: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "scheme",
            "jurisdiction",
            "language",
            "issuing_authority",
            "source_url",
            "title",
            "accessed_at",
            "staging_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if not _SAFE_ID.fullmatch(self.source_id):
            raise ValueError("source_id may contain only letters, numbers, dots, underscores, hyphens")
        if not _SAFE_ID.fullmatch(self.scheme):
            raise ValueError("scheme must be a path-safe slug")
        if not _JURISDICTION.fullmatch(self.jurisdiction):
            raise ValueError("jurisdiction must be an uppercase country/state code")
        if not _LANGUAGE.fullmatch(self.language):
            raise ValueError("language must be a BCP-47-style tag or 'mul'")

        parsed_url = urlparse(self.source_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("source_url must be an absolute HTTPS URL")

        _validate_date(self.accessed_at, "accessed_at")
        _validate_date(self.publication_date, "publication_date")
        _validate_date(self.effective_from, "effective_from")
        _validate_date(self.effective_to, "effective_to")
        if self.publication_year is not None:
            if isinstance(self.publication_year, bool) or not 1000 <= self.publication_year <= 9999:
                raise ValueError("publication_year must be a four-digit year")
            if self.publication_date and int(self.publication_date[:4]) != self.publication_year:
                raise ValueError("publication_year must match publication_date")
        if self.effective_from and self.effective_to and self.effective_to < self.effective_from:
            raise ValueError("effective_to must not be before effective_from")

        path = self.staging_path
        if "\\" in path:
            raise ValueError("staging_path must use forward slashes")
        relative_path = PurePosixPath(path)
        windows_path = PureWindowsPath(path)
        if (
            relative_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("staging_path must be a safe relative path under data/incoming")


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    source_id: str
    scheme: str
    jurisdiction: str
    language: str
    issuing_authority: str
    source_url: str
    title: str
    accessed_at: str
    sha256: str
    local_path: str
    publication_date: str | None = None
    publication_year: int | None = None
    effective_from: str | None = None
    effective_to: str | None = None

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if not self.local_path or "\\" in self.local_path:
            raise ValueError("local_path must be a non-empty POSIX relative path")
        path = PurePosixPath(self.local_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("local_path must be a safe relative path")

    @classmethod
    def from_entry(cls, entry: ManifestEntry, sha256: str, local_path: str) -> "SourceMetadata":
        return cls(
            source_id=entry.source_id,
            scheme=entry.scheme,
            jurisdiction=entry.jurisdiction,
            language=entry.language,
            issuing_authority=entry.issuing_authority,
            source_url=entry.source_url,
            title=entry.title,
            accessed_at=entry.accessed_at,
            sha256=sha256,
            local_path=local_path,
            publication_date=entry.publication_date,
            publication_year=entry.publication_year,
            effective_from=entry.effective_from,
            effective_to=entry.effective_to,
        )


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    block_number: int
    font_size: float
    is_bold: bool


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    source_id: str
    sha256: str
    page_number: int
    text: str
    blocks: tuple[TextBlock, ...] = ()

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be one-based")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    pages: tuple[ExtractedPage, ...]
    parser_version: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    source_id: str
    sha256: str
    ordinal: int
    heading: str | None
    text: str
    page_start: int
    page_end: int

    def __post_init__(self) -> None:
        if not self.chunk_id.strip() or not self.text.strip():
            raise ValueError("chunk_id and text must not be empty")
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("chunk page range must use positive, ascending page numbers")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")