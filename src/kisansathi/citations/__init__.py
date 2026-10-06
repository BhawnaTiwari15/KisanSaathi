"""Citation and provenance resolution for stored evidence.

Document metadata is joined from the registered source manifest at citation time rather
than duplicated into stored chunk payloads. Resolution is fail-closed: provenance that
cannot be fully attributed to a registered source is refused, never approximated.
"""

from kisansathi.citations.models import (
    ChunkSourceMismatchError,
    CitationBatch,
    CitationError,
    EvidenceLike,
    InvalidEvidenceError,
    InvalidPageError,
    RejectedCitation,
    SourceMetadataMismatchError,
    UnknownCitationIdError,
    UnknownSourceError,
)
from kisansathi.citations.registry import ManifestSourceRegistry, SourceRegistry
from kisansathi.citations.resolver import CitationResolver, validate_referenced_citations

__all__ = [
    "ChunkSourceMismatchError",
    "CitationBatch",
    "CitationError",
    "CitationResolver",
    "EvidenceLike",
    "InvalidEvidenceError",
    "InvalidPageError",
    "ManifestSourceRegistry",
    "RejectedCitation",
    "SourceMetadataMismatchError",
    "SourceRegistry",
    "UnknownCitationIdError",
    "UnknownSourceError",
    "validate_referenced_citations",
]
