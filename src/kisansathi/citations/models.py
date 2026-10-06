"""Provenance contracts for the citation layer.

The resolver is deliberately strict. Anything it cannot attribute to registered source
metadata or to stored chunk provenance is refused rather than approximated, so an
answer can never cite a document, page, or authority that does not exist.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kisansathi.domain.schemas import Citation


class CitationError(ValueError):
    """Raised when provenance cannot be resolved into a trustworthy citation."""


class UnknownSourceError(CitationError):
    """Raised when a source_id is absent from the registered source metadata."""


class ChunkSourceMismatchError(CitationError):
    """Raised when a chunk_id does not belong to the source it is cited against."""


class InvalidPageError(CitationError):
    """Raised when a cited page number is not a valid, ascending, one-based range."""


class InvalidEvidenceError(CitationError):
    """Raised when the evidence itself is structurally unusable."""


class SourceMetadataMismatchError(CitationError):
    """Raised when carried metadata contradicts the registered source metadata."""


class UnknownCitationIdError(CitationError):
    """Raised when generated text references a citation that was never resolved."""


@runtime_checkable
class EvidenceLike(Protocol):
    """The minimal provenance a citation needs, matched structurally.

    Declared as a Protocol so ``kisansathi.eligibility`` evidence resolves through this
    layer without either module importing the other.
    """

    @property
    def source_id(self) -> str: ...

    @property
    def sha256(self) -> str: ...

    @property
    def chunk_id(self) -> str: ...

    @property
    def page_start(self) -> int: ...

    @property
    def page_end(self) -> int: ...


@dataclass(frozen=True, slots=True)
class RejectedCitation:
    """Provenance that could not be cited, recorded with the reason it was refused."""

    source_id: str
    chunk_id: str | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise CitationError("RejectedCitation source_id must be a non-empty string")
        if self.chunk_id is not None and (
            not isinstance(self.chunk_id, str) or not self.chunk_id.strip()
        ):
            raise CitationError("RejectedCitation chunk_id must be a non-empty string or None")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise CitationError("RejectedCitation reason must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CitationBatch:
    """The validated citations for one answer, plus everything refused along the way.

    ``citations`` doubles as the allowlist that post-generation validation checks
    generated citation identifiers against, so an answer can only cite evidence that was
    actually resolved first.
    """

    citations: tuple[Citation, ...] = ()
    rejected: tuple[RejectedCitation, ...] = ()

    def __post_init__(self) -> None:
        for name in ("citations", "rejected"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise CitationError(f"CitationBatch {name} must be a tuple")
            expected = Citation if name == "citations" else RejectedCitation
            for item in value:
                if not isinstance(item, expected):
                    raise CitationError(f"CitationBatch {name} must contain {expected.__name__}")

    def by_chunk_id(self) -> dict[str, Citation]:
        """Return the resolvable citations keyed by chunk id, for validation lookups."""
        return {
            citation.chunk_id: citation
            for citation in self.citations
            if citation.chunk_id is not None
        }

    @property
    def citation_ids(self) -> frozenset[str]:
        """Return the chunk ids a generated answer is allowed to reference."""
        return frozenset(self.by_chunk_id())
