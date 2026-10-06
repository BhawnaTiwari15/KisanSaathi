"""Resolve stored chunk and eligibility provenance into validated public citations.

This is a join, not a producer. Identity and page information come from the evidence that
retrieval or eligibility already produced; document metadata comes from the registered
source manifest. The resolver never constructs a URL, an authority, a title, or a page,
and it never reads chunk text, so no source wording can be altered or repaired on the way
to a rendered citation.

Behaviour is fail-closed: provenance that cannot be fully attributed is refused rather
than approximated. Callers that must not lose an entire answer to one bad chunk use
``resolve_payloads``, which records refusals in a ``CitationBatch``.
"""

from collections.abc import Iterable, Mapping, Sequence
import re

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
from kisansathi.citations.registry import SourceRegistry
from kisansathi.domain.schemas import Citation

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_ID = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*):([0-9a-f]{24})$")

# The payload keys the resolver reads. ``text`` is deliberately absent: the citation
# layer must never touch source wording, which keeps OCR damage unaltered by construction.
_CONSUMED_PAYLOAD_FIELDS = ("source_id", "chunk_id", "sha256", "page_start", "page_end")
_OPTIONAL_CROSS_CHECK_FIELD = "title"


class CitationResolver:
    """Join retrieval or eligibility provenance with registered source metadata."""

    def __init__(self, registry: SourceRegistry) -> None:
        if not isinstance(registry, SourceRegistry):
            raise CitationError("CitationResolver requires a SourceRegistry")
        self._registry = registry

    @property
    def registry(self) -> SourceRegistry:
        return self._registry

    def resolve_payload(self, payload: Mapping[str, object]) -> Citation:
        """Resolve one retrieval payload (a ``SearchResult.payload``) into a Citation."""
        if not isinstance(payload, Mapping):
            raise InvalidEvidenceError("citation payload must be a mapping")

        for field_name in _CONSUMED_PAYLOAD_FIELDS:
            if field_name not in payload:
                raise InvalidEvidenceError(f"citation payload is missing {field_name!r}")

        expected_title = payload.get(_OPTIONAL_CROSS_CHECK_FIELD)
        if expected_title is not None and not isinstance(expected_title, str):
            raise InvalidEvidenceError("citation payload title must be a string or absent")

        return self._resolve(
            source_id=_require_text(payload["source_id"], "source_id"),
            chunk_id=_require_text(payload["chunk_id"], "chunk_id"),
            sha256=_require_text(payload["sha256"], "sha256"),
            page_start=_require_page(payload["page_start"], "page_start"),
            page_end=_require_page(payload["page_end"], "page_end"),
            expected_title=expected_title,
        )

    def resolve_evidence(self, evidence: EvidenceLike) -> Citation:
        """Resolve eligibility provenance, such as an ``EvidenceRef``, into a Citation."""
        for field_name in _CONSUMED_PAYLOAD_FIELDS:
            if not hasattr(evidence, field_name):
                raise InvalidEvidenceError(f"evidence is missing {field_name!r}")

        return self._resolve(
            source_id=_require_text(evidence.source_id, "source_id"),
            chunk_id=_require_text(evidence.chunk_id, "chunk_id"),
            sha256=_require_text(evidence.sha256, "sha256"),
            page_start=_require_page(evidence.page_start, "page_start"),
            page_end=_require_page(evidence.page_end, "page_end"),
            expected_title=None,
        )

    def resolve_payloads(
        self, payloads: Iterable[Mapping[str, object]]
    ) -> CitationBatch:
        """Resolve many payloads, collecting refusals instead of raising on the first."""
        if isinstance(payloads, (str, bytes)) or not isinstance(payloads, Iterable):
            raise InvalidEvidenceError("payloads must be an iterable of mappings")

        citations: list[Citation] = []
        rejected: list[RejectedCitation] = []
        seen: set[tuple[str, str]] = set()

        for payload in payloads:
            try:
                citation = self.resolve_payload(payload)
            except CitationError as error:
                rejected.append(_rejection_from(payload, error))
                continue
            key = _identity_key(citation)
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)

        return CitationBatch(citations=tuple(citations), rejected=tuple(rejected))

    def resolve_evidence_batch(self, evidence: Sequence[EvidenceLike]) -> CitationBatch:
        """Resolve many eligibility evidence items into one validated batch."""
        citations: list[Citation] = []
        rejected: list[RejectedCitation] = []
        seen: set[tuple[str, str]] = set()

        for item in evidence:
            try:
                citation = self.resolve_evidence(item)
            except CitationError as error:
                source_id = getattr(item, "source_id", None)
                rejected.append(
                    RejectedCitation(
                        source_id=source_id if isinstance(source_id, str) else "<unknown>",
                        chunk_id=getattr(item, "chunk_id", None),
                        reason=str(error),
                    )
                )
                continue
            key = _identity_key(citation)
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)

        return CitationBatch(citations=tuple(citations), rejected=tuple(rejected))

    def _resolve(
        self,
        *,
        source_id: str,
        chunk_id: str,
        sha256: str,
        page_start: int,
        page_end: int,
        expected_title: str | None,
    ) -> Citation:
        entry = self._registry.get(source_id)
        if entry is None:
            raise UnknownSourceError(f"source_id is not registered: {source_id!r}")

        match = _CHUNK_ID.fullmatch(chunk_id)
        if match is None:
            raise ChunkSourceMismatchError(
                f"chunk_id must be '<source_id>:<24 hex>': {chunk_id!r}"
            )
        if match.group(1) != source_id:
            raise ChunkSourceMismatchError(
                f"chunk_id {chunk_id!r} does not belong to source {source_id!r}"
            )

        if _SHA256.fullmatch(sha256) is None:
            raise InvalidEvidenceError("sha256 must be 64 lowercase hexadecimal characters")

        if page_end < page_start:
            raise InvalidPageError(
                f"page_end {page_end} must not be before page_start {page_start}"
            )

        if expected_title is not None and expected_title.strip() and expected_title != entry.title:
            raise SourceMetadataMismatchError(
                f"title does not match registered metadata for {source_id!r}"
            )

        # page_end is left unset for a single-page span so equality stays meaningful.
        return Citation(
            source_id=entry.source_id,
            title=entry.title,
            url=entry.source_url,
            page_number=page_start,
            page_end=page_end if page_end != page_start else None,
            issuing_authority=entry.issuing_authority,
            chunk_id=chunk_id,
        )


def validate_referenced_citations(
    referenced_ids: Iterable[str],
    batch: CitationBatch,
) -> tuple[Citation, ...]:
    """Second gate: check that generated text only cites already-resolved evidence.

    This is the post-generation counterpart to pre-generation resolution. A generated
    answer may reference only chunk ids present in ``batch``; anything else means the
    answer invented a citation, and the whole answer must be discarded rather than
    published with a fabricated reference.
    """
    if not isinstance(batch, CitationBatch):
        raise CitationError("validate_referenced_citations requires a CitationBatch")
    if isinstance(referenced_ids, (str, bytes)) or not isinstance(referenced_ids, Iterable):
        raise CitationError("referenced_ids must be an iterable of citation identifiers")

    known = batch.by_chunk_id()
    ordered: list[Citation] = []
    seen: set[str] = set()

    for identifier in referenced_ids:
        candidate = identifier.strip() if isinstance(identifier, str) else ""
        if not candidate:
            raise UnknownCitationIdError("generated text referenced an empty citation id")
        if candidate not in known:
            raise UnknownCitationIdError(
                f"generated text referenced an unresolved citation id: {candidate!r}"
            )
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(known[candidate])

    return tuple(ordered)


def _identity_key(citation: Citation) -> tuple[str, str]:
    """Return the dedup identity for a citation.

    A chunk_id is content-addressed over its own page span, so one chunk is one place in
    one document. Citing it twice, for instance from dense and BM25 fusion, is the same
    citation and collapses to one entry.
    """
    return citation.source_id, citation.chunk_id or ""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEvidenceError(f"{field_name} must be a non-empty string")
    return value


def _require_page(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidPageError(f"{field_name} must be a positive one-based integer")
    return value


def _rejection_from(
    payload: Mapping[str, object], error: CitationError
) -> RejectedCitation:
    source_id = payload.get("source_id")
    chunk_id = payload.get("chunk_id")
    return RejectedCitation(
        source_id=source_id if isinstance(source_id, str) and source_id.strip() else "<unknown>",
        chunk_id=chunk_id if isinstance(chunk_id, str) and chunk_id.strip() else None,
        reason=str(error),
    )
