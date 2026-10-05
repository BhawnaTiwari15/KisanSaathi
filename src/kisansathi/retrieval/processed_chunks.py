"""Read and validate retrieval payloads from processed ingestion JSONL."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path


_PAYLOAD_FIELDS = (
    "chunk_id",
    "source_id",
    "sha256",
    "scheme",
    "jurisdiction",
    "language",
    "title",
    "page_start",
    "page_end",
    "heading",
    "text",
)
_STRING_FIELDS = (
    "chunk_id",
    "source_id",
    "sha256",
    "scheme",
    "jurisdiction",
    "language",
    "title",
    "text",
)


@dataclass(frozen=True, slots=True)
class ProcessedChunks:
    payloads: tuple[dict[str, object], ...]
    chunks_read: int
    skipped_records: int
    errors: tuple[str, ...]


def read_chunk_payloads(jsonl_path: str | Path) -> ProcessedChunks:
    """Read chunk records and merge document metadata into complete payloads."""
    path = Path(jsonl_path)
    errors: list[str] = []
    records: list[tuple[int, dict[str, object]]] = []
    skipped_records = 0
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"line {line_number}: invalid JSON: {error.msg}")
                    skipped_records += 1
                    continue
                if not isinstance(record, dict):
                    errors.append(f"line {line_number}: record must be a JSON object")
                    skipped_records += 1
                    continue
                records.append((line_number, record))
    except OSError as error:
        return ProcessedChunks(
            payloads=(),
            chunks_read=0,
            skipped_records=skipped_records,
            errors=(f"cannot read {path}: {error}",),
        )

    document_metadata: Mapping[str, object] = {}
    for line_number, record in records:
        if record.get("record_type") != "document":
            continue
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            document_metadata = metadata
            break
        errors.append(f"line {line_number}: document metadata must be a JSON object")

    payloads: list[dict[str, object]] = []
    chunks_read = 0
    for line_number, record in records:
        if record.get("record_type") != "chunk":
            skipped_records += 1
            continue
        chunks_read += 1
        try:
            payloads.append(payload_for_chunk(record, document_metadata))
        except ValueError as error:
            errors.append(f"line {line_number}: {error}")
            skipped_records += 1

    return ProcessedChunks(
        payloads=tuple(payloads),
        chunks_read=chunks_read,
        skipped_records=skipped_records,
        errors=tuple(errors),
    )


def payload_for_chunk(
    chunk: Mapping[str, object], document_metadata: Mapping[str, object]
) -> dict[str, object]:
    """Build the complete validated retrieval payload for one chunk record."""
    payload: dict[str, object] = {}
    for field_name in _PAYLOAD_FIELDS:
        if field_name in chunk:
            payload[field_name] = chunk[field_name]
        elif field_name in document_metadata:
            payload[field_name] = document_metadata[field_name]
        else:
            raise ValueError(f"chunk is missing required field {field_name!r}")

    return validate_retrieval_payload(payload)


def validate_retrieval_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy a complete retrieval payload."""
    normalized = dict(payload)
    missing = set(_PAYLOAD_FIELDS) - normalized.keys()
    if missing:
        raise ValueError(f"chunk is missing required field {sorted(missing)[0]!r}")
    for field_name in _STRING_FIELDS:
        value = normalized[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"chunk field {field_name!r} must be a non-empty string")
    for field_name in ("page_start", "page_end"):
        value = normalized[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"chunk field {field_name!r} must be a positive integer")
    if normalized["page_end"] < normalized["page_start"]:
        raise ValueError("chunk page_end must not be before page_start")
    if normalized["heading"] is not None and not isinstance(normalized["heading"], str):
        raise ValueError("chunk field 'heading' must be a string or null")
    return normalized