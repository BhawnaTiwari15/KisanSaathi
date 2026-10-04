"""Reproducible local ingestion and immutable source-version storage."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from kisansathi.ingestion.chunking import DEFAULT_MAX_CHARS, chunk_pages
from kisansathi.ingestion.manifest import load_manifest
from kisansathi.ingestion.models import IngestionError, ManifestEntry, SourceMetadata
from kisansathi.ingestion.pdf import extract_pdf


PIPELINE_VERSION = "v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class IngestedSource:
    source_id: str
    sha256: str
    raw_path: str
    processed_path: str
    unchanged: bool
    updated: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestionReport:
    sources: tuple[IngestedSource, ...]
    duplicate_checksums: tuple[tuple[str, ...], ...] = ()


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file using bounded memory."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise IngestionError(f"Cannot read source file {path}: {error}") from error
    return digest.hexdigest()


def _resolve_staging_path(root: Path, entry: ManifestEntry) -> Path:
    incoming = (root / "data" / "incoming").resolve()
    candidate = (incoming / Path(*entry.staging_path.split("/"))).resolve()
    try:
        candidate.relative_to(incoming)
    except ValueError as error:
        raise IngestionError(f"Staging path escapes data/incoming: {entry.staging_path}") from error
    if not candidate.is_file():
        raise IngestionError(f"Missing PDF input for {entry.source_id}: {candidate}")
    return candidate


def _copy_immutable(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise IngestionError(f"Existing immutable raw version failed checksum: {destination}")
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copyfile(source, temporary_path)
        if sha256_file(temporary_path) != expected_sha256:
            raise IngestionError(f"Source changed while it was being copied: {source}")
        if destination.exists():
            if sha256_file(destination) != expected_sha256:
                raise IngestionError(f"Existing immutable raw version failed checksum: {destination}")
        else:
            os.replace(temporary_path, destination)
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def serialize_jsonl(
    metadata: SourceMetadata,
    pages: tuple,
    chunks: tuple,
    parser_version: str,
    warnings: tuple[str, ...],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> bytes:
    """Serialize document, page, and chunk records with stable JSON encoding."""
    records: list[dict[str, object]] = [
        {
            "record_type": "document",
            "metadata": asdict(metadata),
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "parser_version": parser_version,
            "chunk_max_chars": max_chars,
            "warnings": list(warnings),
        }
    ]
    records.extend(
        {
            "record_type": "page",
            "source_id": page.source_id,
            "sha256": page.sha256,
            "page_number": page.page_number,
            "text": page.text,
        }
        for page in pages
    )
    records.extend({"record_type": "chunk", **asdict(chunk)} for chunk in chunks)
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise IngestionError(f"Existing processed version differs; refusing overwrite: {path}")
        return False

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        if path.exists():
            if path.read_bytes() != content:
                raise IngestionError(f"Existing processed version differs; refusing overwrite: {path}")
        else:
            os.replace(temporary_path, path)
            temporary_path = None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ingest_manifest(
    manifest_path: str | Path = "data/sources.json",
    project_root: str | Path = ".",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> IngestionReport:
    """Ingest local PDFs listed in a manifest without network access."""
    root = Path(project_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    entries = load_manifest(manifest)

    results: list[IngestedSource] = []
    hashes_to_sources: dict[str, list[str]] = {}
    for entry in entries:
        staged_pdf = _resolve_staging_path(root, entry)
        checksum = sha256_file(staged_pdf)
        hashes_to_sources.setdefault(checksum, []).append(entry.source_id)

        raw_relative = Path(
            "data",
            "raw",
            entry.scheme,
            entry.jurisdiction,
            entry.language,
            entry.source_id,
            f"{checksum}.pdf",
        )
        raw_path = root / raw_relative
        prior_versions = (
            tuple(path for path in raw_path.parent.glob("*.pdf") if path != raw_path)
            if raw_path.parent.exists()
            else ()
        )
        _copy_immutable(staged_pdf, raw_path, checksum)

        metadata = SourceMetadata.from_entry(
            entry,
            sha256=checksum,
            local_path=raw_relative.as_posix(),
        )
        extraction = extract_pdf(raw_path, entry.source_id, checksum)
        chunks = chunk_pages(extraction.pages, max_chars=max_chars)
        processed_relative = Path(
            "data",
            "processed",
            PIPELINE_VERSION,
            entry.source_id,
            f"{checksum}.jsonl",
        )
        content = serialize_jsonl(
            metadata,
            extraction.pages,
            chunks,
            extraction.parser_version,
            extraction.warnings,
            max_chars,
        )
        was_created = _write_immutable(root / processed_relative, content)
        results.append(
            IngestedSource(
                source_id=entry.source_id,
                sha256=checksum,
                raw_path=raw_relative.as_posix(),
                processed_path=processed_relative.as_posix(),
                unchanged=not was_created,
                updated=bool(prior_versions) and was_created,
                warnings=extraction.warnings,
            )
        )

    duplicates = tuple(
        tuple(sorted(source_ids))
        for source_ids in hashes_to_sources.values()
        if len(source_ids) > 1
    )
    return IngestionReport(tuple(results), duplicates)