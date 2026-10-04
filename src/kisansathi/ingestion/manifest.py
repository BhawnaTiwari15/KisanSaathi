"""Load and validate the manually curated source manifest."""

import json
from pathlib import Path
from typing import Any

from kisansathi.ingestion.models import IngestionError, ManifestEntry


class ManifestError(IngestionError):
    """Raised for malformed or unsafe source manifests."""


_REQUIRED_FIELDS = {
    "source_id",
    "scheme",
    "jurisdiction",
    "language",
    "issuing_authority",
    "source_url",
    "title",
    "accessed_at",
    "staging_path",
}
_OPTIONAL_FIELDS = {
    "publication_date",
    "publication_year",
    "effective_from",
    "effective_to",
}


def _entry_from_dict(value: Any) -> ManifestEntry:
    if not isinstance(value, dict):
        raise ManifestError("Each source entry must be a JSON object")
    missing = _REQUIRED_FIELDS - value.keys()
    unknown = value.keys() - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if missing:
        raise ManifestError(f"Source entry is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ManifestError(f"Source entry has unknown fields: {', '.join(sorted(unknown))}")
    try:
        return ManifestEntry(**value)
    except (TypeError, ValueError) as error:
        raise ManifestError(f"Invalid source entry: {error}") from error


def load_manifest(path: str | Path) -> tuple[ManifestEntry, ...]:
    """Load a JSON manifest containing a single ``sources`` array."""
    manifest_path = Path(path)
    try:
        content = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestError(f"Cannot read source manifest {manifest_path}: {error}") from error
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ManifestError(f"Invalid JSON in source manifest {manifest_path}: {error}") from error

    if not isinstance(data, dict) or set(data) != {"sources"}:
        raise ManifestError("Manifest must be a JSON object with only a 'sources' array")
    if not isinstance(data["sources"], list):
        raise ManifestError("Manifest 'sources' must be an array")

    entries = tuple(_entry_from_dict(item) for item in data["sources"])
    source_ids = [entry.source_id for entry in entries]
    if len(source_ids) != len(set(source_ids)):
        raise ManifestError("Manifest source_id values must be unique")
    return entries