"""The registered source registry: the canonical home of document metadata.

``data/sources.json`` is the canonical source of document-level metadata. It is tracked
in version control, whereas ``data/processed/`` is gitignored local build output and
cannot be relied on. Document fields are therefore never duplicated into every stored
chunk payload; they are joined in here, at citation time, from the curated manifest.

No new metadata model is introduced. ``ManifestEntry`` already carries every field a
citation needs and already validates it, including the requirement that ``source_url``
be an absolute HTTPS URL.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from kisansathi.citations.models import CitationError
from kisansathi.ingestion.manifest import load_manifest
from kisansathi.ingestion.models import ManifestEntry


@runtime_checkable
class SourceRegistry(Protocol):
    """Read-only lookup of registered sources by source_id."""

    def get(self, source_id: str) -> ManifestEntry | None:
        """Return the registered entry, or None when the source is not registered."""
        ...

    def source_ids(self) -> tuple[str, ...]:
        """Return every registered source id, in a deterministic order."""
        ...


class ManifestSourceRegistry:
    """An immutable source_id to ManifestEntry index built from a curated manifest."""

    def __init__(self, entries: Sequence[ManifestEntry]) -> None:
        by_id: dict[str, ManifestEntry] = {}
        for entry in entries:
            if not isinstance(entry, ManifestEntry):
                raise CitationError("ManifestSourceRegistry entries must be ManifestEntry objects")
            if entry.source_id in by_id:
                raise CitationError(f"duplicate source_id in registry: {entry.source_id!r}")
            by_id[entry.source_id] = entry
        if not by_id:
            raise CitationError("ManifestSourceRegistry requires at least one source")
        self._entries = by_id

    @classmethod
    def from_manifest(cls, path: str | Path) -> "ManifestSourceRegistry":
        """Build a registry from a source manifest such as data/sources.json."""
        return cls(load_manifest(path))

    def get(self, source_id: str) -> ManifestEntry | None:
        if not isinstance(source_id, str):
            return None
        return self._entries.get(source_id)

    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def __contains__(self, source_id: object) -> bool:
        return isinstance(source_id, str) and source_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)
