"""Tests for the registered source registry.

Fixtures are fixed literals, except where a test reads the real tracked manifest to
confirm it stays consistent with the ingested corpus.
"""

import json
from pathlib import Path
import unittest

from kisansathi.citations.models import CitationError
from kisansathi.citations.registry import ManifestSourceRegistry, SourceRegistry
from kisansathi.ingestion.models import ManifestEntry

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = REPO_ROOT / "data" / "sources.json"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "v1"


def make_entry(**overrides: object) -> ManifestEntry:
    """Return a valid ManifestEntry, overriding any field under test."""
    values: dict[str, object] = {
        "source_id": "pm-kisan-revised-faq",
        "scheme": "PM-KISAN",
        "jurisdiction": "IN",
        "language": "en",
        "issuing_authority": "Example Ministry (test fixture)",
        "source_url": "https://example.gov.in/faq.pdf",
        "title": "Example Scheme FAQ",
        "accessed_at": "2026-10-05",
        "staging_path": "faq.pdf",
    }
    values.update(overrides)
    return ManifestEntry(**values)  # type: ignore[arg-type]


def make_registry(*entries: ManifestEntry) -> ManifestSourceRegistry:
    """Return a registry over the given entries, defaulting to two distinct sources."""
    if not entries:
        entries = (
            make_entry(),
            make_entry(
                    source_id="mh-pdmc-notification-2026-27",
                    source_url="https://example.gov.in/p.pdf",
                ),
        )
    return ManifestSourceRegistry(entries)


class ManifestSourceRegistryTests(unittest.TestCase):
    def test_lookup_returns_the_registered_entry(self) -> None:
        registry = make_registry()

        self.assertEqual(registry.get("pm-kisan-revised-faq").title, "Example Scheme FAQ")

    def test_unknown_source_id_returns_none(self) -> None:
        self.assertIsNone(make_registry().get("not-registered"))

    def test_non_string_lookup_returns_none(self) -> None:
        self.assertIsNone(make_registry().get(None))

    def test_source_ids_are_deterministic(self) -> None:
        registry = make_registry()

        self.assertEqual(registry.source_ids(), registry.source_ids())
        self.assertEqual(
            registry.source_ids(),
            ("mh-pdmc-notification-2026-27", "pm-kisan-revised-faq"),
        )

    def test_registry_rejects_duplicate_source_ids(self) -> None:
        with self.assertRaisesRegex(CitationError, "duplicate source_id"):
            ManifestSourceRegistry((make_entry(), make_entry()))

    def test_registry_requires_at_least_one_source(self) -> None:
        with self.assertRaisesRegex(CitationError, "at least one source"):
            ManifestSourceRegistry(())

    def test_registry_rejects_non_entry_members(self) -> None:
        with self.assertRaisesRegex(CitationError, "must be ManifestEntry"):
            ManifestSourceRegistry(("not-an-entry",))  # type: ignore[arg-type]

    def test_registry_satisfies_the_protocol(self) -> None:
        self.assertIsInstance(make_registry(), SourceRegistry)

    def test_contains_and_len(self) -> None:
        registry = make_registry()

        self.assertIn("pm-kisan-revised-faq", registry)
        self.assertNotIn("nope", registry)
        self.assertNotIn(None, registry)
        self.assertEqual(len(registry), 2)

    def test_registered_url_is_always_absolute_https(self) -> None:
        registry = make_registry()

        for source_id in registry.source_ids():
            with self.subTest(source_id=source_id):
                self.assertTrue(registry.get(source_id).source_url.startswith("https://"))

    def test_registered_authority_is_never_blank(self) -> None:
        registry = make_registry()

        for source_id in registry.source_ids():
            with self.subTest(source_id=source_id):
                self.assertTrue(registry.get(source_id).issuing_authority.strip())


class ManifestEntryUnusableUrlTests(unittest.TestCase):
    def test_registry_cannot_contain_a_source_without_a_url(self) -> None:
        """A missing URL is unrepresentable, so no placeholder can ever be fabricated."""
        for url in ("", "   ", "http://insecure.gov.in/faq.pdf", "faq.pdf"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    make_entry(source_url=url)


class RealManifestTests(unittest.TestCase):
    """Reads the tracked manifest to keep the registry consistent with the corpus."""

    def setUp(self) -> None:
        self.registry = ManifestSourceRegistry.from_manifest(SOURCES_PATH)

    def test_manifest_loads_both_corpus_sources(self) -> None:
        self.assertEqual(
            self.registry.source_ids(),
            ("mh-pdmc-notification-2026-27", "pm-kisan-revised-faq"),
        )

    def test_every_registered_source_has_complete_citation_metadata(self) -> None:
        for source_id in self.registry.source_ids():
            with self.subTest(source_id=source_id):
                entry = self.registry.get(source_id)
                self.assertTrue(entry.title.strip())
                self.assertTrue(entry.source_url.startswith("https://"))
                self.assertTrue(entry.issuing_authority.strip())

    def test_pm_kisan_authority_caveat_is_preserved_verbatim(self) -> None:
        entry = self.registry.get("pm-kisan-revised-faq")

        self.assertIn("document-level issuer not established", entry.issuing_authority)

    def test_from_manifest_rejects_a_missing_file(self) -> None:
        from kisansathi.ingestion.manifest import ManifestError

        with self.assertRaises(ManifestError):
            ManifestSourceRegistry.from_manifest(REPO_ROOT / "data" / "does-not-exist.json")


class ManifestCorpusAgreementTests(unittest.TestCase):
    """The registry must agree with what ingestion actually stored.

    Skipped when the gitignored processed corpus is absent; this is an opt-in integrity
    check, not a build dependency.
    """

    def document_metadata(self, source_id: str) -> dict | None:
        folder = PROCESSED_ROOT / source_id
        if not folder.is_dir():
            return None
        files = sorted(folder.glob("*.jsonl"))
        if not files:
            return None
        for line in files[0].read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("record_type") == "document":
                metadata = record.get("metadata")
                return metadata if isinstance(metadata, dict) else None
        return None

    def test_registry_agrees_with_ingested_document_metadata(self) -> None:
        registry = ManifestSourceRegistry.from_manifest(SOURCES_PATH)
        checked = 0

        for source_id in registry.source_ids():
            metadata = self.document_metadata(source_id)
            if metadata is None:
                continue
            checked += 1
            entry = registry.get(source_id)
            for field in (
                "title",
                "source_url",
                "issuing_authority",
                "scheme",
                "jurisdiction",
                "language",
                "accessed_at",
            ):
                with self.subTest(source_id=source_id, field=field):
                    self.assertEqual(getattr(entry, field), metadata.get(field))

        if checked == 0:
            self.skipTest("processed corpus not available locally")

    def test_registry_covers_every_ingested_source(self) -> None:
        registry = ManifestSourceRegistry.from_manifest(SOURCES_PATH)
        if not PROCESSED_ROOT.is_dir():
            self.skipTest("processed corpus not available locally")

        ingested = {folder.name for folder in PROCESSED_ROOT.iterdir() if folder.is_dir()}

        self.assertTrue(ingested.issubset(set(registry.source_ids())))


if __name__ == "__main__":
    unittest.main()
