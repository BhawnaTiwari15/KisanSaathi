import json
from pathlib import Path
import tempfile
import unittest

from kisansathi.ingestion.chunking import chunk_pages
from kisansathi.ingestion.manifest import ManifestError, load_manifest
from kisansathi.ingestion.models import ExtractedPage, ManifestEntry, TextBlock
from kisansathi.ingestion.pdf import PdfExtractionError, extract_pdf

try:
    import pymupdf
except ImportError:
    pymupdf = None


def valid_source(**overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "source_id": "pm-kisan-guidelines",
        "scheme": "pm-kisan",
        "jurisdiction": "IN",
        "language": "en",
        "issuing_authority": "Ministry of Agriculture",
        "source_url": "https://example.gov.in/guidelines.pdf",
        "title": "Guidelines",
        "accessed_at": "2026-10-05",
        "staging_path": "pilot.pdf",
    }
    source.update(overrides)
    return source


class ManifestTests(unittest.TestCase):
    def test_loads_empty_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text('{"sources": []}', encoding="utf-8")

            self.assertEqual(load_manifest(path), ())

    def test_loads_valid_entry_and_optional_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps({"sources": [valid_source()]}), encoding="utf-8")

            entry = load_manifest(path)[0]

        self.assertEqual(entry.source_id, "pm-kisan-guidelines")
        self.assertIsNone(entry.publication_date)

    def test_rejects_missing_and_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps({"sources": [{"source_id": "x"}]}), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(path)

            path.write_text(
                json.dumps({"sources": [valid_source(unexpected="value")]}), encoding="utf-8"
            )
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_rejects_unsafe_relative_paths(self) -> None:
        unsafe_paths = ("../secret.pdf", "folder/../../secret.pdf", "C:/secret.pdf", "folder\\file.pdf")
        for unsafe_path in unsafe_paths:
            with self.subTest(path=unsafe_path), self.assertRaisesRegex(ValueError, "staging_path"):
                ManifestEntry(**valid_source(staging_path=unsafe_path))

    def test_rejects_duplicate_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(
                json.dumps({"sources": [valid_source(), valid_source(title="Second")]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "must be unique"):
                load_manifest(path)


class ChunkingTests(unittest.TestCase):
    def test_heading_chunks_preserve_page_spans_and_stable_ids(self) -> None:
        checksum = "a" * 64
        pages = (
            ExtractedPage(
                "source",
                checksum,
                1,
                "Overview\n\nFirst paragraph.",
                (
                    TextBlock("Overview", 0, 20, True),
                    TextBlock("First paragraph.", 1, 11, False),
                ),
            ),
            ExtractedPage(
                "source",
                checksum,
                2,
                "Continuation.\n\nEligibility\n\nSecond section.",
                (
                    TextBlock("Continuation.", 0, 11, False),
                    TextBlock("Eligibility", 1, 20, True),
                    TextBlock("Second section.", 2, 11, False),
                ),
            ),
        )

        first = chunk_pages(pages)
        second = chunk_pages(pages)

        overview = next(chunk for chunk in first if chunk.heading == "Overview")
        self.assertEqual((overview.page_start, overview.page_end), (1, 2))
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertEqual([chunk.heading for chunk in first], ["Overview", "Eligibility"])

    def test_uses_paragraph_fallback_without_heading_signals(self) -> None:
        checksum = "b" * 64
        pages = (
            ExtractedPage(
                "source",
                checksum,
                1,
                "First paragraph.",
                (TextBlock("First paragraph.", 0, 11, False),),
            ),
            ExtractedPage(
                "source",
                checksum,
                2,
                "Second paragraph.",
                (TextBlock("Second paragraph.", 0, 11, False),),
            ),
        )

        chunks = chunk_pages(pages)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].heading)
        self.assertEqual((chunks[0].page_start, chunks[0].page_end), (1, 2))

    def test_splits_oversized_paragraphs(self) -> None:
        text = "alpha beta gamma delta"
        page = ExtractedPage("source", "c" * 64, 1, text, (TextBlock(text, 0, 11, False),))

        chunks = chunk_pages((page,), max_chars=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 10 for chunk in chunks))


@unittest.skipIf(pymupdf is None, "PyMuPDF must be installed to generate PDF fixtures")
class PdfExtractionTests(unittest.TestCase):
    def test_extracts_two_pages_with_one_based_numbers_and_headings(self) -> None:
        from pdf_test_utils import create_two_page_pdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two-pages.pdf"
            create_two_page_pdf(path)
            result = extract_pdf(path, "source", "d" * 64)
            chunks = chunk_pages(result.pages)

        self.assertEqual([page.page_number for page in result.pages], [1, 2])
        self.assertIn("First page paragraph", result.pages[0].text)
        self.assertTrue(any(block.is_bold for block in result.pages[0].blocks))
        overview = next(chunk for chunk in chunks if chunk.heading == "Overview")
        self.assertEqual((overview.page_start, overview.page_end), (1, 2))
        self.assertIn("Eligibility", [chunk.heading for chunk in chunks])

    def test_preserves_textless_page_records_and_warns(self) -> None:
        from pdf_test_utils import create_two_page_pdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.pdf"
            create_two_page_pdf(path, text=False)
            result = extract_pdf(path, "source", "e" * 64)

        self.assertEqual([page.page_number for page in result.pages], [1, 2])
        self.assertTrue(all(not page.text for page in result.pages))
        self.assertTrue(any("no extractable text" in warning.lower() for warning in result.warnings))

    def test_rejects_unreadable_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unreadable_path = Path(directory) / "unreadable.pdf"
            unreadable_path.write_bytes(b"not a PDF")

            with self.assertRaises(PdfExtractionError):
                extract_pdf(unreadable_path, "source", "f" * 64)

    def test_rejects_encrypted_pdf_when_supported(self) -> None:
        from pdf_test_utils import create_encrypted_pdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encrypted.pdf"
            if not create_encrypted_pdf(path):
                self.skipTest("Installed PyMuPDF build does not expose AES-256 encryption")
            with self.assertRaisesRegex(PdfExtractionError, "Encrypted PDF"):
                extract_pdf(path, "source", "1" * 64)


if __name__ == "__main__":
    unittest.main()