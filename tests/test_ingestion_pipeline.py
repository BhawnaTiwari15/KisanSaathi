import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kisansathi.ingestion.models import IngestionError
from kisansathi.ingestion.pipeline import ingest_manifest, sha256_file

try:
    import pymupdf
except ImportError:
    pymupdf = None


def source_entry(source_id: str = "pilot", staging_path: str = "pilot.pdf") -> dict[str, object]:
    return {
        "source_id": source_id,
        "scheme": "pilot-scheme",
        "jurisdiction": "IN",
        "language": "en",
        "issuing_authority": "Test authority",
        "source_url": "https://example.gov.in/pilot.pdf",
        "title": "Synthetic test fixture",
        "accessed_at": "2026-10-05",
        "staging_path": staging_path,
    }


class PipelineValidationTests(unittest.TestCase):
    def test_missing_input_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "sources.json").write_text(
                json.dumps({"sources": [source_entry()]}), encoding="utf-8"
            )

            with self.assertRaisesRegex(IngestionError, "Missing PDF input"):
                ingest_manifest(project_root=root)

    def test_checksum_mismatch_in_existing_version_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incoming = root / "data" / "incoming"
            incoming.mkdir(parents=True)
            (incoming / "pilot.pdf").write_bytes(b"source bytes")
            checksum = hashlib.sha256(b"source bytes").hexdigest()
            raw_path = (
                root / "data" / "raw" / "pilot-scheme" / "IN" / "en" / "pilot" / f"{checksum}.pdf"
            )
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"corrupted existing version")
            (root / "data" / "sources.json").write_text(
                json.dumps({"sources": [source_entry()]}), encoding="utf-8"
            )

            with self.assertRaisesRegex(IngestionError, "failed checksum"):
                ingest_manifest(project_root=root)


@unittest.skipIf(pymupdf is None, "PyMuPDF must be installed to generate PDF fixtures")
class PipelinePdfTests(unittest.TestCase):
    def setUp(self) -> None:
        from pdf_test_utils import create_two_page_pdf

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.incoming = self.root / "data" / "incoming"
        self.incoming.mkdir(parents=True)
        create_two_page_pdf(self.incoming / "pilot.pdf")
        self.manifest_path = self.root / "data" / "sources.json"
        self.write_manifest([source_entry()])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(self, sources: list[dict[str, object]]) -> None:
        self.manifest_path.write_text(json.dumps({"sources": sources}), encoding="utf-8")

    def test_checksum_is_stable(self) -> None:
        path = self.incoming / "pilot.pdf"

        self.assertEqual(sha256_file(path), sha256_file(path))
        self.assertEqual(len(sha256_file(path)), 64)

    def test_same_source_and_checksum_is_idempotent(self) -> None:
        first = ingest_manifest(self.manifest_path, self.root)
        output_path = self.root / first.sources[0].processed_path
        first_bytes = output_path.read_bytes()
        second = ingest_manifest(self.manifest_path, self.root)

        self.assertFalse(first.sources[0].unchanged)
        self.assertTrue(second.sources[0].unchanged)
        self.assertEqual(output_path.read_bytes(), first_bytes)
        self.assertEqual(len(list((self.root / "data" / "raw").rglob("*.pdf"))), 1)

    def test_changed_checksum_creates_a_new_immutable_version(self) -> None:
        first = ingest_manifest(self.manifest_path, self.root)
        first_raw = self.root / first.sources[0].raw_path
        old_bytes = first_raw.read_bytes()

        from pdf_test_utils import create_two_page_pdf

        create_two_page_pdf(self.incoming / "replacement.pdf", headings=False)
        self.write_manifest([source_entry(staging_path="replacement.pdf")])
        second = ingest_manifest(self.manifest_path, self.root)

        self.assertNotEqual(first.sources[0].sha256, second.sources[0].sha256)
        self.assertTrue(second.sources[0].updated)
        self.assertTrue(first_raw.is_file())
        self.assertEqual(first_raw.read_bytes(), old_bytes)
        self.assertTrue((self.root / second.sources[0].raw_path).is_file())

    def test_duplicate_checksums_are_reported_for_distinct_sources(self) -> None:
        self.write_manifest([source_entry("pilot-a"), source_entry("pilot-b")])

        report = ingest_manifest(self.manifest_path, self.root)

        self.assertEqual(report.duplicate_checksums, (("pilot-a", "pilot-b"),))

    def test_empty_page_records_are_serialized(self) -> None:
        from pdf_test_utils import create_two_page_pdf

        create_two_page_pdf(self.incoming / "blank.pdf", text=False)
        self.write_manifest([source_entry(staging_path="blank.pdf")])

        report = ingest_manifest(self.manifest_path, self.root)
        output = (self.root / report.sources[0].processed_path).read_text(encoding="utf-8")
        records = [json.loads(line) for line in output.splitlines()]
        page_numbers = [
            record["page_number"] for record in records if record["record_type"] == "page"
        ]

        self.assertEqual(page_numbers, [1, 2])
        self.assertTrue(report.sources[0].warnings)

    def test_module_command_smoke_ingests_only_local_fixture(self) -> None:
        self.write_manifest([source_entry()])
        environment = dict(os.environ)
        source_dir = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source_dir, environment.get("PYTHONPATH", "")) if part
        )
        project_dir = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "kisansathi.ingestion",
                "--root",
                str(self.root),
                "--manifest",
                "data/sources.json",
            ],
            cwd=project_dir,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("pilot: ingested", completed.stdout)


if __name__ == "__main__":
    unittest.main()