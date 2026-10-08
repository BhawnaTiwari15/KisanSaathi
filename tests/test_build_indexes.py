"""Tests for the deployment index-build entry point."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kisansathi.build_indexes import find_chunk_files, run_build
from kisansathi.retrieval.bm25_store import BM25Store
from kisansathi.retrieval.vector_store import QdrantVectorStore


def document_record() -> dict[str, object]:
    return {
        "record_type": "document",
        "metadata": {
            "scheme": "PM-KISAN",
            "jurisdiction": "IN",
            "language": "en",
            "title": "Test source title",
            "source_id": "test-source",
            "sha256": "a" * 64,
        },
    }


def chunk_record(chunk_id: str) -> dict[str, object]:
    return {
        "record_type": "chunk",
        "chunk_id": chunk_id,
        "source_id": "test-source",
        "sha256": "a" * 64,
        "ordinal": 0,
        "heading": "Eligibility",
        "text": f"Text for {chunk_id}",
        "page_start": 2,
        "page_end": 3,
    }


class FakeEmbeddingService:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        self.calls.append(list(documents))
        return np.ones((len(documents), self.dimension), dtype=np.float32)


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, object]] = {}
        self.upserted = 0

    def upsert(
        self,
        vectors: object,
        payloads: list[dict[str, object]],
    ) -> tuple[str, ...]:
        point_ids = []
        for payload in payloads:
            point_id = QdrantVectorStore.point_id_for_chunk(str(payload["chunk_id"]))
            self.points[point_id] = dict(payload)
            point_ids.append(point_id)
        self.upserted += len(payloads)
        return tuple(point_ids)


class BuildIndexesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "corpus"
        self.root.mkdir()
        self.embeddings = FakeEmbeddingService()
        self.vector_store = FakeVectorStore()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_jsonl(self, relative_path: str, *records: object) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def write_two_chunk_file(self, relative_path: str) -> Path:
        return self.write_jsonl(
            relative_path,
            document_record(),
            chunk_record("chunk-1"),
            chunk_record("chunk-2"),
        )

    def test_find_chunk_files_fails_on_missing_root(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "corpus root does not exist"):
            find_chunk_files(self.root / "absent")

    def test_find_chunk_files_returns_sorted_recursive_files(self) -> None:
        second = self.write_two_chunk_file("nested/inner.jsonl")
        first = self.write_two_chunk_file("a.jsonl")

        files = find_chunk_files(self.root)

        self.assertEqual(files, (first, second))

    def test_run_build_rejects_no_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one index target"):
            run_build(self.root)

    def test_run_build_rejects_unpaired_dense_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "both an embedding service and a vector store"):
            run_build(self.root, embedding_service=self.embeddings)

    def test_run_build_rejects_empty_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "no processed .jsonl files"):
            run_build(self.root, embedding_service=self.embeddings, vector_store=self.vector_store)

    def test_dense_only_build_indexes_all_files(self) -> None:
        self.write_two_chunk_file("a.jsonl")
        self.write_two_chunk_file("nested/b.jsonl")

        summary = run_build(
            self.root,
            embedding_service=self.embeddings,
            vector_store=self.vector_store,
        )

        self.assertEqual(summary.files_indexed, 2)
        self.assertEqual(summary.chunks_read, 4)
        self.assertEqual((summary.dense_embedded, summary.dense_upserted), (4, 4))
        self.assertEqual(summary.bm25_indexed, 0)
        self.assertEqual(summary.errors, ())
        self.assertEqual(self.vector_store.upserted, 4)
        self.assertEqual(len(self.vector_store.points), 2)
        self.assertEqual(self.embeddings.calls[0], ["Text for chunk-1", "Text for chunk-2"])

    def test_bm25_only_build_indexes_all_files(self) -> None:
        self.write_jsonl(  # distinct chunk IDs per file exercise accumulation
            "a.jsonl",
            document_record(),
            chunk_record("chunk-1"),
            chunk_record("chunk-2"),
        )
        self.write_jsonl(
            "b.jsonl",
            document_record(),
            chunk_record("chunk-3"),
            chunk_record("chunk-4"),
        )
        bm25_path = Path(self.temporary_directory.name) / "bm25.sqlite3"
        with BM25Store(database_path=str(bm25_path)) as store:

            summary = run_build(self.root, bm25_store=store)

            self.assertEqual(summary.chunks_read, 4)
            self.assertEqual(summary.bm25_indexed, 4)
            self.assertEqual((summary.dense_embedded, summary.dense_upserted), (0, 0))
            self.assertEqual(summary.errors, ())
            self.assertEqual(len(store.search("Text")), 4)

    def test_build_both_systems(self) -> None:
        self.write_two_chunk_file("a.jsonl")
        self.write_two_chunk_file("b.jsonl")
        bm25_path = Path(self.temporary_directory.name) / "bm25.sqlite3"
        with BM25Store(database_path=str(bm25_path)) as store:

            summary = run_build(
                self.root,
                embedding_service=self.embeddings,
                vector_store=self.vector_store,
                bm25_store=store,
            )

            self.assertEqual(summary.chunks_read, 4)
            self.assertEqual((summary.dense_embedded, summary.dense_upserted), (4, 4))
            self.assertEqual(summary.bm25_indexed, 4)
            self.assertEqual(summary.errors, ())

    def test_reports_per_file_errors_without_aborting(self) -> None:
        good = self.write_two_chunk_file("good.jsonl")
        bad = self.root / "bad.jsonl"
        bad.write_text("{not json}\n", encoding="utf-8")

        summary = run_build(
            self.root,
            embedding_service=self.embeddings,
            vector_store=self.vector_store,
            bm25_store=BM25Store(database_path=":memory:"),
        )

        self.assertEqual(summary.files_indexed, 2)
        self.assertEqual(summary.dense_upserted, 2)
        self.assertEqual(summary.bm25_indexed, 2)
        prefixes = (str(good), str(bad))
        self.assertTrue(
            summary.errors
            and any(message.startswith(f"{bad}:") for message in summary.errors)
            and all(
                any(message.startswith(prefix) for prefix in prefixes)
                for message in summary.errors
            )
        )


if __name__ == "__main__":
    unittest.main()