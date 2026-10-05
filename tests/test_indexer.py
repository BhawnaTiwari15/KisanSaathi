import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kisansathi.retrieval.indexer import CorpusIndexer
from kisansathi.retrieval.vector_store import QdrantVectorStore


class FakeEmbeddingService:
    def __init__(self, dimension: int = 3, error: Exception | None = None) -> None:
        self.dimension = dimension
        self.error = error
        self.calls: list[list[str]] = []

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        self.calls.append(list(documents))
        if self.error is not None:
            raise self.error
        return np.ones((len(documents), self.dimension), dtype=np.float32)


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, tuple[list[float], dict[str, object]]] = {}
        self.upsert_calls = 0

    def upsert(
        self,
        vectors: np.ndarray,
        payloads: list[dict[str, object]],
    ) -> tuple[str, ...]:
        self.upsert_calls += 1
        point_ids = []
        for vector, payload in zip(vectors, payloads, strict=True):
            point_id = QdrantVectorStore.point_id_for_chunk(str(payload["chunk_id"]))
            self.points[point_id] = (vector.tolist(), dict(payload))
            point_ids.append(point_id)
        return tuple(point_ids)


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


def chunk_record(chunk_id: str = "chunk-1", **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
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
    record.update(overrides)
    return record


class CorpusIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "processed.jsonl"
        self.embeddings = FakeEmbeddingService()
        self.vector_store = FakeVectorStore()
        self.indexer = CorpusIndexer(self.embeddings, self.vector_store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_records(self, *records: object) -> None:
        self.path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_skips_document_and_page_records(self) -> None:
        self.write_records(
            document_record(),
            {"record_type": "page", "page_number": 1, "text": "Do not embed this"},
            chunk_record(),
        )

        summary = self.indexer.index_file(self.path)

        self.assertEqual((summary.chunks_read, summary.embedded, summary.upserted), (1, 1, 1))
        self.assertEqual(summary.skipped_records, 2)
        self.assertEqual(summary.errors, ())
        self.assertEqual(self.embeddings.calls, [["Text for chunk-1"]])

    def test_preserves_chunk_and_document_retrieval_metadata(self) -> None:
        self.write_records(document_record(), chunk_record())

        summary = self.indexer.index_file(self.path)

        self.assertEqual(summary.errors, ())
        payload = next(iter(self.vector_store.points.values()))[1]
        self.assertEqual(
            payload,
            {
                "chunk_id": "chunk-1",
                "source_id": "test-source",
                "sha256": "a" * 64,
                "scheme": "PM-KISAN",
                "jurisdiction": "IN",
                "language": "en",
                "title": "Test source title",
                "page_start": 2,
                "page_end": 3,
                "heading": "Eligibility",
                "text": "Text for chunk-1",
            },
        )

    def test_repeated_indexing_is_idempotent_with_deterministic_ids(self) -> None:
        self.write_records(document_record(), chunk_record())

        first = self.indexer.index_file(self.path)
        first_id = next(iter(self.vector_store.points))
        second = self.indexer.index_file(self.path)

        self.assertEqual(first_id, QdrantVectorStore.point_id_for_chunk("chunk-1"))
        self.assertEqual(next(iter(self.vector_store.points)), first_id)
        self.assertEqual(len(self.vector_store.points), 1)
        self.assertEqual((first.upserted, second.upserted), (1, 1))

    def test_reports_batch_counts_and_non_chunk_skips(self) -> None:
        self.indexer = CorpusIndexer(self.embeddings, self.vector_store, batch_size=1)
        self.write_records(
            document_record(),
            {"record_type": "page", "page_number": 1, "text": "not indexed"},
            chunk_record("chunk-1"),
            {"record_type": "page", "page_number": 2, "text": "not indexed"},
            chunk_record("chunk-2"),
        )

        summary = self.indexer.index_file(self.path)

        self.assertEqual(summary.chunks_read, 2)
        self.assertEqual(summary.embedded, 2)
        self.assertEqual(summary.upserted, 2)
        self.assertEqual(summary.skipped_records, 3)
        self.assertEqual(len(self.embeddings.calls), 2)

    def test_malformed_records_are_reported_and_skipped_clearly(self) -> None:
        self.path.write_text(
            json.dumps(document_record())
            + "\n"
            '{broken json\n'
            + json.dumps(chunk_record("bad-chunk", text=""))
            + "\n",
            encoding="utf-8",
        )

        summary = self.indexer.index_file(self.path)

        self.assertEqual(summary.chunks_read, 1)
        self.assertEqual(summary.skipped_records, 3)
        self.assertEqual(summary.embedded, 0)
        self.assertEqual(summary.upserted, 0)
        self.assertIn("line 2: invalid JSON", summary.errors[0])
        self.assertIn("chunk field 'text' must be a non-empty string", summary.errors[1])
        self.assertEqual(self.embeddings.calls, [])

    def test_embedding_errors_are_included_in_summary(self) -> None:
        self.embeddings = FakeEmbeddingService(error=RuntimeError("fake embedding failure"))
        self.indexer = CorpusIndexer(self.embeddings, self.vector_store)
        self.write_records(document_record(), chunk_record())

        summary = self.indexer.index_file(self.path)

        self.assertEqual(summary.chunks_read, 1)
        self.assertEqual(summary.embedded, 0)
        self.assertEqual(summary.upserted, 0)
        self.assertIn("fake embedding failure", summary.errors[0])


if __name__ == "__main__":
    unittest.main()