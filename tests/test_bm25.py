import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from kisansathi.retrieval.bm25_retriever import BM25Retriever
from kisansathi.retrieval.bm25_store import BM25Store
from kisansathi.retrieval.vector_store import SearchResult


def document_record() -> dict[str, object]:
    return {
        "record_type": "document",
        "metadata": {
            "scheme": "PM-KISAN",
            "jurisdiction": "IN",
            "language": "en",
            "title": "PM-KISAN scheme guide",
        },
    }


def chunk_record(chunk_id: str, text: str, heading: str) -> dict[str, object]:
    return {
        "record_type": "chunk",
        "chunk_id": chunk_id,
        "source_id": "pm-kisan-guide",
        "sha256": "b" * 64,
        "ordinal": 0,
        "heading": heading,
        "text": text,
        "page_start": 2,
        "page_end": 3,
    }


class FakeBM25Store:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.query: str | None = None
        self.limit: int | None = None

    def search(self, query: str, *, limit: int = 5) -> tuple[SearchResult, ...]:
        self.query = query
        self.limit = limit
        return self.results


class BM25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "bm25.sqlite3"
        self.store = BM25Store(database_path=self.database_path)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary_directory.cleanup()

    def test_indexes_jsonl_chunks_only_and_preserves_complete_payload(self) -> None:
        jsonl_path = Path(self.temporary_directory.name) / "chunks.jsonl"
        records = (
            document_record(),
            {"record_type": "page", "page_number": 1, "text": "not searchable"},
            chunk_record(
                "eligible-farmers",
                "Eligible landholding farmers can receive PM-KISAN benefits.",
                "Eligibility",
            ),
            chunk_record("payment-details", "Installment payment schedule details.", "Payments"),
            chunk_record("registration", "Farmer registration requires account details.", "Register"),
        )
        jsonl_path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        summary = self.store.index_file(jsonl_path)
        results = self.store.search("landholding", limit=3)

        self.assertEqual(summary.chunks_read, 3)
        self.assertEqual(summary.indexed, 3)
        self.assertEqual(summary.skipped_records, 2)
        self.assertEqual(summary.errors, ())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload["chunk_id"], "eligible-farmers")
        self.assertGreater(results[0].score, 0)
        self.assertEqual(
            results[0].payload,
            {
                "chunk_id": "eligible-farmers",
                "source_id": "pm-kisan-guide",
                "sha256": "b" * 64,
                "scheme": "PM-KISAN",
                "jurisdiction": "IN",
                "language": "en",
                "title": "PM-KISAN scheme guide",
                "page_start": 2,
                "page_end": 3,
                "heading": "Eligibility",
                "text": "Eligible landholding farmers can receive PM-KISAN benefits.",
            },
        )

    def test_reindexing_same_chunks_is_idempotent_and_persistent(self) -> None:
        jsonl_path = Path(self.temporary_directory.name) / "chunks.jsonl"
        jsonl_path.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    document_record(),
                    chunk_record("eligible", "Landholding farmers are eligible.", "Eligibility"),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        first = self.store.index_file(jsonl_path)
        second = self.store.index_file(jsonl_path)
        self.store.close()
        self.store = BM25Store(database_path=self.database_path)
        results = self.store.search("landholding", limit=5)

        self.assertEqual((first.indexed, second.indexed), (1, 1))
        self.assertEqual([result.payload["chunk_id"] for result in results], ["eligible"])

    def test_bm25_retriever_forwards_query_limit_and_results(self) -> None:
        results = (SearchResult(score=0.75, payload={"chunk_id": "one", "text": "match"}),)
        fake_store = FakeBM25Store(results)
        retriever = BM25Retriever(fake_store, default_top_k=4)

        returned = retriever.retrieve("exact keyword", top_k=2)

        self.assertEqual(fake_store.query, "exact keyword")
        self.assertEqual(fake_store.limit, 2)
        self.assertIs(returned, results)

    def test_retriever_uses_default_top_k_and_rejects_invalid_values(self) -> None:
        fake_store = FakeBM25Store(())
        retriever = BM25Retriever(fake_store, default_top_k=7)
        retriever.retrieve("question")
        self.assertEqual(fake_store.limit, 7)

        for top_k in (0, -1, True, 1.2):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(
                ValueError, "top_k must be a positive integer"
            ):
                retriever.retrieve("question", top_k=top_k)
        with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
            BM25Retriever(fake_store, default_top_k=0)

    def test_rejects_blank_queries(self) -> None:
        fake_store = FakeBM25Store(())
        retriever = BM25Retriever(fake_store)

        for query in ("", "  \n"):
            with self.subTest(query=query), self.assertRaisesRegex(
                ValueError, "query must not be blank"
            ):
                retriever.retrieve(query)
        self.assertIsNone(fake_store.query)

    def test_fts_query_is_safely_bound_and_punctuation_only_query_is_empty(self) -> None:
        result = self.store.search('" OR *; DROP TABLE bm25_chunks; --')
        self.assertEqual(result, ())
        self.assertIsNotNone(
            self.store._connection.execute("SELECT count(*) FROM bm25_chunks").fetchone()
        )

    def test_reports_clear_error_when_fts5_is_unavailable(self) -> None:
        class MissingFTS5Connection:
            row_factory = None
            closed = False

            def execute(self, _statement: str):
                raise sqlite3.OperationalError("no such module: fts5")

            def close(self) -> None:
                self.closed = True

        connection = MissingFTS5Connection()
        with patch(
            "kisansathi.retrieval.bm25_store.sqlite3.connect", return_value=connection
        ):
            with self.assertRaisesRegex(RuntimeError, "SQLite FTS5 is required"):
                BM25Store(database_path=":memory:")
        self.assertTrue(connection.closed)

    def test_upsert_is_transactional_when_a_write_fails(self) -> None:
        original_connection = self.store._connection

        class FailSecondInsert:
            def __init__(self) -> None:
                self.insert_count = 0

            def __enter__(self):
                original_connection.__enter__()
                return self

            def __exit__(self, *exception_info: object):
                return original_connection.__exit__(*exception_info)

            def execute(self, statement: str, parameters: tuple[object, ...] = ()):
                if statement.lstrip().startswith("INSERT INTO bm25_chunks"):
                    self.insert_count += 1
                    if self.insert_count == 2:
                        raise sqlite3.IntegrityError("simulated second-row failure")
                return original_connection.execute(statement, parameters)

        connection_proxy = FailSecondInsert()
        self.store._connection = connection_proxy
        payloads = [
            {
                "chunk_id": "first",
                "source_id": "test-source",
                "sha256": "c" * 64,
                "scheme": "PM-KISAN",
                "jurisdiction": "IN",
                "language": "en",
                "title": "Test guide",
                "page_start": 1,
                "page_end": 1,
                "heading": None,
                "text": "rollbackmarker first",
            },
            {
                "chunk_id": "second",
                "source_id": "test-source",
                "sha256": "c" * 64,
                "scheme": "PM-KISAN",
                "jurisdiction": "IN",
                "language": "en",
                "title": "Test guide",
                "page_start": 2,
                "page_end": 2,
                "heading": None,
                "text": "rollbackmarker second",
            },
        ]

        with self.assertRaisesRegex(sqlite3.IntegrityError, "simulated second-row failure"):
            self.store.upsert(payloads)

        self.store._connection = original_connection
        self.assertEqual(self.store.search("rollbackmarker"), ())


if __name__ == "__main__":
    unittest.main()