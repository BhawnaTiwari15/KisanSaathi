import unittest

import numpy as np

from kisansathi.retrieval.retriever import DenseRetriever
from kisansathi.retrieval.vector_store import SearchResult


class FakeEmbeddingService:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def embed_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return self.vector


class FakeVectorStore:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.query_vector: object | None = None
        self.limit: int | None = None

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 5,
    ) -> tuple[SearchResult, ...]:
        self.query_vector = query_vector
        self.limit = limit
        return self.results


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = np.array([0.25, 0.75], dtype=np.float32)
        self.results = (
            SearchResult(
                score=0.93,
                payload={"chunk_id": "chunk-1", "text": "Relevant content"},
            ),
            SearchResult(
                score=0.81,
                payload={"chunk_id": "chunk-2", "text": "Other content"},
            ),
        )
        self.embeddings = FakeEmbeddingService(self.vector)
        self.vector_store = FakeVectorStore(self.results)
        self.retriever = DenseRetriever(self.embeddings, self.vector_store)

    def test_encodes_query_searches_with_top_k_and_preserves_results(self) -> None:
        results = self.retriever.retrieve("How do I check my eligibility?", top_k=2)

        self.assertEqual(self.embeddings.queries, ["How do I check my eligibility?"])
        self.assertIs(self.vector_store.query_vector, self.vector)
        self.assertEqual(self.vector_store.limit, 2)
        self.assertIs(results, self.results)
        self.assertEqual(results[0].score, 0.93)
        self.assertEqual(results[0].payload["chunk_id"], "chunk-1")

    def test_uses_default_top_k(self) -> None:
        self.retriever.retrieve("question")

        self.assertEqual(self.vector_store.limit, 5)

    def test_uses_configured_default_top_k(self) -> None:
        retriever = DenseRetriever(self.embeddings, self.vector_store, default_top_k=7)

        retriever.retrieve("question")

        self.assertEqual(self.vector_store.limit, 7)

    def test_rejects_blank_query_before_embedding_or_search(self) -> None:
        for query in ("", "  \n"):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, "query must not be blank"):
                self.retriever.retrieve(query)

        self.assertEqual(self.embeddings.queries, [])
        self.assertIsNone(self.vector_store.query_vector)

    def test_rejects_invalid_top_k(self) -> None:
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(
                ValueError, "top_k must be a positive integer"
            ):
                self.retriever.retrieve("question", top_k=top_k)

        self.assertEqual(self.embeddings.queries, [])
        self.assertIsNone(self.vector_store.query_vector)

    def test_rejects_invalid_configured_default_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
            DenseRetriever(self.embeddings, self.vector_store, default_top_k=0)


class FakeVectorStoreWithClose:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.query_vector: object | None = None
        self.limit: int | None = None
        self.closed = False
        self.close_count = 0

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 5,
    ) -> tuple[SearchResult, ...]:
        self.query_vector = query_vector
        self.limit = limit
        return self.results

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


class DenseRetrieverCloseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = np.array([0.25, 0.75], dtype=np.float32)
        self.results = (
            SearchResult(
                score=0.93,
                payload={"chunk_id": "chunk-1", "text": "Relevant content"},
            ),
            SearchResult(
                score=0.81,
                payload={"chunk_id": "chunk-2", "text": "Other content"},
            ),
        )
        self.embeddings = FakeEmbeddingService(self.vector)
        self.vector_store = FakeVectorStoreWithClose(self.results)
        self.retriever = DenseRetriever(self.embeddings, self.vector_store)

    def test_close_calls_vector_store_close(self) -> None:
        self.retriever.close()
        self.assertTrue(self.vector_store.closed)
        self.assertEqual(self.vector_store.close_count, 1)

    def test_close_idempotent(self) -> None:
        self.retriever.close()
        self.retriever.close()
        # Second close should not call vector_store.close again
        self.assertEqual(self.vector_store.close_count, 1)

    def test_close_safe_when_vector_store_has_no_close(self) -> None:
        retriever = DenseRetriever(self.embeddings, FakeVectorStore(self.results))
        # Should not raise
        retriever.close()
        retriever.close()