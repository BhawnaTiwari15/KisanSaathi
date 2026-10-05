import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from kisansathi.config import Settings
from kisansathi.retrieval.vector_store import QdrantVectorStore, SearchResult


def valid_payload(chunk_id: str = "chunk-1") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": "source-1",
        "sha256": "a" * 64,
        "scheme": "scheme-1",
        "jurisdiction": "IN",
        "language": "en",
        "title": "Source title",
        "page_start": 1,
        "page_end": 2,
        "heading": "Eligibility",
        "text": "Chunk content",
    }


class FakeBackend:
    def __init__(self, collection_exists: bool = False) -> None:
        self.exists = collection_exists
        self.created: tuple[str, int] | None = None
        self.points: list[object] = []
        self.upsert_collection: str | None = None
        self.search_query: tuple[str, list[float], int] | None = None
        self.results = [SearchResult(score=0.91, payload=valid_payload())]
        self.closed = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        self.created = (collection_name, vector_size)
        self.exists = True

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upsert_collection = collection_name
        self.points = points

    def search(self, collection_name: str, query_vector: list[float], limit: int):
        self.search_query = (collection_name, query_vector, limit)
        return self.results[:limit]

    def close(self) -> None:
        self.closed = True


class VectorStoreTests(unittest.TestCase):
    def test_creates_missing_collection_with_configured_dimension(self) -> None:
        backend = FakeBackend()
        settings = Settings(
            embedding_dimension=3,
            qdrant_collection_name="test_chunks",
            qdrant_storage_path="tmp/qdrant",
        )

        store = QdrantVectorStore(settings, backend=backend)

        self.assertEqual(backend.created, ("test_chunks", 3))
        self.assertTrue(store.collection_exists())
        self.assertEqual(store.vector_dimension, 3)

    def test_does_not_recreate_existing_collection(self) -> None:
        backend = FakeBackend(collection_exists=True)

        QdrantVectorStore(Settings(embedding_dimension=3), backend=backend)

        self.assertIsNone(backend.created)

    def test_point_ids_are_deterministic_from_chunk_id(self) -> None:
        first = QdrantVectorStore.point_id_for_chunk("source:chunk:1")
        second = QdrantVectorStore.point_id_for_chunk("source:chunk:1")

        self.assertEqual(first, second)
        self.assertNotEqual(first, QdrantVectorStore.point_id_for_chunk("source:chunk:2"))

    def test_upsert_preserves_payload_and_returns_stable_ids(self) -> None:
        backend = FakeBackend()
        store = QdrantVectorStore(Settings(embedding_dimension=3), backend=backend)
        vectors = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        payload = valid_payload()

        point_ids = store.upsert(vectors, [payload])

        self.assertEqual(point_ids, (QdrantVectorStore.point_id_for_chunk("chunk-1"),))
        self.assertEqual(backend.upsert_collection, store.collection_name)
        point = backend.points[0]
        self.assertEqual(point.point_id, point_ids[0])
        self.assertTrue(np.allclose(point.vector, [0.1, 0.2, 0.3]))
        self.assertEqual(point.payload, payload)
        self.assertEqual(set(point.payload), set(payload))

    def test_similarity_search_returns_score_and_payload(self) -> None:
        backend = FakeBackend(collection_exists=True)
        store = QdrantVectorStore(Settings(embedding_dimension=3), backend=backend)

        results = store.search(np.array([0.1, 0.2, 0.3]), limit=2)

        self.assertEqual(backend.search_query[0], store.collection_name)
        self.assertTrue(np.allclose(backend.search_query[1], [0.1, 0.2, 0.3]))
        self.assertEqual(backend.search_query[2], 2)
        self.assertEqual(results[0].score, 0.91)
        self.assertEqual(results[0].payload, valid_payload())

    def test_rejects_vector_dimension_mismatch(self) -> None:
        backend = FakeBackend()
        store = QdrantVectorStore(Settings(embedding_dimension=3), backend=backend)

        with self.assertRaisesRegex(ValueError, "does not match configured dimension 3"):
            store.upsert([[0.1, 0.2]], [valid_payload()])
        with self.assertRaisesRegex(ValueError, "does not match configured dimension 3"):
            store.search([0.1, 0.2])
        self.assertEqual(backend.points, [])
        self.assertIsNone(backend.search_query)

    def test_rejects_incomplete_payload(self) -> None:
        store = QdrantVectorStore(Settings(embedding_dimension=3), backend=FakeBackend())
        payload = valid_payload()
        del payload["text"]

        with self.assertRaisesRegex(ValueError, "payload is missing fields: text"):
            store.upsert([[0.1, 0.2, 0.3]], [payload])

    def test_local_client_uses_configured_path_and_cosine_collection(self) -> None:
        clients: list[object] = []

        class FakeQdrantClient:
            def __init__(self, *, path: str) -> None:
                self.path = path
                self.collection: tuple[str, object] | None = None
                clients.append(self)

            def collection_exists(self, *, collection_name: str) -> bool:
                return False

            def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
                self.collection = (collection_name, vectors_config)

            def close(self) -> None:
                pass

        class FakeVectorParams:
            def __init__(self, *, size: int, distance: object) -> None:
                self.size = size
                self.distance = distance

        client_module = ModuleType("qdrant_client")
        client_module.QdrantClient = FakeQdrantClient
        models_module = ModuleType("qdrant_client.models")
        models_module.Distance = SimpleNamespace(COSINE="cosine")
        models_module.VectorParams = FakeVectorParams

        with patch.dict(
            sys.modules,
            {"qdrant_client": client_module, "qdrant_client.models": models_module},
        ):
            QdrantVectorStore(
                Settings(
                    embedding_dimension=768,
                    qdrant_storage_path="tmp/qdrant-store",
                    qdrant_collection_name="local_test",
                )
            )

        self.assertEqual(clients[0].path, "tmp/qdrant-store")
        collection_name, vector_params = clients[0].collection
        self.assertEqual(collection_name, "local_test")
        self.assertEqual(vector_params.size, 768)
        self.assertEqual(vector_params.distance, "cosine")

    def test_import_does_not_import_qdrant_client(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] == 'qdrant_client':
        raise AssertionError('qdrant_client imported during module import')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import kisansathi.retrieval.vector_store
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source_root)

        subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()