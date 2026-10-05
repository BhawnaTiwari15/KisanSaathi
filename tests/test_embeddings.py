import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from kisansathi.config import Settings
from kisansathi.retrieval.embeddings import EmbeddingService


class FakeEmbeddingModel:
    def __init__(self, dimension: int = 2, invalid_query_shape: bool = False) -> None:
        self.dimension = dimension
        self.invalid_query_shape = invalid_query_shape
        self.query_calls: list[tuple[object, bool, bool]] = []
        self.document_calls: list[tuple[object, bool, bool]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode_query(
        self,
        sentences: str,
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.query_calls.append((sentences, normalize_embeddings, convert_to_numpy))
        if self.invalid_query_shape:
            return np.array([1.0, 2.0, 3.0])
        return np.array([3.0, 4.0])

    def encode_document(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.document_calls.append((sentences, normalize_embeddings, convert_to_numpy))
        return np.array([[3.0, 4.0] for _ in sentences])


class EmbeddingServiceTests(unittest.TestCase):
    def test_query_encoding_is_lazy_configured_and_normalized(self) -> None:
        fake_model = FakeEmbeddingModel()
        loader_calls: list[str] = []

        def load_model(model_name: str) -> FakeEmbeddingModel:
            loader_calls.append(model_name)
            return fake_model

        service = EmbeddingService(
            Settings(embedding_model_name="test/query-model"), model_loader=load_model
        )

        self.assertEqual(loader_calls, [])
        embedding = service.embed_query("How can I check eligibility?")

        self.assertEqual(loader_calls, ["test/query-model"])
        self.assertEqual(
            fake_model.query_calls,
            [("How can I check eligibility?", True, True)],
        )
        self.assertEqual(embedding.shape, (2,))
        self.assertTrue(np.allclose(np.linalg.norm(embedding), 1.0))
        self.assertEqual(service.embedding_dimension, 2)

    def test_document_encoding_passes_batch_and_normalization_option(self) -> None:
        fake_model = FakeEmbeddingModel()
        service = EmbeddingService(
            Settings(embedding_model_name="test/document-model"),
            normalize_embeddings=False,
            model_loader=lambda _: fake_model,
        )

        embeddings = service.embed_documents(("First document", "Second document"))

        self.assertEqual(
            fake_model.document_calls,
            [(["First document", "Second document"], False, True)],
        )
        self.assertEqual(embeddings.shape, (2, 2))
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), [5.0, 5.0]))

    def test_rejects_output_with_wrong_embedding_dimension(self) -> None:
        service = EmbeddingService(
            model_loader=lambda _: FakeEmbeddingModel(dimension=2, invalid_query_shape=True)
        )

        with self.assertRaisesRegex(ValueError, r"expected \(1, 2\)"):
            service.embed_query("test query")

    def test_import_does_not_import_or_load_sentence_transformers(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] == 'sentence_transformers':
        raise AssertionError('sentence_transformers imported during module import')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import kisansathi.retrieval.embeddings
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