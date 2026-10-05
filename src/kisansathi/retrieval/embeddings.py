"""Dense text embeddings backed by Sentence Transformers."""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from kisansathi.config import Settings


class _EmbeddingModel(Protocol):
    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode_query(
        self,
        sentences: str,
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> object: ...

    def encode_document(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> object: ...


ModelLoader = Callable[[str], _EmbeddingModel]


def _load_sentence_transformer(model_name: str) -> _EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingService:
    """Encode queries and documents with lazily loaded, normalized embeddings."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        normalize_embeddings: bool = True,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if not isinstance(normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be a bool")
        self._settings = settings if settings is not None else Settings.from_env()
        self._normalize_embeddings = normalize_embeddings
        self._model_loader = model_loader
        self._model: _EmbeddingModel | None = None
        self._embedding_dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model_name

    @property
    def embedding_dimension(self) -> int:
        self._get_model()
        assert self._embedding_dimension is not None
        return self._embedding_dimension

    def embed_query(self, query: str) -> np.ndarray:
        """Return one query embedding as a one-dimensional NumPy array."""
        self._validate_text(query, "query")
        return self._encode(query, expected_count=1, method="encode_query")[0]

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        """Return document embeddings as a two-dimensional NumPy array."""
        if isinstance(documents, str):
            raise TypeError("documents must be a sequence of strings, not a string")
        document_batch = list(documents)
        if not document_batch:
            raise ValueError("documents must contain at least one document")
        for document in document_batch:
            self._validate_text(document, "document")
        return self._encode(
            document_batch, expected_count=len(document_batch), method="encode_document"
        )

    def _get_model(self) -> _EmbeddingModel:
        if self._model is None:
            loader = self._model_loader or _load_sentence_transformer
            model = loader(self.model_name)
            dimension = model.get_sentence_embedding_dimension()
            if dimension is None or dimension <= 0:
                raise ValueError("Embedding model must report a positive embedding dimension")
            self._embedding_dimension = int(dimension)
            self._model = model
        return self._model

    def _encode(
        self,
        texts: str | Sequence[str],
        *,
        expected_count: int,
        method: str,
    ) -> np.ndarray:
        model = self._get_model()
        encode = getattr(model, method)
        output = encode(
            texts,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
        )
        embeddings = np.array(output, dtype=np.float32, copy=True)
        if embeddings.ndim == 1 and expected_count == 1:
            embeddings = embeddings.reshape(1, -1)

        expected_shape = (expected_count, self.embedding_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Embedding model returned shape {embeddings.shape}; expected {expected_shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding model returned non-finite values")
        if self._normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.divide(embeddings, norms, out=embeddings, where=norms > 0)
        return embeddings

    @staticmethod
    def _validate_text(text: str, field_name: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"{field_name} must be a string")
        if not text.strip():
            raise ValueError(f"{field_name} must not be empty")