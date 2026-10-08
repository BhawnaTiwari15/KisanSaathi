"""Dense retrieval over the configured vector store."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from kisansathi.retrieval.vector_store import SearchResult


class _EmbeddingService(Protocol):
    def embed_query(self, query: str) -> np.ndarray: ...


class _VectorStore(Protocol):
    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> tuple[SearchResult, ...]: ...


class DenseRetriever:
    """Encode a natural-language query and return its closest stored chunks."""

    def __init__(
        self,
        embedding_service: _EmbeddingService,
        vector_store: _VectorStore,
        *,
        default_top_k: int = 5,
    ) -> None:
        self._validate_top_k(default_top_k)
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._closed = False

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        """Return up to ``top_k`` dense matches for a non-blank query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be blank")

        result_limit = self._default_top_k if top_k is None else top_k
        self._validate_top_k(result_limit)
        query_vector = self._embedding_service.embed_query(query)
        return self._vector_store.search(query_vector, limit=result_limit)

    def close(self) -> None:
        """Close the underlying vector store if it supports closing.

        Safe to call multiple times.
        """
        if self._closed:
            return
        if hasattr(self._vector_store, "close"):
            self._vector_store.close()
        self._closed = True

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")