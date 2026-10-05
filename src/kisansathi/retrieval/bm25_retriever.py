"""BM25 lexical retrieval over the local FTS5 store."""

from typing import Protocol

from kisansathi.retrieval.vector_store import SearchResult


class _BM25Store(Protocol):
    def search(self, query: str, *, limit: int = 5) -> tuple[SearchResult, ...]: ...


class BM25Retriever:
    """Validate a natural-language query and return lexical matches."""

    def __init__(self, store: _BM25Store, *, default_top_k: int = 5) -> None:
        self._validate_top_k(default_top_k)
        self._store = store
        self._default_top_k = default_top_k

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be blank")
        result_limit = self._default_top_k if top_k is None else top_k
        self._validate_top_k(result_limit)
        return self._store.search(query, limit=result_limit)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")