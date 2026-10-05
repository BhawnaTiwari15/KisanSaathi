"""Reciprocal rank fusion over dense and BM25 retrieval results."""

from typing import Protocol

from kisansathi.retrieval.vector_store import SearchResult


class _Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]: ...


class HybridRetriever:
    """Fuse dense and BM25 result rankings without comparing their raw scores."""

    _RRF_K = 60

    def __init__(
        self,
        dense_retriever: _Retriever,
        bm25_retriever: _Retriever,
        *,
        default_top_k: int = 5,
        candidate_multiplier: int = 3,
    ) -> None:
        self._validate_top_k(default_top_k)
        self._validate_candidate_multiplier(candidate_multiplier)
        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever
        self._default_top_k = default_top_k
        self._candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        """Return top results fused from both retrievers using equal-weight RRF."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be blank")

        result_limit = self._default_top_k if top_k is None else top_k
        self._validate_top_k(result_limit)
        candidate_limit = self._candidate_multiplier * result_limit
        dense_results = self._dense_retriever.retrieve(query, top_k=candidate_limit)
        bm25_results = self._bm25_retriever.retrieve(query, top_k=candidate_limit)

        fused: dict[str, tuple[float, SearchResult, int]] = {}
        first_seen_order = 0
        for ranked_results in (dense_results, bm25_results):
            for rank, result in enumerate(ranked_results, start=1):
                chunk_id = result.payload["chunk_id"]
                contribution = 1.0 / (self._RRF_K + rank)
                current = fused.get(chunk_id)
                if current is None:
                    fused[chunk_id] = (contribution, result, first_seen_order)
                    first_seen_order += 1
                else:
                    fused[chunk_id] = (current[0] + contribution, current[1], current[2])

        ordered = sorted(fused.values(), key=lambda candidate: (-candidate[0], candidate[2]))
        return tuple(
            SearchResult(score=score, payload=result.payload)
            for score, result, _ in ordered[:result_limit]
        )

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    @staticmethod
    def _validate_candidate_multiplier(candidate_multiplier: int) -> None:
        if (
            isinstance(candidate_multiplier, bool)
            or not isinstance(candidate_multiplier, int)
            or candidate_multiplier <= 0
        ):
            raise ValueError("candidate_multiplier must be a positive integer")