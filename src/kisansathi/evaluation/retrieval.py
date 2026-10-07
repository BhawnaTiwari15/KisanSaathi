"""Retrieval evaluation metrics: Hit@K and Recall@K."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Any
import json

from kisansathi.retrieval.vector_store import SearchResult

from kisansathi.evaluation.schemas import RetrievalDataset, RetrievalExample


class _Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]: ...


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Metrics for a single query."""

    query: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    k: int
    hit_at_k: int
    recall_at_k: float

    @property
    def matched_chunk_ids(self) -> tuple[str, ...]:
        relevant = set(self.relevant_chunk_ids)
        retrieved = set(self.retrieved_chunk_ids)
        return tuple(sorted(relevant & retrieved))

    @property
    def missed_chunk_ids(self) -> tuple[str, ...]:
        relevant = set(self.relevant_chunk_ids)
        retrieved = set(self.retrieved_chunk_ids)
        return tuple(sorted(relevant - retrieved))

    @property
    def extra_chunk_ids(self) -> tuple[str, ...]:
        relevant = set(self.relevant_chunk_ids)
        retrieved = set(self.retrieved_chunk_ids)
        return tuple(sorted(retrieved - relevant))


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    """Aggregate metrics for a retrieval system."""

    system_name: str
    query_count: int
    k: int
    hit_rate_at_k: float
    recall_at_k: float
    per_query: tuple[QueryMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "query_count": self.query_count,
            "k": self.k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "recall_at_k": self.recall_at_k,
            "per_query": [
                {
                    "query": q.query,
                    "relevant_chunk_ids": q.relevant_chunk_ids,
                    "retrieved_chunk_ids": q.retrieved_chunk_ids,
                    "k": q.k,
                    "hit_at_k": q.hit_at_k,
                    "recall_at_k": q.recall_at_k,
                    "matched_chunk_ids": q.matched_chunk_ids,
                    "missed_chunk_ids": q.missed_chunk_ids,
                    "extra_chunk_ids": q.extra_chunk_ids,
                }
                for q in self.per_query
            ],
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    """Complete retrieval evaluation result."""

    evaluation_set_version: str
    corpus_version: str
    top_k: int
    systems: tuple[SystemMetrics, ...]
    timestamp_utc: str
    system_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_set_version": self.evaluation_set_version,
            "corpus_version": self.corpus_version,
            "system_version": self.system_version,
            "top_k": self.top_k,
            "timestamp_utc": self.timestamp_utc,
            "systems": [s.to_dict() for s in self.systems],
        }

    def to_jsonl(self) -> list[str]:
        """Convert to JSONL format for machine-readable output."""
        lines = []
        for sys in self.systems:
            lines.append(json.dumps({
                "benchmark_version": self.evaluation_set_version,
                "corpus_version": self.corpus_version,
                "system_version": self.system_version,
                "metric": "hit_rate_at_k",
                "language": self._infer_language(),
                "system": sys.system_name,
                "value": sys.hit_rate_at_k,
                "k": self.top_k,
                "timestamp_utc": self.timestamp_utc,
                "query_count": sys.query_count,
            }, ensure_ascii=False))
            lines.append(json.dumps({
                "benchmark_version": self.evaluation_set_version,
                "corpus_version": self.corpus_version,
                "system_version": self.system_version,
                "metric": "recall_at_k",
                "language": self._infer_language(),
                "system": sys.system_name,
                "value": sys.recall_at_k,
                "k": self.top_k,
                "timestamp_utc": self.timestamp_utc,
                "query_count": sys.query_count,
            }, ensure_ascii=False))
        return lines

    def _infer_language(self) -> str:
        # This would be set from the dataset; for now return unknown
        return "unknown"


def evaluate_retriever(
    retriever: _Retriever,
    dataset: RetrievalDataset,
    k: int = 5,
) -> tuple[list[QueryMetrics], float, float]:
    """Evaluate a single retriever against a dataset at top-k.

    Returns:
        - per-query metrics
        - hit_rate_at_k (binary hit)
        - recall_at_k (fraction of relevant found)
    """
    if k <= 0:
        raise ValueError("k must be positive")

    query_metrics: list[QueryMetrics] = []

    for example in dataset.examples:
        results = retriever.retrieve(example.query, top_k=k)
        retrieved_ids = tuple(
            _chunk_id(result) for result in results[:k]
        )

        relevant = set(example.expected_chunk_ids)
        matched_count = len(relevant.intersection(retrieved_ids))

        query_metrics.append(QueryMetrics(
            query=example.query,
            relevant_chunk_ids=example.expected_chunk_ids,
            retrieved_chunk_ids=retrieved_ids,
            k=k,
            hit_at_k=1 if matched_count > 0 else 0,
            recall_at_k=matched_count / len(relevant) if relevant else 0.0,
        ))

    hit_rate = sum(q.hit_at_k for q in query_metrics) / len(query_metrics)
    recall = sum(q.recall_at_k for q in query_metrics) / len(query_metrics)

    return query_metrics, hit_rate, recall


def evaluate_systems(
    systems: Mapping[str, _Retriever],
    dataset: RetrievalDataset,
    k: int = 5,
    system_version: str = "",
) -> RetrievalEvaluationResult:
    """Evaluate multiple retrieval systems against a dataset.

    Systems are evaluated in stable sorted order by name.
    """
    if not systems:
        raise ValueError("at least one retrieval system is required")

    system_results: list[SystemMetrics] = []

    for system_name in sorted(systems):
        if not system_name.strip():
            raise ValueError("system names must not be empty")
        retriever = systems[system_name]

        per_query, hit_rate, recall = evaluate_retriever(retriever, dataset, k)

        system_results.append(SystemMetrics(
            system_name=system_name,
            query_count=len(per_query),
            k=k,
            hit_rate_at_k=hit_rate,
            recall_at_k=recall,
            per_query=tuple(per_query),
        ))

    from datetime import datetime
    timestamp = datetime.utcnow().isoformat() + "Z"

    return RetrievalEvaluationResult(
        evaluation_set_version=dataset.evaluation_set_version,
        corpus_version=dataset.corpus_version,
        top_k=k,
        systems=tuple(system_results),
        timestamp_utc=timestamp,
        system_version=system_version,
    )


def _chunk_id(result: SearchResult) -> str:
    chunk_id = result.payload.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("retriever result payload must contain a non-empty chunk_id")
    return chunk_id