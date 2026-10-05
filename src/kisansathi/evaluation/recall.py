"""Hand-labeled Hit Rate@5 and true Recall@5 evaluation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from kisansathi.retrieval.vector_store import SearchResult


EVALUATION_TOP_K = 5


class _Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]: ...


class _HybridRetriever(Protocol):
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]: ...


class _Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int | None = None,
    ) -> tuple[SearchResult, ...]: ...


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    query: str
    relevant_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("evaluation query must be a non-empty string")
        if not isinstance(self.relevant_chunk_ids, tuple) or not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must be a non-empty tuple")
        if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must contain non-empty strings")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain duplicates")


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    evaluation_set_version: str
    corpus_version: str
    examples: tuple[EvaluationExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_set_version, str) or not self.evaluation_set_version.strip():
            raise ValueError("evaluation_set_version must not be empty")
        if not isinstance(self.corpus_version, str) or not self.corpus_version.strip():
            raise ValueError("corpus_version must not be empty")
        if not isinstance(self.examples, tuple) or not self.examples:
            raise ValueError("evaluation dataset must contain at least one example")


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate the versioned, hand-labeled JSON evaluation set."""
    dataset_path = Path(path)
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read evaluation dataset {dataset_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation dataset JSON: {error.msg}") from error
    if not isinstance(raw, dict):
        raise ValueError("evaluation dataset must be a JSON object")
    examples_data = raw.get("examples")
    if not isinstance(examples_data, list):
        raise ValueError("evaluation dataset examples must be a list")
    examples: list[EvaluationExample] = []
    for index, item in enumerate(examples_data):
        if not isinstance(item, dict):
            raise ValueError(f"evaluation example {index} must be a JSON object")
        query = item.get("query")
        relevant_ids = item.get("relevant_chunk_ids")
        if not isinstance(relevant_ids, list):
            raise ValueError(f"evaluation example {index} relevant_chunk_ids must be a list")
        examples.append(EvaluationExample(query=query, relevant_chunk_ids=tuple(relevant_ids)))
    try:
        return EvaluationDataset(
            evaluation_set_version=raw["evaluation_set_version"],
            corpus_version=raw["corpus_version"],
            examples=tuple(examples),
        )
    except KeyError as error:
        raise ValueError(f"evaluation dataset is missing {error.args[0]!r}") from error


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    hit_at_5: int
    recall_at_5: float


@dataclass(frozen=True, slots=True)
class SystemEvaluation:
    system_name: str
    query_count: int
    hit_rate_at_5: float
    recall_at_5: float
    per_query: tuple[QueryEvaluation, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    evaluation_set_version: str
    corpus_version: str
    top_k: int
    systems: tuple[SystemEvaluation, ...]


class RerankedHybridRetriever:
    """Adapt hybrid candidate retrieval followed by reranking to retrieve()."""

    def __init__(
        self,
        hybrid_retriever: _HybridRetriever,
        reranker: _Reranker,
        *,
        candidate_depth: int = 10,
    ) -> None:
        if (
            isinstance(candidate_depth, bool)
            or not isinstance(candidate_depth, int)
            or candidate_depth <= 0
        ):
            raise ValueError("candidate_depth must be a positive integer")
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._candidate_depth = candidate_depth

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        output_limit = EVALUATION_TOP_K if top_k is None else top_k
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit <= 0:
            raise ValueError("top_k must be a positive integer")
        candidates = self._hybrid_retriever.retrieve(query, top_k=self._candidate_depth)
        return self._reranker.rerank(query, candidates, top_k=output_limit)


def evaluate_systems(
    systems: Mapping[str, _Retriever],
    dataset: EvaluationDataset,
) -> EvaluationSummary:
    """Evaluate retrievers in stable name order with binary Hit Rate@5 and Recall@5."""
    if not systems:
        raise ValueError("at least one retrieval system is required")
    system_results: list[SystemEvaluation] = []
    for system_name in sorted(systems):
        if not system_name.strip():
            raise ValueError("system names must not be empty")
        retriever = systems[system_name]
        query_results: list[QueryEvaluation] = []
        for example in dataset.examples:
            results = retriever.retrieve(example.query, top_k=EVALUATION_TOP_K)
            retrieved_ids = tuple(
                _chunk_id(result) for result in results[:EVALUATION_TOP_K]
            )
            relevant = set(example.relevant_chunk_ids)
            matched_count = len(relevant.intersection(retrieved_ids))
            query_results.append(
                QueryEvaluation(
                    query=example.query,
                    relevant_chunk_ids=example.relevant_chunk_ids,
                    retrieved_chunk_ids=retrieved_ids,
                    hit_at_5=int(matched_count > 0),
                    recall_at_5=matched_count / len(relevant),
                )
            )
        query_count = len(query_results)
        system_results.append(
            SystemEvaluation(
                system_name=system_name,
                query_count=query_count,
                hit_rate_at_5=sum(item.hit_at_5 for item in query_results) / query_count,
                recall_at_5=sum(item.recall_at_5 for item in query_results) / query_count,
                per_query=tuple(query_results),
            )
        )
    return EvaluationSummary(
        evaluation_set_version=dataset.evaluation_set_version,
        corpus_version=dataset.corpus_version,
        top_k=EVALUATION_TOP_K,
        systems=tuple(system_results),
    )


def _chunk_id(result: SearchResult) -> str:
    chunk_id = result.payload.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("retriever result payload must contain a non-empty chunk_id")
    return chunk_id