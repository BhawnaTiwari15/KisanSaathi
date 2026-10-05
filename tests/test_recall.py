import json
from pathlib import Path
import tempfile
import unittest

from kisansathi.evaluation.recall import (
    EvaluationDataset,
    EvaluationExample,
    RerankedHybridRetriever,
    evaluate_systems,
    load_evaluation_dataset,
)
from kisansathi.retrieval.vector_store import SearchResult


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, tuple[SearchResult, ...]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, ())[:top_k]


class FakeHybridRetriever:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeReranker:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, tuple[SearchResult, ...], int | None]] = []

    def rerank(
        self,
        query: str,
        candidates: tuple[SearchResult, ...],
        *,
        top_k: int | None = None,
    ) -> tuple[SearchResult, ...]:
        self.calls.append((query, candidates, top_k))
        return self.results[:top_k]


def result(chunk_id: str) -> SearchResult:
    return SearchResult(score=0.5, payload={"chunk_id": chunk_id, "text": chunk_id})


def dataset(*examples: EvaluationExample) -> EvaluationDataset:
    return EvaluationDataset("test-v1", "corpus-v1", tuple(examples))


class RetrievalEvaluationTests(unittest.TestCase):
    def test_hit_when_a_relevant_chunk_is_in_top_five(self) -> None:
        examples = dataset(EvaluationExample("query", ("relevant",)))
        retriever = FakeRetriever({"query": (result("other"), result("relevant"))})

        report = evaluate_systems({"fake": retriever}, examples).systems[0]

        self.assertEqual(report.per_query[0].hit_at_5, 1)
        self.assertEqual(report.per_query[0].recall_at_5, 1.0)
        self.assertEqual(report.hit_rate_at_5, 1.0)

    def test_miss_when_no_relevant_chunk_is_in_top_five(self) -> None:
        examples = dataset(EvaluationExample("query", ("relevant",)))
        retriever = FakeRetriever({"query": (result("other"),)})

        report = evaluate_systems({"fake": retriever}, examples).systems[0]

        self.assertEqual(report.per_query[0].hit_at_5, 0)
        self.assertEqual(report.per_query[0].recall_at_5, 0.0)
        self.assertEqual(report.hit_rate_at_5, 0.0)

    def test_multiple_relevant_chunks_compute_true_recall_fraction(self) -> None:
        examples = dataset(EvaluationExample("query", ("relevant-a", "relevant-b")))
        retriever = FakeRetriever(
            {"query": (result("relevant-a"), result("irrelevant"))}
        )

        report = evaluate_systems({"fake": retriever}, examples).systems[0]

        self.assertEqual(report.per_query[0].hit_at_5, 1)
        self.assertEqual(report.per_query[0].recall_at_5, 0.5)
        self.assertEqual(report.recall_at_5, 0.5)

    def test_hit_rate_and_true_recall_aggregate_per_query(self) -> None:
        examples = dataset(
            EvaluationExample("hit", ("a", "b")),
            EvaluationExample("miss", ("c",)),
        )
        retriever = FakeRetriever({"hit": (result("a"),), "miss": (result("x"),)})

        report = evaluate_systems({"fake": retriever}, examples).systems[0]

        self.assertEqual(report.query_count, 2)
        self.assertEqual(report.hit_rate_at_5, 0.5)
        self.assertEqual(report.recall_at_5, 0.25)

    def test_empty_and_invalid_examples_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation query"):
            EvaluationExample("  ", ("chunk",))
        with self.assertRaisesRegex(ValueError, "relevant_chunk_ids"):
            EvaluationExample("query", ())
        with self.assertRaisesRegex(ValueError, "at least one example"):
            EvaluationDataset("v1", "c1", ())
        with self.assertRaisesRegex(ValueError, "at least one retrieval system"):
            evaluate_systems({}, dataset(EvaluationExample("query", ("chunk",))))

    def test_loads_versioned_ten_query_dataset(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/evaluation/retrieval_en_v1.json"

        loaded = load_evaluation_dataset(path)

        self.assertEqual(loaded.evaluation_set_version, "retrieval-en-v1")
        self.assertIn("95d4e289", loaded.corpus_version)
        self.assertEqual(len(loaded.examples), 10)

    def test_aggregation_order_is_deterministic(self) -> None:
        examples = dataset(
            EvaluationExample("first", ("a",)),
            EvaluationExample("second", ("b",)),
        )
        systems = {
            "z-system": FakeRetriever({"first": (result("a"),), "second": ()}),
            "a-system": FakeRetriever({"first": (), "second": (result("b"),)}),
        }

        first = evaluate_systems(systems, examples)
        second = evaluate_systems(systems, examples)

        self.assertEqual([system.system_name for system in first.systems], ["a-system", "z-system"])
        self.assertEqual(first, second)
        self.assertEqual([query.query for query in first.systems[0].per_query], ["first", "second"])

    def test_reranked_hybrid_adapter_fetches_ten_and_returns_reranked_results(self) -> None:
        candidates = tuple(result(f"candidate-{index}") for index in range(10))
        hybrid = FakeHybridRetriever(candidates)
        reranked = (candidates[4], candidates[2])
        reranker = FakeReranker(reranked)
        adapter = RerankedHybridRetriever(hybrid, reranker)

        output = adapter.retrieve("query", top_k=5)

        self.assertEqual(hybrid.calls, [("query", 10)])
        self.assertEqual(reranker.calls, [("query", candidates, 5)])
        self.assertEqual(output, reranked)


if __name__ == "__main__":
    unittest.main()