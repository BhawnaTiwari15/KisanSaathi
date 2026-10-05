import unittest

from kisansathi.retrieval.hybrid_retriever import HybridRetriever
from kisansathi.retrieval.vector_store import SearchResult


class FakeRetriever:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        self.calls.append((query, top_k))
        return self.results


def result(chunk_id: str, raw_score: float, **payload_fields: object) -> SearchResult:
    payload = {"chunk_id": chunk_id, "text": f"Text for {chunk_id}"}
    payload.update(payload_fields)
    return SearchResult(score=raw_score, payload=payload)


class HybridRetrieverTests(unittest.TestCase):
    def test_overlapping_results_receive_both_rank_contributions(self) -> None:
        dense_overlap = result("overlap", 1000.0, source_id="dense-source")
        bm25_overlap = result("overlap", -1000.0, source_id="bm25-source")
        dense = FakeRetriever((result("dense-only", 500.0), dense_overlap))
        bm25 = FakeRetriever((bm25_overlap, result("bm25-only", -2000.0)))
        retriever = HybridRetriever(dense, bm25)

        results = retriever.retrieve("query", top_k=3)

        by_chunk = {item.payload["chunk_id"]: item for item in results}
        self.assertAlmostEqual(by_chunk["overlap"].score, 1 / 62 + 1 / 61)
        self.assertIs(by_chunk["overlap"].payload, dense_overlap.payload)
        self.assertEqual(by_chunk["overlap"].payload["source_id"], "dense-source")

    def test_single_retriever_results_get_only_their_own_contribution(self) -> None:
        dense = FakeRetriever((result("dense-only", 0.99),))
        bm25 = FakeRetriever((result("bm25-only", 999.0),))

        results = HybridRetriever(dense, bm25).retrieve("query", top_k=2)

        by_chunk = {item.payload["chunk_id"]: item.score for item in results}
        self.assertAlmostEqual(by_chunk["dense-only"], 1 / 61)
        self.assertAlmostEqual(by_chunk["bm25-only"], 1 / 61)

    def test_uses_one_based_ranks_and_truncates_after_fusion(self) -> None:
        dense = FakeRetriever(
            (result("d1", 0.1), result("d2", 0.2), result("d3", 0.3), result("d4", 0.4))
        )
        bm25 = FakeRetriever(
            (result("b1", 0.9), result("b2", 0.8), result("b3", 0.7), result("b4", 0.6))
        )

        results = HybridRetriever(dense, bm25, candidate_multiplier=2).retrieve("query", top_k=2)

        self.assertEqual([item.payload["chunk_id"] for item in results], ["d1", "b1"])
        self.assertAlmostEqual(results[0].score, 1 / 61)
        self.assertAlmostEqual(results[1].score, 1 / 61)
        self.assertEqual(dense.calls, [("query", 4)])
        self.assertEqual(bm25.calls, [("query", 4)])

    def test_default_candidate_multiplier_is_three(self) -> None:
        dense = FakeRetriever(())
        bm25 = FakeRetriever(())

        HybridRetriever(dense, bm25).retrieve("query", top_k=4)

        self.assertEqual(dense.calls, [("query", 12)])
        self.assertEqual(bm25.calls, [("query", 12)])

    def test_deterministic_first_seen_order_for_equal_scores(self) -> None:
        dense = FakeRetriever((result("dense-first", -999.0),))
        bm25 = FakeRetriever((result("bm25-second", 999.0),))

        results = HybridRetriever(dense, bm25).retrieve("query", top_k=2)

        self.assertEqual(
            [item.payload["chunk_id"] for item in results],
            ["dense-first", "bm25-second"],
        )

    def test_raw_retriever_scores_do_not_affect_fused_scores(self) -> None:
        dense_a = FakeRetriever((result("a", 1e30), result("b", -1e30)))
        bm25_a = FakeRetriever((result("b", 1e100), result("a", -1e100)))
        dense_b = FakeRetriever((result("a", -1e100), result("b", 1e100)))
        bm25_b = FakeRetriever((result("b", -1e30), result("a", 1e30)))

        first = HybridRetriever(dense_a, bm25_a).retrieve("query", top_k=2)
        second = HybridRetriever(dense_b, bm25_b).retrieve("query", top_k=2)

        self.assertEqual(
            [(item.payload["chunk_id"], item.score) for item in first],
            [(item.payload["chunk_id"], item.score) for item in second],
        )

    def test_preserves_payload_fields_without_modification(self) -> None:
        original_payload = {
            "chunk_id": "chunk-1",
            "source_id": "source-1",
            "page_start": 2,
            "page_end": 3,
            "text": "Complete text",
            "additional": {"preserve": True},
        }
        dense_result = SearchResult(score=0.42, payload=original_payload)
        dense = FakeRetriever((dense_result,))
        bm25 = FakeRetriever(())

        fused = HybridRetriever(dense, bm25).retrieve("query", top_k=1)[0]

        self.assertIs(fused.payload, original_payload)
        self.assertEqual(fused.payload, original_payload)

    def test_rejects_blank_query_and_invalid_top_k(self) -> None:
        dense = FakeRetriever(())
        bm25 = FakeRetriever(())
        retriever = HybridRetriever(dense, bm25)

        for query in ("", "  \n"):
            with self.subTest(query=query), self.assertRaisesRegex(
                ValueError, "query must not be blank"
            ):
                retriever.retrieve(query)
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(
                ValueError, "top_k must be a positive integer"
            ):
                retriever.retrieve("query", top_k=top_k)
        with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
            HybridRetriever(dense, bm25, default_top_k=0)

    def test_rejects_invalid_candidate_multiplier(self) -> None:
        for multiplier in (0, -1, True, 1.5):
            with self.subTest(multiplier=multiplier), self.assertRaisesRegex(
                ValueError, "candidate_multiplier must be a positive integer"
            ):
                HybridRetriever(FakeRetriever(()), FakeRetriever(()), candidate_multiplier=multiplier)


if __name__ == "__main__":
    unittest.main()