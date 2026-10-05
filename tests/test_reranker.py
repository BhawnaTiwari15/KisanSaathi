import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from kisansathi.config import Settings
from kisansathi.retrieval.reranker import Reranker
from kisansathi.retrieval.vector_store import SearchResult


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], bool]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.calls.append((sentences, convert_to_numpy))
        return np.asarray(self.scores, dtype=np.float32)


def candidate(chunk_id: str, text: str, score: float = 0.0) -> SearchResult:
    return SearchResult(
        score=score,
        payload={
            "chunk_id": chunk_id,
            "source_id": f"source-{chunk_id}",
            "page_start": 3,
            "page_end": 4,
            "title": "Scheme document",
            "text": text,
            "extra": {"preserve": True},
        },
    )


class RerankerTests(unittest.TestCase):
    def test_model_loads_lazily_and_receives_query_document_pairs(self) -> None:
        model = FakeCrossEncoder([0.7, 0.9])
        loaded: list[str] = []

        def load_model(model_name: str) -> FakeCrossEncoder:
            loaded.append(model_name)
            return model

        reranker = Reranker(
            Settings(reranker_model_name="test/model"), model_loader=load_model
        )
        candidates = [candidate("first", "First document"), candidate("second", "Second document")]

        self.assertEqual(loaded, [])
        reranker.rerank("Eligibility?", candidates)

        self.assertEqual(loaded, ["test/model"])
        self.assertEqual(
            model.calls,
            [
                (
                    [
                        ("Eligibility?", "First document"),
                        ("Eligibility?", "Second document"),
                    ],
                    True,
                )
            ],
        )

    def test_sorts_by_relevance_score_and_truncates_top_k(self) -> None:
        candidates = [
            candidate("low", "Low relevance", score=1000.0),
            candidate("high", "High relevance", score=-1000.0),
            candidate("middle", "Middle relevance", score=0.0),
        ]
        model = FakeCrossEncoder([0.1, 0.9, 0.5])
        reranker = Reranker(model_loader=lambda _: model)

        reranked = reranker.rerank("query", candidates, top_k=2)

        self.assertEqual([result.payload["chunk_id"] for result in reranked], ["high", "middle"])
        self.assertAlmostEqual(reranked[0].score, 0.9)
        self.assertAlmostEqual(reranked[1].score, 0.5)

    def test_returns_all_when_fewer_candidates_than_top_k(self) -> None:
        model = FakeCrossEncoder([0.4, 0.8])
        reranker = Reranker(model_loader=lambda _: model)

        results = reranker.rerank(
            "query", [candidate("a", "A"), candidate("b", "B")], top_k=5
        )

        self.assertEqual([result.payload["chunk_id"] for result in results], ["b", "a"])

    def test_empty_candidates_do_not_load_model(self) -> None:
        loaded: list[str] = []
        reranker = Reranker(model_loader=lambda name: loaded.append(name))

        self.assertEqual(reranker.rerank("query", []), ())
        self.assertEqual(loaded, [])

    def test_blank_query_and_invalid_top_k_are_rejected(self) -> None:
        model = FakeCrossEncoder([])
        reranker = Reranker(model_loader=lambda _: model)

        for query in ("", "  \n"):
            with self.subTest(query=query), self.assertRaisesRegex(
                ValueError, "query must not be blank"
            ):
                reranker.rerank(query, [])
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(
                ValueError, "top_k must be a positive integer"
            ):
                reranker.rerank("query", [], top_k=top_k)

    def test_candidate_depth_and_candidate_content_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_depth must be a positive integer"):
            Reranker(candidate_depth=0)
        reranker = Reranker(model_loader=lambda _: FakeCrossEncoder([]))
        candidates = [candidate(str(index), "content") for index in range(11)]

        with self.assertRaisesRegex(ValueError, "candidate count must not exceed candidate_depth"):
            reranker.rerank("query", candidates)
        with self.assertRaisesRegex(ValueError, "candidate payload text must be a non-empty string"):
            reranker.rerank("query", [candidate("blank", "   ")])

    def test_payload_is_preserved_and_equal_scores_keep_input_order(self) -> None:
        first = candidate("first", "First", score=-100.0)
        second = candidate("second", "Second", score=100.0)
        model = FakeCrossEncoder([0.5, 0.5])
        reranker = Reranker(model_loader=lambda _: model)

        results = reranker.rerank("query", [first, second])

        self.assertEqual([result.payload["chunk_id"] for result in results], ["first", "second"])
        self.assertIs(results[0].payload, first.payload)
        self.assertEqual(results[0].payload, first.payload)
        self.assertEqual(results[0].score, 0.5)

    def test_import_does_not_import_sentence_transformers(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] == 'sentence_transformers':
        raise AssertionError('sentence_transformers imported during reranker module import')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import kisansathi.retrieval.reranker
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