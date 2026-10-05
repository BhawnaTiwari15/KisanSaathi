import unittest
from typing import Any

from kisansathi.domain.schemas import (
    AssistantResponse,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.orchestration.graph import OrchestrationState, build_graph
from kisansathi.retrieval.vector_store import SearchResult


class FakeRetriever:
    def __init__(self, results: tuple[SearchResult, ...] | None = None) -> None:
        self.results = results or ()
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        self.calls.append((query, top_k))
        return self.results


def make_result(chunk_id: str) -> SearchResult:
    payload: dict[str, Any] = {
        "chunk_id": chunk_id,
        "source_id": "test-source",
        "sha256": "0" * 64,
        "scheme": "TEST",
        "jurisdiction": "IN",
        "language": "en",
        "title": "Test Title",
        "page_start": 1,
        "page_end": 1,
        "heading": None,
        "text": "Test text",
    }
    return SearchResult(score=1.0, payload=payload)


class OrchestrationGraphTests(unittest.TestCase):
    def test_normal_message_takes_retrieval_route(self) -> None:
        retriever = FakeRetriever((make_result("c1"),))
        graph = build_graph(retriever).compile()
        message = UserMessage(text="What documents are required for PM-KISAN enrollment?", language=Language.ENGLISH)
        initial_state: OrchestrationState = {
            "message": message,
            "route": "",
            "retrieved_chunks": (),
            "response": AssistantResponse(
                text="placeholder",
                language=Language.ENGLISH,
                status=ResponseStatus.ANSWERED,
            ),
        }

        result = graph.invoke(initial_state)

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][0], message.text)
        self.assertEqual(retriever.calls[0][1], 5)
        self.assertEqual(len(result["retrieved_chunks"]), 1)
        self.assertIsInstance(result["response"], AssistantResponse)
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)

    def test_very_short_message_takes_clarification_route(self) -> None:
        retriever = FakeRetriever()
        graph = build_graph(retriever).compile()
        message = UserMessage(text="Help", language=Language.ENGLISH)
        initial_state: OrchestrationState = {
            "message": message,
            "route": "",
            "retrieved_chunks": (),
            "response": AssistantResponse(
                text="placeholder",
                language=Language.ENGLISH,
                status=ResponseStatus.ANSWERED,
            ),
        }

        result = graph.invoke(initial_state)

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(result["retrieved_chunks"], ())
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_underspecified_message_with_few_words_takes_clarification_route(self) -> None:
        retriever = FakeRetriever()
        graph = build_graph(retriever).compile()
        message = UserMessage(text="PM-KISAN", language=Language.ENGLISH)
        initial_state: OrchestrationState = {
            "message": message,
            "route": "",
            "retrieved_chunks": (),
            "response": AssistantResponse(
                text="placeholder",
                language=Language.ENGLISH,
                status=ResponseStatus.ANSWERED,
            ),
        }

        result = graph.invoke(initial_state)

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_retrieval_preserves_chunks_and_sets_answered_status(self) -> None:
        retriever = FakeRetriever((make_result("c1"), make_result("c2")))
        graph = build_graph(retriever).compile()
        message = UserMessage(text="When was PM-KISAN launched and how often are payments made?", language=Language.ENGLISH)
        initial_state: OrchestrationState = {
            "message": message,
            "route": "",
            "retrieved_chunks": (),
            "response": AssistantResponse(
                text="placeholder",
                language=Language.ENGLISH,
                status=ResponseStatus.ANSWERED,
            ),
        }

        result = graph.invoke(initial_state)

        self.assertEqual(len(result["retrieved_chunks"]), 2)
        self.assertEqual(result["retrieved_chunks"][0].payload["chunk_id"], "c1")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("retrieved relevant excerpts", result["response"].text.lower())
        self.assertIn("answer generation has not yet been implemented", result["response"].text.lower())

    def test_clarification_does_not_call_retriever_and_has_needs_clarification(self) -> None:
        retriever = FakeRetriever()
        graph = build_graph(retriever).compile()
        message = UserMessage(text="details?", language=Language.HINDI)
        initial_state: OrchestrationState = {
            "message": message,
            "route": "",
            "retrieved_chunks": (),
            "response": AssistantResponse(
                text="placeholder",
                language=Language.ENGLISH,
                status=ResponseStatus.ANSWERED,
            ),
        }

        result = graph.invoke(initial_state)

        self.assertEqual(retriever.calls, [])
        self.assertEqual(result["route"], "clarify")
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result["response"].language, Language.HINDI)

    def test_graph_reaches_final_response(self) -> None:
        retriever = FakeRetriever((make_result("c1"),))
        graph = build_graph(retriever).compile()
        message = UserMessage(text="Which categories are excluded from PM-KISAN benefits?", language=Language.ENGLISH)
        initial_state: OrchestrationState = {
            "message": message,
            "route": "",
            "retrieved_chunks": (),
            "response": AssistantResponse(
                text="placeholder",
                language=Language.ENGLISH,
                status=ResponseStatus.ANSWERED,
            ),
        }

        result = graph.invoke(initial_state)

        self.assertIsNotNone(result["response"])
        self.assertTrue(len(result["response"].text) > 0)
        self.assertEqual(result["response"].citations, ())
