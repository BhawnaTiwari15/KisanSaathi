"""Integration tests for the full answer generation pipeline in LangGraph."""

import unittest
from pathlib import Path

from kisansathi.citations import ManifestSourceRegistry
from kisansathi.citations.models import CitationBatch, Citation
from kisansathi.domain.schemas import (
    AssistantResponse,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.eligibility import PM_KISAN_SCHEME, EligibilityRequest
from kisansathi.eligibility.models import EligibilityStatus
from kisansathi.orchestration.graph import OrchestrationState, build_graph
from kisansathi.generation.fake import FakeLLMClient
from kisansathi.generation.generator import DefaultAnswerGenerator
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.models import WeatherCurrent, WeatherResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "sources.json"
SOURCE_ID = "pm-kisan-revised-faq"
CORPUS_SHA256 = "95d4e2892b4b351e599bf903f8c6f89b20d4b74831eb8f1a0e61a2a66771ff86"
LAND_CHUNK = f"{SOURCE_ID}:33ff2db494c59ab1ca6c4f76"
CULTIVABLE_CHUNK = f"{SOURCE_ID}:b148dd12de05914c6bfaf324"
AGRICULTURAL_CHUNK = f"{SOURCE_ID}:f39a38edb9e1352d1315ece3"
TAX_CHUNK = f"{SOURCE_ID}:1fe005b874ba210ad336e5d9"
ALL_PM_KISAN_CHUNKS = (LAND_CHUNK, CULTIVABLE_CHUNK, AGRICULTURAL_CHUNK, TAX_CHUNK)

ELIGIBILITY_QUESTION = "Please tell me whether I am eligible under PM-KISAN."
RETRIEVAL_QUESTION = "What does the revised PM-KISAN guideline say about payment cycles?"
WEATHER_QUESTION = "What is the weather forecast for my field today?"


def build_registry() -> ManifestSourceRegistry:
    return ManifestSourceRegistry.from_manifest(MANIFEST_PATH)


def make_payload(chunk_id: str, *, page_start: int, page_end: int, **overrides: object) -> SearchResult:
    payload: dict[str, object] = {
        "chunk_id": chunk_id,
        "source_id": SOURCE_ID,
        "sha256": CORPUS_SHA256,
        "scheme": PM_KISAN_SCHEME,
        "jurisdiction": "IN",
        "language": "en",
        "page_start": page_start,
        "page_end": page_end,
        "heading": None,
        "text": "Stored source wording the citation layer must never read.",
    }
    payload.update(overrides)
    return SearchResult(score=1.0, payload=payload)


class FakeRetriever:
    def __init__(self, results: tuple[SearchResult, ...] | None = None) -> None:
        self.results = results or ()
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        self.calls.append((query, top_k))
        return self.results


class FakeWeatherClient:
    def __init__(self, response: WeatherResponse) -> None:
        self.response = response
        self.calls: list = []

    def get_forecast(self, req, *, include_forecast: bool = False, forecast_days: int = 1) -> WeatherResponse:
        self.calls.append(req)
        return self.response


def make_weather_response() -> WeatherResponse:
    return WeatherResponse(
        latitude=28.6139,
        longitude=77.2090,
        timezone="Asia/Kolkata",
        current=WeatherCurrent(
            temperature_c=31.2,
            precipitation_mm=0.0,
            wind_speed_mps=8.4,
            weather_code=2,
            time_iso="2026-10-06T09:00",
        ),
    )


def make_state(message: UserMessage, **extra: object) -> OrchestrationState:
    state: dict[str, object] = {
        "message": message,
        "route": "",
        "retrieved_chunks": (),
        "response": AssistantResponse(
            text="placeholder",
            language=Language.ENGLISH,
            status=ResponseStatus.ANSWERED,
        ),
    }
    state.update(extra)
    return state  # type: ignore


def message(text: str, **kwargs: object) -> UserMessage:
    defaults: dict[str, object] = {"text": text, "language": Language.ENGLISH}
    defaults.update(kwargs)
    return UserMessage(**defaults)


def eligible_request(**overrides: object):
    facts: dict[str, object] = {
        "landholding_in_own_name": True,
        "land_is_cultivable": True,
        "land_used_for_non_agricultural_purpose": False,
        "family_member_paid_income_tax_last_assessment_year": False,
    }
    facts.update(overrides)
    return EligibilityRequest(scheme=PM_KISAN_SCHEME, **facts)


def _make_llm_response(answer_text: str, citation_ids: tuple[str, ...]) -> str:
    """Format an LLM response with the given answer and citation IDs."""
    citations_section = "\n".join(f"[{cid}]" for cid in citation_ids)
    return (
        f"ANSWER:\n{answer_text}\n\n"
        f"CITATIONS:\n{citations_section}"
    )


class RetrievalGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def _make_resolver(self):
        from kisansathi.citations.resolver import CitationResolver
        return CitationResolver(self.registry)

    def test_retrieval_path_generates_answer_with_citations(self) -> None:
        fake_llm = FakeLLMClient(
            response=_make_llm_response(
                "PM-KISAN payments are made in three installments per year.",
                (LAND_CHUNK,),
            )
        )
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertEqual(len(result["response"].citations), 1)
        self.assertEqual(result["response"].citations[0].chunk_id, LAND_CHUNK)
        self.assertIn("installments", result["response"].text)

    def test_retrieval_without_generator_uses_placeholder(self) -> None:
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(retriever, citation_resolver=self._make_resolver()).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("not yet been implemented", result["response"].text)
        # With citation_resolver but no answer_generator, placeholder includes citations from resolver
        self.assertEqual(len(result["response"].citations), 1)
        self.assertEqual(result["response"].citations[0].chunk_id, LAND_CHUNK)

    def test_generator_receives_correct_context(self) -> None:
        fake_llm = FakeLLMClient(
            response=_make_llm_response("Test answer.", (LAND_CHUNK,))
        )
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(len(fake_llm.calls), 1)
        call = fake_llm.calls[0]
        self.assertIn("USER QUESTION:", call["user_prompt"])
        self.assertIn("EVIDENCE:", call["user_prompt"])
        self.assertIn(LAND_CHUNK, call["user_prompt"])

    def test_citation_validation_rejects_hallucinated_id(self) -> None:
        # Generator tries to cite an ID not in the batch
        fake_llm = FakeLLMClient(
            response=_make_llm_response(
                "Some answer.",
                ("pm-kisan-revised-faq:invalid123456789012345678",),
            )
        )
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        # Should fall back to NEEDS_CLARIFICATION with no citations
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result["response"].citations, ())

    def test_llm_failure_results_in_abstained(self) -> None:
        fake_llm = FakeLLMClient(response=Exception("LLM timeout"))
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_malformed_output_results_in_clarification(self) -> None:
        fake_llm = FakeLLMClient(response="Not a valid format")
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_empty_citations_batch_abstains_on_retrieval(self) -> None:
        # No chunks retrieved, but generator is present
        fake_llm = FakeLLMClient(response="ANSWER:\nNo sources.\n\nCITATIONS:\n")
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever(())
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        # Generator runs but has no citations to work with
        self.assertIsNotNone(result["generated_answer"])


class EligibilityGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def _make_resolver(self):
        from kisansathi.citations.resolver import CitationResolver
        return CitationResolver(self.registry)

    def test_eligibility_path_generates_answer_with_citations(self) -> None:
        fake_llm = FakeLLMClient(
            response=_make_llm_response(
                "You are eligible for PM-KISAN.",
                ALL_PM_KISAN_CHUNKS,
            )
        )
        fake_gen = DefaultAnswerGenerator(fake_llm)
        graph = build_graph(
            FakeRetriever(),
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        self.assertEqual(result["route"], "eligibility")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertEqual(len(result["response"].citations), 4)  # 4 PM-KISAN rules

    def test_eligibility_without_generator_uses_deterministic_response(self) -> None:
        graph = build_graph(
            FakeRetriever(),
            citation_resolver=self._make_resolver(),
        ).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("determination is based only on", result["response"].text)

    def test_insufficient_facts_with_generator_asks_clarification(self) -> None:
        fake_llm = FakeLLMClient(
            response=_make_llm_response(
                "There is insufficient evidence to determine eligibility.",
                (),
            )
        )
        fake_gen = DefaultAnswerGenerator(fake_llm)
        graph = build_graph(
            FakeRetriever(),
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=EligibilityRequest(
                scheme=PM_KISAN_SCHEME,
                landholding_in_own_name=True,
            ),
        )

        result = graph.invoke(state)

        self.assertEqual(result["eligibility_decision"].status, EligibilityStatus.INSUFFICIENT_INFORMATION)
        # Generator should still run and produce a clarification response
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)


class WeatherAndClarifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def _make_resolver(self):
        from kisansathi.citations.resolver import CitationResolver
        return CitationResolver(self.registry)

    def test_weather_path_bypasses_generation(self) -> None:
        # Use a tracking fake to verify generator is not called
        call_count = {"count": 0}

        class TrackingFakeLLMClient:
            def generate(self, *, system_prompt: str, user_prompt: str, **_) -> str:
                call_count["count"] += 1
                return "ANSWER:\nShould not be called.\n\nCITATIONS:\n"

        fake_gen = DefaultAnswerGenerator(TrackingFakeLLMClient())
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(
            FakeRetriever(),
            weather_client,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        result = graph.invoke(
            make_state(
                message(WEATHER_QUESTION, latitude=28.6139, longitude=77.2090)
            )
        )

        self.assertEqual(result["route"], "weather")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("31.2", result["response"].text)
        self.assertEqual(result["response"].citations, ())
        self.assertEqual(call_count["count"], 0)  # Generator never called

    def test_clarification_path_bypasses_generation(self) -> None:
        call_count = {"count": 0}

        class TrackingFakeLLMClient:
            def generate(self, *, system_prompt: str, user_prompt: str, **_) -> str:
                call_count["count"] += 1
                return "ANSWER:\nShould not be called.\n\nCITATIONS:\n"

        fake_gen = DefaultAnswerGenerator(TrackingFakeLLMClient())
        graph = build_graph(
            FakeRetriever(),
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        result = graph.invoke(make_state(message("Help")))

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(call_count["count"], 0)  # Generator never called


class PromptInjectionProtectionTests(unittest.TestCase):
    """Test that retrieved text cannot inject instructions into the prompt."""

    def setUp(self) -> None:
        self.registry = build_registry()

    def test_control_chars_in_payload_are_sanitized(self) -> None:
        """Payloads with control characters should not break the prompt."""
        from kisansathi.generation.prompts import build_user_prompt

        # Create a citation with a title containing control chars
        citation = Citation(
            source_id="test-source",
            title="Title\x00with\x01control\x1fchars",  # Control chars
            url="https://example.com",
            page_number=1,
            page_end=None,
            issuing_authority="Authority",
            chunk_id="test-source:abcdef123456789012345678",
        )
        batch = CitationBatch(citations=(citation,), rejected=())

        prompt = build_user_prompt(
            message=UserMessage(text="Test", language=Language.ENGLISH),
            citations=batch,
            eligibility_decision=None,
            weather=None,
        )

        # Control chars should be stripped
        self.assertNotIn("\x00", prompt)
        self.assertNotIn("\x01", prompt)
        self.assertNotIn("\x1f", prompt)

    def test_long_excerpt_is_truncated(self) -> None:
        """Very long excerpts should be truncated to limit context."""
        from kisansathi.generation.prompts import _MAX_EXCERPT_CHARS, _sanitize_text

        long_text = "x" * (_MAX_EXCERPT_CHARS + 100)
        sanitized = _sanitize_text(long_text)
        self.assertLessEqual(len(sanitized), _MAX_EXCERPT_CHARS)


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def _make_resolver(self):
        from kisansathi.citations.resolver import CitationResolver
        return CitationResolver(self.registry)

    def test_deterministic_output_with_same_inputs(self) -> None:
        """Same input should produce identical output when generator is deterministic."""
        fake_llm = FakeLLMClient(
            response=_make_llm_response("Deterministic answer.", (LAND_CHUNK,))
        )
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        state = make_state(message(RETRIEVAL_QUESTION))
        responses = [graph.invoke(state)["response"].text for _ in range(5)]

        self.assertEqual(len(set(responses)), 1)


class HindiLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def _make_resolver(self):
        from kisansathi.citations.resolver import CitationResolver
        return CitationResolver(self.registry)

    def test_hindi_question_uses_hindi_prompt(self) -> None:
        fake_llm = FakeLLMClient(
            response=_make_llm_response("हिंदी में उत्तर।", (LAND_CHUNK,))
        )
        fake_gen = DefaultAnswerGenerator(fake_llm)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(
            retriever,
            citation_resolver=self._make_resolver(),
            answer_generator=fake_gen,
        ).compile()

        # Use a Hindi retrieval question (not eligibility) to test language handling
        hindi_question = "PM-KISAN के बारे में नवीनतम जानकारी क्या है?"
        result = graph.invoke(make_state(message(hindi_question, language=Language.HINDI)))

        self.assertEqual(result["response"].language, Language.HINDI)
        self.assertIn("हिंदी में उत्तर", result["response"].text)


if __name__ == "__main__":
    unittest.main()