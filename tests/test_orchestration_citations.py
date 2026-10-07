"""Graph-level citation integration for retrieval and eligibility routes.

These tests pin the boundary between orchestration and the citation layer: the graph
selects evidence and attaches whatever the resolver returns, while the resolver alone
decides what is citable. Provenance comes from the real ``data/sources.json`` manifest and
from the real deterministic PM-KISAN rule set, so a passing test means the graph is wired
to genuine registered sources rather than to test doubles of them.
"""

from pathlib import Path
from typing import Any, cast
import re
import unittest

from kisansathi.citations.models import CitationError
from kisansathi.citations.registry import ManifestSourceRegistry
from kisansathi.citations.resolver import CitationResolver
from kisansathi.domain.schemas import (
    AssistantResponse,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.eligibility.evaluator import evaluate
from kisansathi.eligibility.models import FACT_NAMES, EligibilityRequest
from kisansathi.eligibility.pm_kisan_rules import PM_KISAN_SCHEME
from kisansathi.orchestration.graph import OrchestrationState, build_graph
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.models import WeatherCurrent, WeatherRequest, WeatherResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "sources.json"

SOURCE_ID = "pm-kisan-revised-faq"
CORPUS_SHA256 = "95d4e2892b4b351e599bf903f8c6f89b20d4b74831eb8f1a0e61a2a66771ff86"
LAND_CHUNK = f"{SOURCE_ID}:33ff2db494c59ab1ca6c4f76"
CULTIVABLE_CHUNK = f"{SOURCE_ID}:b148dd12de05914c6bfaf324"
AGRICULTURAL_CHUNK = f"{SOURCE_ID}:f39a38edb9e1352d1315ece3"
TAX_CHUNK = f"{SOURCE_ID}:1fe005b874ba210ad336e5d9"

ELIGIBILITY_QUESTION = "Please tell me whether I am eligible under PM-KISAN."
RETRIEVAL_QUESTION = "What does the revised PM-KISAN guideline say about payment cycles?"
WEATHER_QUESTION = "What is the weather forecast for my field today?"


def build_registry() -> ManifestSourceRegistry:
    return ManifestSourceRegistry.from_manifest(MANIFEST_PATH)


def make_payload(
    chunk_id: str,
    *,
    page_start: int,
    page_end: int,
    **overrides: object,
) -> SearchResult:
    payload: dict[str, Any] = {
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
        self.calls: list[WeatherRequest] = []

    def get_forecast(
        self,
        req: WeatherRequest,
        *,
        include_forecast: bool = False,
        forecast_days: int = 1,
    ) -> WeatherResponse:
        self.calls.append(req)
        return self.response


class RecordingResolver(CitationResolver):
    """A real resolver that also records how it was called.

    It delegates every decision to the production resolver, so these tests cannot pass by
    reimplementing citation rules; they only observe which entry point the graph used.
    """

    def __init__(
        self,
        registry: ManifestSourceRegistry,
        *,
        raise_exc: Exception | None = None,
    ) -> None:
        super().__init__(registry)
        self.raise_exc = raise_exc
        self.payload_calls: list[tuple[str, ...]] = []
        self.evidence_calls: list[tuple[str, ...]] = []

    def resolve_payloads(self, payloads: Any) -> Any:
        materialized = tuple(payloads)
        self.payload_calls.append(tuple(str(item.get("chunk_id")) for item in materialized))
        if self.raise_exc is not None:
            raise self.raise_exc
        return super().resolve_payloads(materialized)

    def resolve_evidence_batch(self, evidence: Any) -> Any:
        materialized = tuple(evidence)
        self.evidence_calls.append(tuple(item.chunk_id for item in materialized))
        if self.raise_exc is not None:
            raise self.raise_exc
        return super().resolve_evidence_batch(materialized)


class RecordingEvaluator:
    """Delegates to the production evaluator while recording the requests it received."""

    def __init__(self) -> None:
        self.calls: list[EligibilityRequest] = []

    def __call__(self, request: EligibilityRequest, *, rules: Any = None) -> Any:
        self.calls.append(request)
        return evaluate(request, rules=rules)


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
    state: dict[str, Any] = {
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
    return cast(OrchestrationState, state)


def message(text: str, **kwargs: object) -> UserMessage:
    defaults: dict[str, Any] = {"text": text, "language": Language.ENGLISH}
    defaults.update(kwargs)
    return UserMessage(**defaults)


def eligible_request(**overrides: Any) -> EligibilityRequest:
    scheme = overrides.pop("scheme", PM_KISAN_SCHEME)
    facts: dict[str, Any] = {
        "landholding_in_own_name": True,
        "land_is_cultivable": True,
        "land_used_for_non_agricultural_purpose": False,
        "family_member_paid_income_tax_last_assessment_year": False,
    }
    facts.update(overrides)
    return EligibilityRequest(scheme=scheme, **facts)


class RetrievalCitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_retrieval_route_produces_validated_citations(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever(
            (make_payload(LAND_CHUNK, page_start=4, page_end=4),)
        )
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(len(result["response"].citations), 1)
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)

    def test_retrieval_citation_uses_registered_metadata(self) -> None:
        resolver = RecordingResolver(self.registry)
        entry = self.registry.get(SOURCE_ID)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        citation = result["response"].citations[0]
        self.assertEqual(citation.source_id, SOURCE_ID)
        self.assertEqual(citation.title, entry.title)
        self.assertEqual(citation.url, entry.source_url)
        self.assertEqual(citation.issuing_authority, entry.issuing_authority)
        self.assertEqual(citation.page_number, 4)
        self.assertIsNone(citation.page_end)
        self.assertEqual(citation.chunk_id, LAND_CHUNK)

    def test_title_mismatch_in_payload_is_refused(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever(
            (
                make_payload(
                    LAND_CHUNK,
                    page_start=4,
                    page_end=4,
                    title="A title that contradicts the manifest",
                ),
            )
        )
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].citations, ())
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_multi_page_chunk_reports_page_span(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever((make_payload(TAX_CHUNK, page_start=3, page_end=4),))
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        citation = result["response"].citations[0]
        self.assertEqual(citation.page_number, 3)
        self.assertEqual(citation.page_end, 4)
        self.assertEqual(citation.page_span, (3, 4))

    def test_mixed_provenance_cites_only_the_valid_chunks(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever(
            (
                make_payload(LAND_CHUNK, page_start=4, page_end=4),
                make_payload("not-a-valid-chunk-id", page_start=1, page_end=1),
            )
        )
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        citations = result["response"].citations
        self.assertEqual([c.chunk_id for c in citations], [LAND_CHUNK])
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)

    def test_unregistered_source_abstains_and_fabricates_nothing(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever(
            (
                make_payload(
                    "unknown-source:33ff2db494c59ab1ca6c4f76",
                    page_start=4,
                    page_end=4,
                    source_id="unknown-source",
                ),
            )
        )
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].citations, ())
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_malformed_chunk_id_abstains(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever((make_payload("c1", page_start=1, page_end=1),))
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].citations, ())
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_invalid_page_range_abstains(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=7, page_end=4),))
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].citations, ())
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_resolver_failure_abstains(self) -> None:
        resolver = RecordingResolver(self.registry, raise_exc=CitationError("boom"))
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].citations, ())
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_retrieval_cites_each_chunk_once(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever(
            (
                make_payload(LAND_CHUNK, page_start=4, page_end=4),
                make_payload(LAND_CHUNK, page_start=4, page_end=4),
            )
        )
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(len(result["response"].citations), 1)


class EligibilityCitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_eligibility_route_produces_citations(self) -> None:
        resolver = RecordingResolver(self.registry)
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        self.assertEqual(result["route"], "eligibility")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertEqual(
            {c.chunk_id for c in result["response"].citations},
            {LAND_CHUNK, CULTIVABLE_CHUNK, AGRICULTURAL_CHUNK, TAX_CHUNK},
        )

    def test_ineligible_cites_only_the_violated_rule(self) -> None:
        resolver = RecordingResolver(self.registry)
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(landholding_in_own_name=False),
        )

        result = graph.invoke(state)

        self.assertEqual(result["eligibility_decision"].status.value, "ineligible")
        citations = result["response"].citations
        self.assertEqual([c.chunk_id for c in citations], [LAND_CHUNK])
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)

    def test_partial_facts_yield_clarification_without_citations(self) -> None:
        resolver = RecordingResolver(self.registry)
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=EligibilityRequest(
                scheme=PM_KISAN_SCHEME,
                landholding_in_own_name=True,
            ),
        )

        result = graph.invoke(state)

        self.assertEqual(
            result["eligibility_decision"].status.value,
            "insufficient_information",
        )
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result["response"].citations, ())
        self.assertEqual(result["eligibility_decision"].missing_facts, FACT_NAMES[1:])
        for fact in result["eligibility_decision"].missing_facts:
            self.assertIn(fact, result["response"].text)

    def test_absent_request_yields_clarification_not_a_guess(self) -> None:
        resolver = RecordingResolver(self.registry)
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message(ELIGIBILITY_QUESTION)))

        self.assertEqual(result["route"], "eligibility")
        self.assertIsNone(result["eligibility_decision"])
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result["response"].citations, ())
        for fact in FACT_NAMES:
            self.assertIn(fact, result["response"].text)

    def test_unsupported_scheme_abstains_without_citations(self) -> None:
        resolver = RecordingResolver(self.registry)
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(scheme="some-other-scheme"),
        )

        result = graph.invoke(state)

        self.assertEqual(result["eligibility_decision"].status.value, "unsupported_scheme")
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertEqual(result["response"].citations, ())

    def test_verdict_without_citable_evidence_is_withheld(self) -> None:
        resolver = RecordingResolver(self.registry, raise_exc=CitationError("boom"))
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        self.assertEqual(result["eligibility_decision"].status.value, "eligible")
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertEqual(result["response"].citations, ())

    def test_malformed_request_is_not_trusted(self) -> None:
        evaluator = RecordingEvaluator()
        resolver = RecordingResolver(self.registry)
        graph = build_graph(
            FakeRetriever(),
            citation_resolver=resolver,
            eligibility_evaluator=evaluator,
        ).compile()
        bogus = cast(
            EligibilityRequest,
            {"scheme": PM_KISAN_SCHEME, "landholding_in_own_name": "yes", "invented_fact": True},
        )
        state = make_state(message(ELIGIBILITY_QUESTION), eligibility_request=bogus)

        result = graph.invoke(state)

        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result["eligibility_decision"].status.value, "insufficient_information")
        self.assertEqual(
            result["eligibility_decision"].missing_facts,
            FACT_NAMES,
        )

    def test_evaluator_receives_only_sanitised_facts(self) -> None:
        evaluator = RecordingEvaluator()
        resolver = RecordingResolver(self.registry)
        graph = build_graph(
            FakeRetriever(),
            citation_resolver=resolver,
            eligibility_evaluator=evaluator,
        ).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(),
        )

        graph.invoke(state)

        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(evaluator.calls[0].scheme, PM_KISAN_SCHEME)
        self.assertEqual(evaluator.calls[0].missing_facts(), ())


class ResolverInvocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_retrieval_calls_the_payload_entry_point_once(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever(
            (
                make_payload(LAND_CHUNK, page_start=4, page_end=4),
                make_payload(TAX_CHUNK, page_start=3, page_end=4),
            )
        )
        graph = build_graph(retriever, citation_resolver=resolver).compile()

        graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(resolver.payload_calls, [(LAND_CHUNK, TAX_CHUNK)])
        self.assertEqual(resolver.evidence_calls, [])

    def test_eligibility_calls_the_evidence_entry_point_once(self) -> None:
        resolver = RecordingResolver(self.registry)
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(),
        )

        graph.invoke(state)

        self.assertEqual(resolver.payload_calls, [])
        self.assertEqual(len(resolver.evidence_calls), 1)

    def test_clarification_never_calls_the_resolver(self) -> None:
        resolver = RecordingResolver(self.registry)
        graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()

        result = graph.invoke(make_state(message("Help")))

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result["response"].citations, ())
        self.assertEqual(resolver.payload_calls, [])
        self.assertEqual(resolver.evidence_calls, [])

    def test_weather_never_calls_the_resolver_and_still_works(self) -> None:
        resolver = RecordingResolver(self.registry)
        retriever = FakeRetriever()
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client, citation_resolver=resolver).compile()

        result = graph.invoke(
            make_state(message(WEATHER_QUESTION, latitude=28.6139, longitude=77.2090))
        )

        self.assertEqual(result["route"], "weather")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("31.2", result["response"].text)
        self.assertEqual(result["response"].citations, ())
        self.assertEqual(len(weather_client.calls), 1)
        self.assertEqual(retriever.calls, [])
        self.assertEqual(resolver.payload_calls, [])
        self.assertEqual(resolver.evidence_calls, [])

    def test_weather_intent_outranks_eligibility_intent(self) -> None:
        resolver = RecordingResolver(self.registry)
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(
            FakeRetriever(), weather_client, citation_resolver=resolver
        ).compile()

        result = graph.invoke(
            make_state(
                message(
                    "Is the weather affecting my eligibility for PM-KISAN?",
                    latitude=28.6139,
                    longitude=77.2090,
                )
            )
        )

        self.assertEqual(result["route"], "weather")


class PreservedBehaviourTests(unittest.TestCase):
    def test_retrieval_without_resolver_is_unchanged(self) -> None:
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = build_graph(retriever).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertEqual(result["response"].citations, ())
        self.assertIn(
            "retrieved relevant excerpts",
            result["response"].text.lower(),
        )
        self.assertIn(
            "answer generation has not yet been implemented",
            result["response"].text.lower(),
        )

    def test_empty_retrieval_abstains_when_no_chunks(self) -> None:
        graph = build_graph(
            FakeRetriever(),
            citation_resolver=RecordingResolver(build_registry()),
        ).compile()

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertEqual(result["response"].citations, ())

    def test_document_questions_still_reach_retrieval(self) -> None:
        resolver = RecordingResolver(build_registry())
        texts = (
            "What documents are required for PM-KISAN enrollment?",
            "Which categories are excluded from PM-KISAN benefits?",
            "Does the scheme cover training for grain storage and drainage work?",
            "When was PM-KISAN launched and how often are payments made?",
        )
        for text in texts:
            with self.subTest(text=text):
                graph = build_graph(FakeRetriever(), citation_resolver=resolver).compile()
                result = graph.invoke(make_state(message(text)))
                self.assertEqual(result["route"], "retrieval")

    def test_eligibility_answer_text_is_deterministic(self) -> None:
        registry = build_registry()
        graph = build_graph(
            FakeRetriever(), citation_resolver=RecordingResolver(registry)
        ).compile()
        state = make_state(
            message(ELIGIBILITY_QUESTION),
            eligibility_request=eligible_request(),
        )

        responses = {
            graph.invoke(state)["response"].text
            for _ in range(25)
        }

        self.assertEqual(len(responses), 1)

    def test_graph_module_does_not_construct_citations(self) -> None:
        source = (REPO_ROOT / "src" / "kisansathi" / "orchestration" / "graph.py").read_text(
            encoding="utf-8"
        )

        for forbidden in (
            r"\bCitation\(",
            r"ManifestSourceRegistry",
            r"load_manifest",
            r"CitationResolver\(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIsNone(re.search(forbidden, source))

    def test_graph_module_creates_only_empty_batches(self) -> None:
        """The graph may create a batch, but only to record refusals, never to add citations."""
        source = (REPO_ROOT / "src" / "kisansathi" / "orchestration" / "graph.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("CitationBatch(citations", source)
        self.assertNotIn("citations=(", source.replace("citations=(),", ""))


if __name__ == "__main__":
    unittest.main()
