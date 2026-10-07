"""Integration tests for vision input in LangGraph."""

import unittest
from pathlib import Path

from kisansathi.citations import ManifestSourceRegistry
from kisansathi.citations.models import CitationBatch
from kisansathi.domain.schemas import (
    AssistantResponse,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.eligibility import PM_KISAN_SCHEME, EligibilityRequest
from kisansathi.eligibility.models import EligibilityStatus
from kisansathi.orchestration.graph import OrchestrationState, build_graph
from kisansathi.generation.fake import FakeAnswerGenerator
from kisansathi.voice.fake import FakeSpeechToText
from kisansathi.voice.models import TranscriptionResult
from kisansathi.vision import (
    VisionResult,
    VisualObservation,
    VisionStatus,
    FakeVisionAnalyzer,
    make_fake_vision_analyzer,
)
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.models import WeatherCurrent, WeatherResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "sources.json"

SOURCE_ID = "pm-kisan-revised-faq"
CORPUS_SHA256 = "95d4e2892b4b351e599bf903f8c6f89b20d4b74831eb8f1a0e61a2a66771ff86"
LAND_CHUNK = f"{SOURCE_ID}:33ff2db494c59ab1ca6c4f76"
ALL_PM_KISAN_CHUNKS = (
    f"{SOURCE_ID}:33ff2db494c59ab1ca6c4f76",
    f"{SOURCE_ID}:b148dd12de05914c6bfaf324",
    f"{SOURCE_ID}:f39a38edb9e1352d1315ece3",
    f"{SOURCE_ID}:1fe005b874ba210ad336e5d9",
)

ELIGIBILITY_QUESTION = "Please tell me whether I am eligible under PM-KISAN."
RETRIEVAL_QUESTION = "What does the revised PM-KISAN guideline say about payment cycles?"


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
        "audio_data": None,
        "audio_content_type": None,
        "image_data": None,
        "image_content_type": None,
        "image_filename": None,
    }
    state.update(extra)
    return state  # type: ignore


def message(text: str, **kwargs: object) -> UserMessage:
    if not text or not text.strip():
        text = "image input"  # placeholder for image-only input
    defaults: dict[str, object] = {"text": text}
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


def _make_resolver():
    from kisansathi.citations.resolver import CitationResolver
    return CitationResolver(build_registry())


# Valid fake image data (JPEG > 100 bytes)
VALID_FAKE_IMAGE = (
    b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0C\x14\r\x0C\x0B\x0B\x0C\x19\x12\x13\x0F\x14\x1D\x1A"
    b"\x1F\x1E\x1D\x1A\x1C\x1C $.' \",#\x1C\x1C(7),01444\x1F'9=82<.342"
    b"\xFF\xC0\x00\x0B\x08\x00\x01\x00\x01\x01\x11\x00\xFF\xD9"
)

# Valid fake audio data (100+ bytes to satisfy MIN_AUDIO_SIZE_BYTES)
VALID_FAKE_AUDIO = b"x" * 150  # 150 bytes of fake audio data


def make_vision_result(
    status: VisionStatus = VisionStatus.SUCCESS,
    observations: tuple[VisualObservation, ...] | None = None,
    language: str = "en",
) -> VisionResult:
    """Create a VisionResult for testing."""
    if observations is None and status == VisionStatus.SUCCESS:
        observations = (
            VisualObservation(
                label="tomato plant",
                category="crop",
                confidence=0.9,
                description="Healthy tomato plant",
            ),
        )
    return VisionResult(
        status=status,
        observations=observations or (),
        language=language,
        confidence_overall=0.9 if status == VisionStatus.SUCCESS else 0.0,
    )


class VisionInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def _make_graph(
        self,
        speech_to_text=None,
        vision_analyzer=None,
        citation_resolver=None,
        answer_generator=None,
        retriever=None,
    ):
        return build_graph(
            retriever or FakeRetriever(),
            citation_resolver=citation_resolver or _make_resolver(),
            answer_generator=answer_generator,
            speech_to_text=speech_to_text,
            vision_analyzer=vision_analyzer,
        ).compile()

    def test_image_only_retrieval_route(self) -> None:
        """Image-only input should be analyzed and routed to retrieval for crop observations."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(
                    label="wheat crop",
                    category="crop",
                    confidence=0.9,
                    description="Wheat field ready for harvest",
                ),
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(vision_analyzer=vision_analyzer, retriever=retriever)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
            image_filename="crop.jpg",
        ))

        self.assertEqual(result["route"], "retrieval")
        self.assertIsNotNone(result.get("vision_result"))
        self.assertEqual(result["vision_result"].status, VisionStatus.SUCCESS)
        self.assertTrue(result["vision_result"].has_crop_observation)
        self.assertEqual(len(retriever.calls), 1)

    def test_image_only_eligibility_route(self) -> None:
        """Image with disease/pest should route to eligibility."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(
                    label="late blight",
                    category="disease",
                    confidence=0.85,
                    description="Late blight on tomato leaves",
                ),
            )
        )
        graph = self._make_graph(
            vision_analyzer=vision_analyzer,
            citation_resolver=_make_resolver(),
            answer_generator=FakeAnswerGenerator(),
        )
        state = make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
            image_filename="diseased.jpg",
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        self.assertEqual(result["route"], "eligibility")
        self.assertTrue(result["vision_result"].has_disease_or_pest)

    def test_text_plus_image_preserves_text(self) -> None:
        """Text + image should preserve text and add visual context."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(
                    label="tomato plant",
                    category="crop",
                    confidence=0.9,
                    description="Healthy tomato plant",
                ),
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(vision_analyzer=vision_analyzer, retriever=retriever)

        result = graph.invoke(make_state(
            message("What fertilizer should I use for my tomatoes?"),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
            image_filename="tomato.jpg",
        ))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["message"].text, "What fertilizer should I use for my tomatoes?")
        self.assertIsNotNone(result.get("vision_result"))
        self.assertTrue(result["vision_result"].has_crop_observation)

    def test_audio_plus_image_transcribes_first(self) -> None:
        """Audio + image: audio transcribed first, then vision, then combined through pipeline."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="What disease does my tomato plant have?",
                language=Language.ENGLISH,
                confidence=0.95,
            )
        )
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(
                    label="early blight",
                    category="disease",
                    confidence=0.8,
                    description="Early blight symptoms on tomato leaves",
                ),
            )
        )
        graph = self._make_graph(
            speech_to_text=fake_stt,
            vision_analyzer=vision_analyzer,
            citation_resolver=_make_resolver(),
            answer_generator=FakeAnswerGenerator(),
        )
        state = make_state(
            message(""),
            audio_data=VALID_FAKE_AUDIO,
            audio_content_type="audio/wav",
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
            image_filename="tomato.jpg",
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        # Audio transcribed first
        self.assertEqual(result["message"].text, "What disease does my tomato plant have?")
        self.assertEqual(result["message"].language, Language.ENGLISH)
        # Then vision analyzed
        self.assertIsNotNone(result.get("vision_result"))
        self.assertTrue(result["vision_result"].has_disease_or_pest)
        # Routed to eligibility due to disease
        self.assertEqual(result["route"], "eligibility")

    def test_invalid_image_format_abstained(self) -> None:
        """Invalid image format -> ABSTAINED."""
        vision_analyzer = make_fake_vision_analyzer()  # Won't be called due to validation error
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        # Corrupted JPEG data
        corrupted = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50
        result = graph.invoke(make_state(
            message(""),
            image_data=corrupted,
            image_content_type="image/jpeg",
        ))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertIn("could not process", result["response"].text.lower())

    def test_unsupported_format_abstained(self) -> None:
        """Unsupported image format -> ABSTAINED."""
        vision_analyzer = make_fake_vision_analyzer()
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/gif",  # Unsupported
        ))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertIn("could not process", result["response"].text.lower())

    def test_oversized_image_abstained(self) -> None:
        """Oversized image -> ABSTAINED."""
        from kisansathi.vision.image import MAX_IMAGE_SIZE_BYTES
        vision_analyzer = make_fake_vision_analyzer()
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        oversized = b"\xFF\xD8\xFF\xE0" + b"\x00" * (MAX_IMAGE_SIZE_BYTES + 100)
        result = graph.invoke(make_state(
            message(""),
            image_data=oversized,
            image_content_type="image/jpeg",
        ))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_analyzer_unavailable_abstained(self) -> None:
        """Analyzer unavailable -> ABSTAINED."""
        from kisansathi.vision.models import AnalyzerUnavailableError
        vision_analyzer = make_fake_vision_analyzer(
            error=AnalyzerUnavailableError("Vision service unavailable")
        )
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertIn("unavailable", result["response"].text.lower())

    def test_low_confidence_result_clarification(self) -> None:
        """Low confidence vision result -> NEEDS_CLARIFICATION."""
        vision_analyzer = make_fake_vision_analyzer(status="low_confidence")
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_unable_to_analyze_clarification(self) -> None:
        """Unable to analyze -> NEEDS_CLARIFICATION."""
        vision_analyzer = make_fake_vision_analyzer(status="unable_to_analyze")
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_vision_observation_reaches_retrieval(self) -> None:
        """Vision observations should augment retrieval query."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(label="wheat", category="crop", confidence=0.9, description="Wheat field"),
                VisualObservation(label="healthy", category="healthy", confidence=0.8, description="Healthy crop"),
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(vision_analyzer=vision_analyzer, retriever=retriever)

        result = graph.invoke(make_state(
            message("How to treat this?"),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        # Retrieval query should include vision keywords
        self.assertEqual(len(retriever.calls), 1)
        query = retriever.calls[0][0]
        self.assertIn("How to treat this?", query)
        self.assertIn("wheat", query)
        self.assertIn("healthy", query)

    def test_vision_uncertainty_reaches_generation(self) -> None:
        """Uncertain vision observations should be marked in generation context."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(label="tomato", category="crop", confidence=0.9, description="Tomato plant"),
                VisualObservation(label="possible pest", category="pest", confidence=0.4, description="Possible pest damage"),
            )
        )
        graph = self._make_graph(
            vision_analyzer=vision_analyzer,
            citation_resolver=_make_resolver(),
            answer_generator=FakeAnswerGenerator(),
        )

        result = graph.invoke(make_state(
            message("What do you see?"),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        # Check that vision result with uncertainty was passed to generator
        self.assertIsNotNone(result.get("vision_result"))
        vr = result["vision_result"]
        self.assertTrue(vr.has_uncertain_observations)
        self.assertTrue(vr.has_disease_or_pest)

    def test_citation_preservation_in_vision_path(self) -> None:
        """Citation IDs preserved exactly in vision + retrieval path."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(label="tomato", category="crop", confidence=0.9, description="Tomato plant"),
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(
            vision_analyzer=vision_analyzer,
            retriever=retriever,
            answer_generator=FakeAnswerGenerator(),
        )

        result = graph.invoke(make_state(
            message("When are payments made?"),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        self.assertEqual(len(result["response"].citations), 1)
        self.assertEqual(result["response"].citations[0].chunk_id, LAND_CHUNK)

    def test_downstream_routing_disease_to_eligibility(self) -> None:
        """Disease/pest observations should route to eligibility."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(label="aphids", category="pest", confidence=0.85, description="Aphid infestation"),
            )
        )
        graph = self._make_graph(
            vision_analyzer=vision_analyzer,
            citation_resolver=_make_resolver(),
            answer_generator=FakeAnswerGenerator(),
        )
        state = make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        self.assertEqual(result["route"], "eligibility")
        self.assertEqual(result["eligibility_decision"].status, EligibilityStatus.ELIGIBLE)

    def test_downstream_routing_crop_to_retrieval(self) -> None:
        """Crop observations should route to retrieval."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(label="rice", category="crop", confidence=0.9, description="Rice paddy"),
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(vision_analyzer=vision_analyzer, retriever=retriever)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(len(retriever.calls), 1)

    def test_existing_text_only_behavior_unchanged(self) -> None:
        """Text input without image follows original path."""
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(retriever=retriever)

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertEqual(len(result["response"].citations), 1)

    def test_existing_audio_only_behavior_unchanged(self) -> None:
        """Audio input without image follows original path."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="What are PM-KISAN payment dates?",
                language=Language.ENGLISH,
                confidence=0.95,
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(speech_to_text=fake_stt, retriever=retriever)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["message"].text, "What are PM-KISAN payment dates?")

    def test_no_vision_analyzer_preserves_text_path(self) -> None:
        """When no vision analyzer is injected, text-only path is unchanged."""
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(retriever=retriever)

        result = graph.invoke(make_state(
            message("What is PM-KISAN?"),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        # Vision node not executed, image data passes through
        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["message"].text, "What is PM-KISAN?")

    def test_privacy_no_raw_image_logged(self) -> None:
        """Vision analyzer should not log raw image bytes."""
        vision_analyzer = make_fake_vision_analyzer()
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
        ))

        # Check calls only contain metadata, not raw bytes
        for call in vision_analyzer.calls:
            self.assertIn("image_size", call)
            self.assertNotIn("image_data", call)
            self.assertEqual(call["image_size"], len(VALID_FAKE_IMAGE))

    def test_image_cleared_after_analysis(self) -> None:
        """Image data should be cleared from state after analysis."""
        vision_analyzer = make_fake_vision_analyzer()
        graph = self._make_graph(vision_analyzer=vision_analyzer)

        result = graph.invoke(make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
            image_filename="test.jpg",
        ))

        # Image data cleared from state
        self.assertIsNone(result.get("image_data"))
        self.assertIsNone(result.get("image_content_type"))
        self.assertIsNone(result.get("image_filename"))

    def test_vision_result_structure_preserved(self) -> None:
        """VisionResult structure should be preserved through pipeline."""
        vision_analyzer = make_fake_vision_analyzer(
            observations=(
                VisualObservation(label="maize", category="crop", confidence=0.95, description="Maize crop"),
                VisualObservation(label="fall armyworm", category="pest", confidence=0.7, description="Fall armyworm damage"),
            ),
            confidence=0.95,
        )
        graph = self._make_graph(
            vision_analyzer=vision_analyzer,
            citation_resolver=_make_resolver(),
            answer_generator=FakeAnswerGenerator(),
        )
        state = make_state(
            message(""),
            image_data=VALID_FAKE_IMAGE,
            image_content_type="image/jpeg",
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        vr = result["vision_result"]
        self.assertEqual(vr.status, VisionStatus.SUCCESS)
        self.assertEqual(len(vr.observations), 2)
        self.assertEqual(vr.observations[0].label, "maize")
        self.assertEqual(vr.observations[1].label, "fall armyworm")
        self.assertEqual(vr.observations[0].category, "crop")
        self.assertEqual(vr.observations[1].category, "pest")
        self.assertAlmostEqual(vr.confidence_overall, 0.95, places=1)


if __name__ == "__main__":
    unittest.main()