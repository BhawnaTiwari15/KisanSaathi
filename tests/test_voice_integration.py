"""Integration tests for voice input in LangGraph."""

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
from kisansathi.voice.models import (
    EmptyTranscriptError,
    InvalidAudioError,
    TranscriptionResult,
    TranscriptionError,
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
    }
    state.update(extra)
    return state  # type: ignore


def message(text: str, **kwargs: object) -> UserMessage:
    if not text or not text.strip():
        text = "audio input"  # placeholder for voice input
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


# Valid fake audio data (100+ bytes to satisfy MIN_AUDIO_SIZE_BYTES)
VALID_FAKE_AUDIO = b"x" * 150  # 150 bytes of fake audio data


def make_speech_to_text(text: str, language: Language = Language.ENGLISH, confidence: float = 1.0):
    return TranscriptionResult(text=text, language=language, confidence=confidence)


def _make_resolver():
    from kisansathi.citations.resolver import CitationResolver
    return CitationResolver(build_registry())


class VoiceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def _make_graph(self, speech_to_text=None, citation_resolver=None, answer_generator=None, retriever=None):
        return build_graph(
            retriever or FakeRetriever(),
            citation_resolver=citation_resolver or _make_resolver(),
            answer_generator=answer_generator,
            speech_to_text=speech_to_text,
        ).compile()

    def test_english_transcription_to_retrieval(self) -> None:
        """English audio transcribed and routed to retrieval."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="What are the PM-KISAN payment dates?",
                language=Language.ENGLISH,
                confidence=0.95,
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(speech_to_text=fake_stt, retriever=retriever)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["message"].text, "What are the PM-KISAN payment dates?")
        self.assertEqual(result["message"].language, Language.ENGLISH)

    def test_hindi_transcription_to_eligibility(self) -> None:
        """Hindi audio transcribed and routed to eligibility."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="क्या मैं PM-KISAN के लिए पात्र हूँ?",
                language=Language.HINDI,
                confidence=0.9,
            )
        )
        graph = self._make_graph(
            speech_to_text=fake_stt,
            citation_resolver=_make_resolver(),
            answer_generator=FakeAnswerGenerator(),
        )
        state = make_state(
            message(""),
            audio_data=VALID_FAKE_AUDIO,
            audio_content_type="audio/wav",
            eligibility_request=eligible_request(),
        )

        result = graph.invoke(state)

        self.assertEqual(result["route"], "eligibility")
        self.assertEqual(result["message"].text, "क्या मैं PM-KISAN के लिए पात्र हूँ?")
        self.assertEqual(result["message"].language, Language.HINDI)
        self.assertEqual(result["eligibility_decision"].status, EligibilityStatus.ELIGIBLE)

    def test_kannada_transcription_to_retrieval(self) -> None:
        """Kannada audio transcribed and routed to retrieval."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="PM-KISAN ಯೋಜನೆಯ ವಿವರಗಳನ್ನು ತಿಳಿ",
                language=Language.KANNADA,
                confidence=0.85,
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(speech_to_text=fake_stt, retriever=retriever)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["message"].text, "PM-KISAN ಯೋಜನೆಯ ವಿವರಗಳನ್ನು ತಿಳಿ")
        self.assertEqual(result["message"].language, Language.KANNADA)

    def test_telugu_transcription_to_retrieval(self) -> None:
        """Telugu audio transcribed and routed to retrieval."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="PM-KISAN యోజన వివరాలు తెలియజేయండి",
                language=Language.TELUGU,
                confidence=0.88,
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(speech_to_text=fake_stt, retriever=retriever)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["message"].text, "PM-KISAN యోజన వివరాలు తెలియజేయండి")
        self.assertEqual(result["message"].language, Language.TELUGU)

    def test_explicit_language_override(self) -> None:
        """Explicit language in UserMessage overrides Whisper detection."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="What is PM-KISAN?",
                language=Language.HINDI,  # Whisper detects Hindi
                confidence=0.9,
            )
        )
        # User explicitly set English in the message
        initial_message = UserMessage(text="What is PM-KISAN?", language=Language.ENGLISH)
        graph = self._make_graph(
            speech_to_text=FakeSpeechToText(
                TranscriptionResult(text="What is PM-KISAN?", language=Language.HINDI, confidence=0.9)
            ),
            retriever=FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),)),
        )

        result = graph.invoke(make_state(initial_message, audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        # Explicit language should be preserved in the transcribed message
        self.assertEqual(result["message"].language, Language.ENGLISH)

    def test_whisper_language_detection(self) -> None:
        """Whisper's detected language is used when no explicit language."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="PM-KISAN के बारे में बताएं",
                language=Language.HINDI,
                confidence=0.92,
            )
        )
        graph = self._make_graph(speech_to_text=fake_stt, retriever=FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),)))

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["message"].language, Language.HINDI)

    def test_whisper_detector_disagreement(self) -> None:
        """Whisper detects Hindi, but deterministic detector would say English - Whisper wins."""
        # The deterministic detector runs on the transcribed text
        # If Whisper says Hindi, the message.language=HINDI, so detector is not called
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="PM-KISAN के बारे में बताएं",  # Hindi text
                language=Language.HINDI,
                confidence=0.9,
            )
        )
        graph = self._make_graph(speech_to_text=fake_stt, retriever=FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),)))

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["message"].language, Language.HINDI)
        self.assertEqual(result["detected_language"], None)  # detector not called when explicit language present

    def test_unsupported_language_fallback(self) -> None:
        """When Whisper returns an unsupported language code, it falls back to detector/English.

        Since our TranscriptionResult only accepts supported Language enum values,
        we simulate an unsupported language by having the STT return None for language,
        which should trigger the detector fallback.
        """
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="Bonjour",
                language=None,  # Unsupported language -> None
                confidence=0.9,
            )
        )
        graph = self._make_graph(speech_to_text=fake_stt, retriever=FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),)))

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        # Unsupported language -> detector runs on the text
        # The deterministic detector only supports en/hi/kn/te, so "Bonjour" returns ENGLISH
        self.assertEqual(result["message"].language, Language.ENGLISH)

    def test_empty_transcript(self) -> None:
        """Empty transcript -> NEEDS_CLARIFICATION."""
        fake_stt = FakeSpeechToText(EmptyTranscriptError("empty"))
        graph = self._make_graph(speech_to_text=fake_stt)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertIn("could not understand", result["response"].text.lower())

    def test_transcription_failure(self) -> None:
        """Transcription service failure -> NEEDS_CLARIFICATION."""
        fake_stt = FakeSpeechToText(TranscriptionError("model timeout"))
        graph = self._make_graph(speech_to_text=fake_stt)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertIn("could not understand", result["response"].text.lower())

    def test_invalid_audio(self) -> None:
        """Invalid audio format -> ABSTAINED."""
        fake_stt = FakeSpeechToText()  # Won't be called
        graph = self._make_graph(speech_to_text=fake_stt)

        # Pass invalid audio (too small)
        result = graph.invoke(make_state(message(""), audio_data=b"x" * 50, audio_content_type="audio/wav"))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertIn("could not process", result["response"].text.lower())

    def test_transcription_service_unavailable(self) -> None:
        """Speech-to-text service unavailable -> ABSTAINED."""
        fake_stt = FakeSpeechToText(TranscriptionError("service unavailable"))
        graph = self._make_graph(speech_to_text=fake_stt)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertIn("unavailable", result["response"].text.lower())

    def test_invalid_audio_format(self) -> None:
        """Unsupported audio format -> ABSTAINED."""
        fake_stt = FakeSpeechToText()
        graph = self._make_graph(speech_to_text=fake_stt)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/unknown"))

        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_existing_text_path_unchanged(self) -> None:
        """Text input without audio follows original path."""
        graph = self._make_graph(retriever=FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),)))

        result = graph.invoke(make_state(message(RETRIEVAL_QUESTION)))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertEqual(len(result["response"].citations), 1)

    def test_clarification_path_with_audio(self) -> None:
        """Very short audio -> clarification."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(text="Hi", language=Language.ENGLISH, confidence=0.9)
        )
        graph = self._make_graph(speech_to_text=fake_stt)

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        # "Hi" is underspecified -> clarification
        self.assertEqual(result["route"], "clarify")
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_weather_path_with_audio(self) -> None:
        """Weather question via audio -> weather path."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="What's the weather today?",
                language=Language.ENGLISH,
                confidence=0.95,
            )
        )
        from kisansathi.weather.models import WeatherCurrent, WeatherResponse
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(
            FakeRetriever(),
            weather_client,
            citation_resolver=_make_resolver(),
            speech_to_text=FakeSpeechToText(
                TranscriptionResult(text="What's the weather today?", language=Language.ENGLISH, confidence=0.95)
            ),
        ).compile()

        result = graph.invoke(make_state(
            message("", latitude=28.6139, longitude=77.2090),
            audio_data=VALID_FAKE_AUDIO,
            audio_content_type="audio/wav",
        ))

        self.assertEqual(result["route"], "weather")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("31.2", result["response"].text)

    def test_citation_preservation_in_voice_path(self) -> None:
        """Citation IDs preserved exactly in voice path."""
        fake_stt = FakeSpeechToText(
            TranscriptionResult(
                text="When are PM-KISAN payments made?",
                language=Language.ENGLISH,
                confidence=0.95,
            )
        )
        retriever = FakeRetriever((make_payload(LAND_CHUNK, page_start=4, page_end=4),))
        graph = self._make_graph(
            speech_to_text=FakeSpeechToText(
                TranscriptionResult(text="When are PM-KISAN payments made?", language=Language.ENGLISH, confidence=0.95)
            ),
            retriever=retriever,
            answer_generator=FakeAnswerGenerator(),
        )

        result = graph.invoke(make_state(message(""), audio_data=VALID_FAKE_AUDIO, audio_content_type="audio/wav"))

        self.assertEqual(len(result["response"].citations), 1)
        self.assertEqual(result["response"].citations[0].chunk_id, LAND_CHUNK)


class FakeWeatherClient:
    def __init__(self, response: WeatherResponse) -> None:
        self.response = response
        self.calls: list = []
    
    def get_forecast(self, req, *, include_forecast: bool = False, forecast_days: int = 1) -> WeatherResponse:
            self.calls.append(req)
            return self.response
        
    
if __name__ == "__main__":
    unittest.main()