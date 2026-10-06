"""Deterministic LangGraph orchestration skeleton."""

from collections.abc import Mapping
from enum import StrEnum
from typing import NotRequired, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from kisansathi.citations.models import CitationBatch, CitationError, RejectedCitation
from kisansathi.citations.resolver import CitationResolver, validate_referenced_citations
from kisansathi.domain.schemas import (
    AssistantResponse,
    Citation,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.eligibility.evaluator import evaluate
from kisansathi.eligibility.models import (
    FACT_NAMES,
    EligibilityDecision,
    EligibilityError,
    EligibilityRequest,
    EligibilityStatus,
)
from kisansathi.generation.models import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationContext,
    GenerationError,
    GroundingError,
    LLMError,
    MalformedOutputError,
)
from kisansathi.language import DeterministicLanguageDetector, LanguageDetector
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.voice import (
    EmptyTranscriptError,
    InvalidAudioError,
    SpeechToText,
    SpeechToTextError,
    TranscriptionError,
    TranscriptionResult,
)
from kisansathi.weather.exceptions import WeatherError
from kisansathi.weather.models import WeatherCurrent, WeatherRequest, WeatherResponse


class Route(StrEnum):
    """The routes the request may take out of ``route_request``."""

    RETRIEVAL = "retrieval"
    ELIGIBILITY = "eligibility"
    WEATHER = "weather"
    CLARIFY = "clarify"
    FINALIZE = "finalize"


_ROUTE_NODES: dict[str, str] = {
    Route.RETRIEVAL: "retrieve",
    Route.ELIGIBILITY: "eligibility",
    Route.WEATHER: "weather",
    Route.CLARIFY: "clarify",
    Route.FINALIZE: "finalize_response",
}

_TOKEN_STRIP = ".,?!:;\"'()[]{}-"

# Marker for provenance that failed to resolve before it could be attributed to any
# source. It is only ever a rejection reason, never a citation.
_UNRESOLVED_SOURCE = "<unresolved>"

_WEATHER_KEYWORDS = frozenset(
    {
        "weather",
        "forecast",
        "rain",
        "rainfall",
        "temperature",
        "मौसम",
        "बारिश",
        "तापमान",
    }
)

# Deliberately narrow. Terms such as "enrollment", "excluded" and "categories" are
# excluded because they also occur in document questions that must keep reaching
# retrieval rather than the eligibility evaluator.
_ELIGIBILITY_KEYWORDS = frozenset(
    {
        "eligible",
        "eligibility",
        "योग्य",
        "पात्र",
        "पात्रता",
    }
)


class OrchestrationState(TypedDict):
    """Graph state for the orchestration skeleton."""

    message: UserMessage
    route: str
    retrieved_chunks: tuple[SearchResult, ...]
    response: AssistantResponse
    weather: NotRequired[WeatherResponse | None]
    citations: NotRequired[CitationBatch | None]
    eligibility_decision: NotRequired[EligibilityDecision | None]
    eligibility_request: NotRequired[EligibilityRequest | None]
    generated_answer: NotRequired[GeneratedAnswer | None]
    validated_citations: NotRequired[tuple[Citation, ...] | None]
    detected_language: NotRequired[Language | None]
    audio_data: NotRequired[bytes | None]
    audio_content_type: NotRequired[str | None]


class _RetrieverProtocol:
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        raise NotImplementedError


class _WeatherClientProtocol(Protocol):
    def get_forecast(
        self,
        req: WeatherRequest,
        *,
        include_forecast: bool = False,
        forecast_days: int = 1,
    ) -> WeatherResponse:
        """Return current weather for the requested coordinates."""
        ...


class _EligibilityEvaluatorProtocol(Protocol):
    def __call__(
        self, request: EligibilityRequest, *, rules: object = ...
    ) -> EligibilityDecision:
        """Evaluate one eligibility request and return its decision."""
        ...


class _AnswerGeneratorProtocol(Protocol):
    def generate(self, context: object) -> GeneratedAnswer:
        """Generate an answer from grounded context."""
        ...


class _LanguageDetectorProtocol(Protocol):
    def detect(self, text: str) -> Language | None:
        """Detect the language of the given text."""
        ...


class _SpeechToTextProtocol(Protocol):
    def transcribe(self, audio_data: bytes, *, content_type: str | None = None) -> TranscriptionResult:
        """Transcribe audio data to text."""
        ...


def _is_underspecified(message: UserMessage) -> bool:
    """Return True if the user message is very short or underspecified."""
    text = message.text.strip()
    if len(text) < 10:
        return True
    words = [token for token in text.split() if token]
    if len(words) < 3:
        return True
    return False


def _has_weather_intent(message: UserMessage) -> bool:
    """Return True if the message matches a provisional weather keyword.

    Provisional by design: a fixed keyword set with no scoring, no stemming and no
    model. It exists so the weather route is reachable end to end, and it is expected
    to be replaced by a real intent classifier once one is available.
    """
    tokens = message.text.lower().split()
    return any(token.strip(_TOKEN_STRIP) in _WEATHER_KEYWORDS for token in tokens)


def _has_eligibility_intent(message: UserMessage) -> bool:
    """Return True if the message matches a provisional eligibility keyword.

    Provisional for the same reason as ``_has_weather_intent``: a fixed keyword set with
    no scoring, no stemming and no model. Matching an intent does not imply an answer is
    available, because eligibility still needs structured facts from the caller.
    """
    tokens = message.text.lower().split()
    return any(token.strip(_TOKEN_STRIP) in _ELIGIBILITY_KEYWORDS for token in tokens)


def _coerce_eligibility_request(value: object) -> EligibilityRequest | None:
    """Return a trustworthy EligibilityRequest, or None when the input cannot supply one.

    Facts are never inferred. Only the four conditions in ``FACT_NAMES`` are read, every
    non-boolean is treated as not supplied, and the request is rebuilt through
    ``EligibilityRequest`` so its own validation still applies. A malformed request is
    reported as absent rather than partially trusted.
    """
    if isinstance(value, EligibilityRequest):
        scheme: object = value.scheme
        supplied = {fact: getattr(value, fact) for fact in FACT_NAMES}
    elif isinstance(value, Mapping):
        scheme = value.get("scheme")
        supplied = {fact: value.get(fact) for fact in FACT_NAMES}
    else:
        return None

    if not isinstance(scheme, str) or not scheme.strip():
        return None

    facts = {
        fact: item if isinstance(item, bool) else None
        for fact, item in supplied.items()
    }
    try:
        return EligibilityRequest(scheme=scheme, **facts)
    except EligibilityError:
        return None


def _resolve_coordinates(message: UserMessage) -> tuple[float, float] | None:
    """Return caller-supplied coordinates, or None when they are unavailable.

    Coordinates are never inferred or defaulted here. There is no geocoding yet, and
    inventing a location would silently report another farmer's weather, so a partial
    or absent pair is reported as unavailable and routed to clarification.
    """
    latitude = message.latitude
    longitude = message.longitude
    if latitude is None or longitude is None:
        return None
    return float(latitude), float(longitude)


def _build_clarification_response(message: UserMessage) -> AssistantResponse:
    """Create a deterministic clarification response."""
    language = message.language or Language.ENGLISH
    text = "Could you please provide more specific details so I can help?"
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.NEEDS_CLARIFICATION,
        citations=(),
    )


def _build_retrieval_response(
    message: UserMessage,
    chunk_count: int,
    batch: CitationBatch | None = None,
) -> AssistantResponse:
    """Create a response noting that retrieval occurred but generation is not implemented.

    Provenance that resolved is attached, so the citations already point at real pages of
    registered documents. When every retrieved chunk was refused, no claim can be
    attributed to anything, so the response abstains instead of citing nothing while
    still sounding answered.
    """
    language = message.language or Language.ENGLISH
    citations: tuple[Citation, ...] = batch.citations if batch is not None else ()
    if chunk_count > 0 and not citations and batch is not None and batch.rejected:
        text = (
            "I could not verify the sources behind these excerpts, "
            "so I am not citing them."
        )
        return AssistantResponse(
            text=text,
            language=language,
            status=ResponseStatus.ABSTAINED,
            citations=(),
        )
    text = (
        "I retrieved relevant excerpts from the available sources, "
        "but answer generation has not yet been implemented."
    )
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=citations,
    )


def _build_eligibility_response(
    message: UserMessage,
    decision: EligibilityDecision | None,
    batch: CitationBatch | None,
) -> AssistantResponse:
    """Render an eligibility outcome and its citations, refusing unciteable verdicts.

    The decision summary is deterministic text from the rule set, not generated prose, so
    it is safe to state directly. A verdict that cannot cite its own evidence is withheld:
    reporting "you are not eligible" without a source would be an unauditable claim about
    a farmer's benefit.
    """
    language = message.language or Language.ENGLISH
    citations: tuple[Citation, ...] = batch.citations if batch is not None else ()

    if decision is None:
        return AssistantResponse(
            text=(
                "I cannot determine eligibility without the required facts. "
                "Please provide the following so I can check them against the official "
                f"guidelines: {', '.join(FACT_NAMES)}."
            ),
            language=language,
            status=ResponseStatus.NEEDS_CLARIFICATION,
            citations=(),
        )

    if decision.status is EligibilityStatus.UNSUPPORTED_SCHEME:
        return AssistantResponse(
            text=decision.summary,
            language=language,
            status=ResponseStatus.ABSTAINED,
            citations=(),
        )

    if decision.status is EligibilityStatus.INSUFFICIENT_INFORMATION:
        return AssistantResponse(
            text=(
                f"{decision.summary} "
                f"Please provide the following: {', '.join(decision.missing_facts)}."
            ),
            language=language,
            status=ResponseStatus.NEEDS_CLARIFICATION,
            citations=(),
        )

    if not citations:
        return AssistantResponse(
            text=(
                "I reached a determination, but I could not verify the official documents "
                "behind it, so I am withholding it rather than stating it without a source."
            ),
            language=language,
            status=ResponseStatus.ABSTAINED,
            citations=(),
        )

    return AssistantResponse(
        text=(
            f"{decision.summary} "
            "This determination is based only on the official documents cited below."
        ),
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=citations,
    )


def _build_weather_location_request(message: UserMessage) -> AssistantResponse:
    """Ask for a location when the request carries no coordinates."""
    language = message.language or Language.ENGLISH
    text = (
        "I can share the weather once I know where you are. "
        "Please provide your latitude and longitude, or your district and village name."
    )
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.NEEDS_CLARIFICATION,
        citations=(),
    )


def _build_weather_unavailable_response(message: UserMessage) -> AssistantResponse:
    """Report that weather data could not be retrieved at all."""
    language = message.language or Language.ENGLISH
    text = "Weather data is currently unavailable. Please try again later."
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.ABSTAINED,
        citations=(),
    )


def _format_measurement(value: float | None, unit: str = "") -> str:
    """Render a measured value, or say plainly that it was not reported."""
    if value is None:
        return "unavailable"
    return f"{value:g}{unit}"


def _build_weather_response(
    message: UserMessage,
    weather: WeatherResponse,
) -> AssistantResponse:
    """Render a weather response from actual WeatherResponse data.

    Only values reported by the weather tool are shown. No agricultural advice is
    generated here, and values the tool did not report are rendered as ``unavailable``
    rather than as zero.
    """
    language = message.language or Language.ENGLISH
    current: WeatherCurrent = weather.current
    text = (
        f"Weather at {weather.latitude:g}, {weather.longitude:g}: "
        f"temperature {_format_measurement(current.temperature_c, ' °C')}, "
        f"precipitation {_format_measurement(current.precipitation_mm, ' mm')}, "
        f"wind speed {_format_measurement(current.wind_speed_mps, ' m/s')}"
    )
    if current.weather_code is not None:
        text += f", weather code {current.weather_code}"
    return AssistantResponse(
        text=text + ".",
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=(),
    )


def build_graph(
    retriever: _RetrieverProtocol,
    weather_client: _WeatherClientProtocol | None = None,
    *,
    citation_resolver: CitationResolver | None = None,
    eligibility_evaluator: _EligibilityEvaluatorProtocol | None = None,
    answer_generator: _AnswerGeneratorProtocol | None = None,
    language_detector: _LanguageDetectorProtocol | None = None,
    speech_to_text: _SpeechToTextProtocol | None = None,
) -> StateGraph:
    """Build the deterministic orchestration graph.

    Collaborators are injected rather than held in graph state, so building and compiling
    the graph performs no I/O. Passing ``weather_client=None`` keeps the weather route
    reachable but answers it without calling a client. Passing ``citation_resolver=None``
    keeps the retrieval and eligibility routes reachable but publishes no citations, which
    is the original behaviour. Passing ``answer_generator=None`` keeps all routes using
    deterministic placeholder responses. When injected, the generator produces grounded
    answers that are then validated against the resolved citation batch.
    Passing ``language_detector=None`` uses a deterministic script-based detector as default.
    Passing ``speech_to_text=None`` keeps the text-only path unchanged; when injected,
    audio input is transcribed before routing.
    """

    evaluator = evaluate if eligibility_evaluator is None else eligibility_evaluator
    generator = answer_generator
    detector = language_detector or DeterministicLanguageDetector()
    stt = speech_to_text

    graph = StateGraph(OrchestrationState)

    def speech_to_text_node(state: OrchestrationState) -> OrchestrationState:
        """Transcribe audio to text if audio input is provided.

        If no audio input is present, passes through unchanged.
        On transcription failure, returns a fallback response with appropriate status.
        """
        if stt is None:
            return state

        audio_data = state.get("audio_data")
        content_type = state.get("audio_content_type")

        if not audio_data:
            return state

        try:
            # Validate audio input
            from kisansathi.voice.audio import validate_audio_input
            validate_audio_input(audio_data, content_type or "audio/wav")

            # Transcribe
            result = stt.transcribe(audio_data, content_type=content_type)

            # Validate transcription
            from kisansathi.voice.audio import validate_transcription_result
            text = validate_transcription_result(result.text)

            # Map Whisper language to supported language
            from kisansathi.voice.audio import map_whisper_language_to_supported
            whisper_lang = map_whisper_language_to_supported(result.language)

            # Determine the language for the transcribed message.
            # Priority: explicit user language (from original message) > Whisper detected > detector > English.
            # If the original message had an explicit language, preserve it.
            original_language = state["message"].language
            whisper_language = result.language
            # Use explicit user language if set, otherwise Whisper detected, otherwise English
            user_language = original_language if original_language is not None else (whisper_language if whisper_language is not None else Language.ENGLISH)
            message = UserMessage(
                text=text,
                language=user_language,
                latitude=state["message"].latitude,
                longitude=state["message"].longitude,
            )

            return {
                **state,
                "message": message,
                # Clear audio data after transcription
                "audio_data": None,
                "audio_content_type": None,
            }

        except InvalidAudioError as e:
            # Invalid audio format -> ABSTAINED
            lang = state["message"].language or Language.ENGLISH
            return {
                **state,
                "response": AssistantResponse(
                    text="I could not process the audio. Please check the format and try again.",
                    language=lang,
                    status=ResponseStatus.ABSTAINED,
                    citations=(),
                ),
                "audio_data": None,
                "audio_content_type": None,
            }
        except EmptyTranscriptError as e:
            # Empty transcript -> NEEDS_CLARIFICATION
            lang = state["message"].language or Language.ENGLISH
            return {
                **state,
                "response": AssistantResponse(
                    text="I could not understand the audio. Please speak clearly and try again.",
                    language=lang,
                    status=ResponseStatus.NEEDS_CLARIFICATION,
                    citations=(),
                ),
                "audio_data": None,
                "audio_content_type": None,
            }
        except TranscriptionError as e:
            # Transcription error: check if it's a service availability issue
            error_msg = str(e).lower()
            lang = state["message"].language or Language.ENGLISH
            if "unavailable" in error_msg or "service" in error_msg:
                # Service unavailable -> ABSTAINED
                return {
                    **state,
                    "response": AssistantResponse(
                        text="Speech recognition service is unavailable. Please try again later.",
                        language=lang,
                        status=ResponseStatus.ABSTAINED,
                        citations=(),
                    ),
                    "audio_data": None,
                    "audio_content_type": None,
                }
            # Other transcription errors (timeout, model error) -> NEEDS_CLARIFICATION
            return {
                **state,
                "response": AssistantResponse(
                    text="I could not understand the audio. Please speak clearly and try again.",
                    language=lang,
                    status=ResponseStatus.NEEDS_CLARIFICATION,
                    citations=(),
                ),
                "audio_data": None,
                "audio_content_type": None,
            }
        except SpeechToTextError as e:
            # Other speech-to-text errors -> ABSTAINED
            lang = state["message"].language or Language.ENGLISH
            return {
                **state,
                "response": AssistantResponse(
                    text="Speech recognition service is unavailable. Please try again later.",
                    language=lang,
                    status=ResponseStatus.ABSTAINED,
                    citations=(),
                ),
                "audio_data": None,
                "audio_content_type": None,
            }

    def route_request(state: OrchestrationState) -> OrchestrationState:
        # If a terminal response (ABSTAINED/NEEDS_CLARIFICATION) is already set by a previous node
        # (e.g., speech_to_text failure), preserve it and skip routing to go directly to finalize.
        existing_response = state.get("response")
        if existing_response is not None and existing_response.status in (
            ResponseStatus.ABSTAINED,
            ResponseStatus.NEEDS_CLARIFICATION,
        ):
            return {**state, "route": Route.FINALIZE}

        message = state["message"]
        if _is_underspecified(message):
            return {**state, "route": Route.CLARIFY}
        if _has_weather_intent(message):
            return {**state, "route": Route.WEATHER}
        if _has_eligibility_intent(message):
            return {**state, "route": Route.ELIGIBILITY}
        return {**state, "route": Route.RETRIEVAL}

    def detect_language(state: OrchestrationState) -> OrchestrationState:
        """Detect the language of the user message if not explicitly provided.

        Explicit user-requested language (message.language) takes priority.
        If not provided, run the detector on the message text.
        On detection failure or ambiguity, fall back to None (which resolves to English later).
        """
        message = state["message"]
        if message.language is not None:
            return {**state, "detected_language": None}
        try:
            detected = detector.detect(message.text)
        except AmbiguousLanguageError:
            detected = None
        except DetectionError:
            detected = None
        return {**state, "detected_language": detected}

    def resolve_language(message: UserMessage, detected: Language | None) -> Language:
        """Resolve the final language for response generation.

        Priority: explicit user language > detected language > English default.
        """
        if message.language is not None:
            return message.language
        if detected is not None:
            return detected
        return Language.ENGLISH

    def retrieve(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        results = retriever.retrieve(message.text, top_k=5)
        return {**state, "retrieved_chunks": tuple(results)}

    def eligibility(state: OrchestrationState) -> OrchestrationState:
        """Evaluate caller-supplied facts, or ask for them when none were supplied.

        Facts come from ``eligibility_request`` only. Nothing here parses the message
        text, because guessing a farmer's circumstances would silently produce a wrong
        verdict about a real benefit.
        """
        request = _coerce_eligibility_request(state.get("eligibility_request"))
        if request is None:
            return {
                **state,
                "eligibility_decision": None,
                "retrieved_chunks": (),
            }
        try:
            decision = evaluator(request)
        except EligibilityError:
            decision = None
        return {**state, "eligibility_decision": decision, "retrieved_chunks": ()}

    def resolve_citations(state: OrchestrationState) -> OrchestrationState:
        """Join resolved provenance into citations, or record that nothing resolved.

        Both evidence routes converge here so citation rules live in one place and are
        never duplicated per route. Only retrieval payloads and eligibility evidence are
        handed to the resolver, never source text.

        An empty batch with no rejections means citation resolution is not configured, so
        the routes keep their original un-cited behaviour. Anything the resolver refused,
        including a resolver that failed outright, is recorded as a rejection so the
        response abstains instead of claiming to have sourced something.
        """
        if citation_resolver is None:
            return {**state, "citations": CitationBatch()}

        route = state.get("route")
        try:
            if route == Route.RETRIEVAL:
                chunks = state.get("retrieved_chunks") or ()
                batch = citation_resolver.resolve_payloads(
                    result.payload for result in chunks
                )
            elif route == Route.ELIGIBILITY:
                decision = state.get("eligibility_decision")
                if decision is None:
                    batch = CitationBatch()
                else:
                    batch = citation_resolver.resolve_evidence_batch(decision.evidence)
            else:
                batch = CitationBatch()
        except CitationError as error:
            batch = CitationBatch(
                rejected=(
                    RejectedCitation(
                        source_id=_UNRESOLVED_SOURCE,
                        chunk_id=None,
                        reason=f"citation resolution failed: {error}",
                    ),
                )
            )
        return {**state, "citations": batch}

    def generate_answer(state: OrchestrationState) -> OrchestrationState:
        """Generate a grounded answer using the injected AnswerGenerator.

        If no generator is configured, passes through with generated_answer=None.
        Failures are caught and converted to a safe fallback answer with appropriate
        status so the graph never crashes and never fabricates citations.
        """
        if generator is None:
            return {**state, "generated_answer": None}

        from kisansathi.generation.models import GenerationContext

        # Resolve the language for this request
        resolved_language = resolve_language(
            state["message"], state.get("detected_language")
        )

        context = GenerationContext(
            message=state["message"],
            citations=state.get("citations") or CitationBatch(),
            eligibility_decision=state.get("eligibility_decision"),
            weather=state.get("weather"),
        )

        try:
            answer = generator.generate(context)
            # Override the answer's language with the resolved language
            if answer is not None:
                answer = GeneratedAnswer(
                    text=answer.text,
                    citation_ids=answer.citation_ids,
                    status=answer.status,
                    language=resolved_language,
                )
        except LLMError:
            # Infrastructure failure (network, timeout, provider error) -> ABSTAINED
            return {
                **state,
                "generated_answer": GeneratedAnswer(
                    text="I could not generate an answer due to a service error. Please try again later.",
                    citation_ids=(),
                    status=ResponseStatus.ABSTAINED,
                    language=resolved_language,
                ),
            }
        except (MalformedOutputError, GroundingError):
            # Model misbehaved (bad format, hallucinated citation) -> NEEDS_CLARIFICATION
            return {
                **state,
                "generated_answer": GeneratedAnswer(
                    text="I could not produce a reliable answer from the available sources.",
                    citation_ids=(),
                    status=ResponseStatus.NEEDS_CLARIFICATION,
                    language=resolved_language,
                ),
            }
        except GenerationError:
            # Any other generation error -> NEEDS_CLARIFICATION
            return {
                **state,
                "generated_answer": GeneratedAnswer(
                    text="I could not produce a reliable answer from the available sources.",
                    citation_ids=(),
                    status=ResponseStatus.NEEDS_CLARIFICATION,
                    language=resolved_language,
                ),
            }

        return {**state, "generated_answer": answer}

    def validate_and_attach_citations(state: OrchestrationState) -> OrchestrationState:
        """Post-generation citation validation using the existing resolver gate.

        If the generator produced an answer with citation IDs, validate them against
        the resolved CitationBatch. On validation failure, fall back to a safe answer.
        """
        answer = state.get("generated_answer")
        batch = state.get("citations") or CitationBatch()

        if answer is None or not answer.citation_ids:
            return {**state, "validated_citations": ()}

        try:
            validated = validate_referenced_citations(answer.citation_ids, batch)
        except Exception:
            # Model cited an ID that wasn't in the batch -> NEEDS_CLARIFICATION
            return {
                **state,
                "generated_answer": GeneratedAnswer(
                    text="I could not verify the sources for my answer.",
                    citation_ids=(),
                    status=ResponseStatus.NEEDS_CLARIFICATION,
                    language=answer.language,
                ),
                "validated_citations": (),
            }

        return {**state, "validated_citations": validated}

    def weather(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        resolved_language = resolve_language(message, state.get("detected_language"))
        coordinates = _resolve_coordinates(message)
        if coordinates is None:
            return {
                **state,
                "response": _build_weather_location_request(message),
                "weather": None,
                "retrieved_chunks": (),
            }
        if weather_client is None:
            return {
                **state,
                "response": _build_weather_unavailable_response(message),
                "weather": None,
                "retrieved_chunks": (),
            }
        latitude, longitude = coordinates
        try:
            result = weather_client.get_forecast(
                WeatherRequest(latitude=latitude, longitude=longitude)
            )
        except (WeatherError, TypeError, ValueError):
            return {
                **state,
                "response": _build_weather_unavailable_response(message),
                "weather": None,
                "retrieved_chunks": (),
            }
        return {**state, "weather": result, "retrieved_chunks": ()}

    def clarify(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        response = _build_clarification_response(message)
        return {**state, "response": response, "retrieved_chunks": ()}

    def finalize_response(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        route = state.get("route")
        if route == Route.FINALIZE:
            # Terminal response already set by error handler; pass it through
            existing_response = state.get("response")
            if existing_response is None:
                existing_response = _build_clarification_response(message)
            return {**state, "response": existing_response}
        if route == Route.CLARIFY:
            existing_response = state.get("response")
            if existing_response is None:
                existing_response = _build_clarification_response(message)
            return {**state, "response": existing_response}
        if route == Route.WEATHER:
            weather_result = state.get("weather")
            if weather_result is None:
                existing_response = state.get("response")
                if existing_response is None:
                    existing_response = _build_weather_location_request(message)
                return {**state, "response": existing_response}
            return {**state, "response": _build_weather_response(message, weather_result)}
        if route == Route.ELIGIBILITY:
            # Use generated answer if available, otherwise fall back to deterministic response
            answer = state.get("generated_answer")
            validated = state.get("validated_citations") or ()
            if answer is not None:
                return {
                    **state,
                    "response": AssistantResponse(
                        text=answer.text,
                        language=answer.language,
                        status=answer.status,
                        citations=validated,
                    ),
                }
            # Fallback to deterministic response (uses citations from resolver)
            resolved_language = resolve_language(message, state.get("detected_language"))
            response = _build_eligibility_response(
                message,
                state.get("eligibility_decision"),
                state.get("citations"),
            )
            # Override language in deterministic response
            response = AssistantResponse(
                text=response.text,
                language=resolved_language,
                status=response.status,
                citations=response.citations,
            )
            return {**state, "response": response}
        # Retrieval route
        answer = state.get("generated_answer")
        validated = state.get("validated_citations") or ()
        batch = state.get("citations")

        # If a generator was configured, generated_answer will be set (even on fallback).
        # If no generator was configured, generated_answer is None and we use the raw batch.
        if answer is not None:
            return {
                **state,
                "response": AssistantResponse(
                    text=answer.text,
                    language=answer.language,
                    status=answer.status,
                    citations=validated,
                ),
            }
        # No generator configured: use the raw citation batch for the placeholder
        resolved_language = resolve_language(message, state.get("detected_language"))
        retrieved = state.get("retrieved_chunks") or ()
        response = _build_retrieval_response(
            message,
            len(retrieved),
            batch if batch else CitationBatch(),
        )
        # Override language in deterministic response
        response = AssistantResponse(
            text=response.text,
            language=resolved_language,
            status=response.status,
            citations=response.citations,
        )
        return {**state, "response": response}

    graph.add_node("speech_to_text", speech_to_text_node)
    graph.add_node("route_request", route_request)
    graph.add_node("detect_language", detect_language)
    graph.add_node("retrieve", retrieve)
    graph.add_node("eligibility", eligibility)
    graph.add_node("resolve_citations", resolve_citations)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_and_attach_citations", validate_and_attach_citations)
    graph.add_node("weather", weather)
    graph.add_node("clarify", clarify)
    graph.add_node("finalize_response", finalize_response)

    graph.add_edge("__start__", "speech_to_text")
    graph.add_edge("speech_to_text", "route_request")
    # route_request determines the route; if a terminal response is already set, go directly to finalize
    # For retrieval/eligibility, run language detection first; weather/clarify/finalize bypass it.
    graph.add_conditional_edges(
        "route_request",
        lambda state: state["route"],
        {
            Route.RETRIEVAL: "detect_language",
            Route.ELIGIBILITY: "detect_language",
            Route.WEATHER: "weather",
            Route.CLARIFY: "clarify",
            Route.FINALIZE: "finalize_response",
        },
    )
    # Language detection runs before retrieval/eligibility
    graph.add_conditional_edges(
        "detect_language",
        lambda state: state["route"],
        _ROUTE_NODES,
    )
    # Retrieval and eligibility paths go through citation resolution and generation
    graph.add_edge("retrieve", "resolve_citations")
    graph.add_edge("eligibility", "resolve_citations")
    graph.add_edge("resolve_citations", "generate_answer")
    graph.add_edge("generate_answer", "validate_and_attach_citations")
    graph.add_edge("validate_and_attach_citations", "finalize_response")
    # Weather and clarify paths bypass citation resolution and generation (no document evidence)
    graph.add_edge("weather", "finalize_response")
    graph.add_edge("clarify", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph
