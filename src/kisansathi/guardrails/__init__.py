"""Final safety guardrails: evidence sufficiency, citation gates, and abstention policy.

This module consolidates all abstention/clarification logic into a single
deterministic layer that runs after citation/generation processing and before
the final response is emitted.
"""

from enum import StrEnum
from typing import Any

from kisansathi.citations.models import CitationBatch
from kisansathi.domain.schemas import AssistantResponse, Language, ResponseStatus, Route
from kisansathi.eligibility.models import EligibilityDecision, EligibilityStatus
from kisansathi.generation.models import GeneratedAnswer
from kisansathi.orchestration.graph import Route as GraphRoute  # for type checking if needed
from kisansathi.vision.models import VisionResult, VisionStatus
from kisansathi.weather.models import WeatherResponse


class GuardrailCategory(StrEnum):
    """Structured internal error categories for observability."""

    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_FAILURE = "retrieval_failure"
    CITATION_FAILURE = "citation_failure"
    CITATION_ALL_REJECTED = "citation_all_rejected"
    CITATION_INVENTED_ID = "citation_invented_id"
    GENERATION_FAILURE = "generation_failure"
    GENERATION_MALFORMED = "generation_malformed"
    GENERATION_GROUNDING_ERROR = "generation_grounding_error"
    WEATHER_UNAVAILABLE = "weather_unavailable"
    WEATHER_INVALID_COORDS = "weather_invalid_coords"
    VISION_UNAVAILABLE = "vision_unavailable"
    VISION_INVALID_INPUT = "vision_invalid_input"
    VISION_LOW_CONFIDENCE = "vision_low_confidence"
    ELIGIBILITY_INSUFFICIENT_FACTS = "eligibility_insufficient_facts"
    ELIGIBILITY_UNSUPPORTED_SCHEME = "eligibility_unsupported_scheme"
    ELIGIBILITY_CONTRADICTORY = "eligibility_contradictory"
    UNDERSPECIFIED_QUERY = "underspecified_query"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    MISSING_REQUIRED_CITATIONS = "missing_required_citations"
    EMPTY_ANSWER = "empty_answer"


def _log_guardrail(category: GuardrailCategory, route: str, language: Language | str, details: str = "") -> None:
    """Log a guardrail decision for observability.
    
    Uses stdlib logging with structured data. No secrets, no raw user content.
    """
    import logging
    logger = logging.getLogger("kisansathi.guardrails")
    language_value = getattr(language, "value", language)
    logger.info({
        "category": category.value,
        "route": route,
        "language": language_value,
        "details": details,
    })


def _build_clarification_response(message_text: str, language: Language, reason: str) -> AssistantResponse:
    """Build a NEEDS_CLARIFICATION response with a helpful explanation."""
    text = (
        f"I need more information to help you. {reason} "
        "Please provide the missing details and try again."
    )
    return AssistantResponse(text=text, language=language, status=ResponseStatus.NEEDS_CLARIFICATION, citations=())


def _build_abstained_response(message_text: str, language: Language, reason: str) -> AssistantResponse:
    """Build an ABSTAINED response with a clear explanation."""
    text = f"I cannot provide a reliable answer because {reason}."
    return AssistantResponse(text=text, language=language, status=ResponseStatus.ABSTAINED, citations=())


def _resolve_language(message_language: Language | None, detected_language: Language | None) -> Language:
    """Resolve final response language: explicit > detected > English default."""
    if message_language is not None:
        return message_language
    if detected_language is not None:
        return detected_language
    return Language.ENGLISH


def apply_final_guardrails(
    state: dict[str, Any],
    route: str | None = None,
) -> dict[str, Any]:
    """Apply final safety/abstention policy before response is finalized.
    
    This is the single guardrails node that consolidates all abstention logic.
    It runs after citation resolution, generation, and citation validation.
    
    Args:
        state: The orchestration graph state dictionary.
        route: The route taken (retrieval, eligibility, weather, clarify, etc.)
        
    Returns:
        Updated state with a finalized `response` field.
    """
    message = state.get("message")
    if message is None:
        # Should not happen, but fail closed
        return {
            **state,
            "response": AssistantResponse(
                text="Internal error: no message in state.",
                language=Language.ENGLISH,
                status=ResponseStatus.ABSTAINED,
                citations=(),
            ),
        }

    language = _resolve_language(message.language, state.get("detected_language"))
    route_enum = Route(route) if route else Route.FINALIZE

    # If a terminal response was already set by an earlier node (e.g., vision/speech failure),
    # preserve it and skip further processing.
    existing_response = state.get("response")
    if existing_response is not None and existing_response.status in (
        ResponseStatus.ABSTAINED,
        ResponseStatus.NEEDS_CLARIFICATION,
    ):
        return {**state, "response": existing_response}

    generated_answer = state.get("generated_answer")
    validated_citations = state.get("validated_citations") or ()
    citation_batch = state.get("citations") or CitationBatch()
    eligibility_decision = state.get("eligibility_decision")
    vision_result = state.get("vision_result")
    weather = state.get("weather")
    retrieved_chunks = state.get("retrieved_chunks") or ()

    # --- Route-specific checks ---

    if route_enum == Route.WEATHER:
        return _handle_weather_route(state, language, weather, message.text)

    if route_enum == Route.CLARIFY:
        return _handle_clarify_route(state, language, message.text)

    if route_enum == Route.ELIGIBILITY:
        return _handle_eligibility_route(
            state, language, eligibility_decision, citation_batch, vision_result, message.text
        )

    if route_enum == Route.RETRIEVAL:
        return _handle_retrieval_route(
            state, language, generated_answer, validated_citations, citation_batch,
            retrieved_chunks, vision_result, message.text
        )

    # Fallback for FINALIZE or unknown route
    return _handle_fallback_route(state, language, generated_answer, validated_citations, citation_batch, message.text)


def _handle_weather_route(
    state: dict[str, Any],
    language: Language,
    weather: WeatherResponse | None,
    query: str,
) -> dict[str, Any]:
    """Handle weather route: weather is independent of citation layer."""
    from kisansathi.orchestration.graph import _build_weather_response, _build_weather_unavailable_response, _build_weather_location_request
    from kisansathi.domain.schemas import UserMessage

    # Reconstruct message for response builders
    msg = UserMessage(text=query, language=language)

    if weather is None:
        _log_guardrail(GuardrailCategory.WEATHER_UNAVAILABLE, "weather", language, "weather data unavailable")
        return {**state, "response": _build_weather_unavailable_response(msg)}

    return {**state, "response": _build_weather_response(msg, weather)}


def _handle_clarify_route(
    state: dict[str, Any],
    language: Language,
    query: str,
) -> dict[str, Any]:
    """Handle clarify route: always NEEDS_CLARIFICATION."""
    from kisansathi.orchestration.graph import _build_clarification_response
    from kisansathi.domain.schemas import UserMessage

    msg = UserMessage(text=query, language=language)
    return {**state, "response": _build_clarification_response(msg)}


def _handle_eligibility_route(
    state: dict[str, Any],
    language: Language,
    eligibility_decision: EligibilityDecision | None,
    citation_batch: CitationBatch,
    vision_result: VisionResult | None,
    query: str,
) -> dict[str, Any]:
    """Handle eligibility route with deterministic eligibility + citation gates."""
    from kisansathi.orchestration.graph import _build_eligibility_response

    # If LLM generated an answer, use it with validated citations
    generated_answer = state.get("generated_answer")
    validated_citations = state.get("validated_citations") or ()

    if generated_answer is not None:
        # LLM path: generated answer already has status and citations validated
        response = AssistantResponse(
            text=generated_answer.text,
            language=generated_answer.language,
            status=generated_answer.status,
            citations=validated_citations,
        )
        # Override language to match resolved language
        response = AssistantResponse(
            text=response.text,
            language=language,
            status=response.status,
            citations=response.citations,
        )
        return {**state, "response": response}

    # Deterministic fallback path
    from kisansathi.domain.schemas import UserMessage
    msg = UserMessage(text=query, language=language)
    response = _build_eligibility_response(
        message=msg,
        decision=eligibility_decision,
        batch=citation_batch,
        vision_result=vision_result,
    )
    return {**state, "response": response}


def _handle_retrieval_route(
    state: dict[str, Any],
    language: Language,
    generated_answer: GeneratedAnswer | None,
    validated_citations: tuple,
    citation_batch: CitationBatch,
    retrieved_chunks: tuple,
    vision_result: VisionResult | None,
    query: str,
) -> dict[str, Any]:
    """Handle retrieval route with evidence sufficiency and citation gates."""

    # 1. Evidence sufficiency: zero retrieved chunks where evidence is required
    if not retrieved_chunks:
        _log_guardrail(GuardrailCategory.RETRIEVAL_EMPTY, "retrieval", language, "zero chunks retrieved")
        return {
            **state,
            "response": _build_abstained_response(
                query, language, "I could not find relevant information in the official documents"
            ),
        }

    # 2. Citation resolution: all citations refused
    if not citation_batch.citations and citation_batch.rejected:
        _log_guardrail(GuardrailCategory.CITATION_ALL_REJECTED, "retrieval", language, "all citations refused")
        return {
            **state,
            "response": _build_abstained_response(
                query, language, "I could not verify the sources behind the retrieved excerpts"
            ),
        }

    # 3. If LLM generated an answer, apply generation + citation gates
    if generated_answer is not None:
        return _handle_generated_answer(
            state, language, generated_answer, validated_citations, citation_batch, query
        )

    # 4. Deterministic placeholder path (no generator configured)
    from kisansathi.orchestration.graph import _build_retrieval_response
    from kisansathi.domain.schemas import UserMessage
    msg = UserMessage(text=query, language=language)
    response = _build_retrieval_response(
        message=msg,
        chunk_count=len(retrieved_chunks),
        batch=citation_batch,
        vision_result=vision_result,
    )
    # Override language
    response = AssistantResponse(
        text=response.text,
        language=language,
        status=response.status,
        citations=response.citations,
    )
    return {**state, "response": response}


def _handle_generated_answer(
    state: dict[str, Any],
    language: Language,
    generated_answer: GeneratedAnswer,
    validated_citations: tuple,
    citation_batch: CitationBatch,
    query: str,
) -> dict[str, Any]:
    """Apply gates to LLM-generated answer."""

    # Gate 1: Empty answer text
    if not generated_answer.text.strip():
        _log_guardrail(GuardrailCategory.EMPTY_ANSWER, "retrieval", language, "empty generated answer")
        return {
            **state,
            "response": _build_abstained_response(query, language, "the generated answer was empty"),
        }

    # Gate 2: Generation status from generator (LLMError -> ABSTAINED, Malformed/Grounding -> NEEDS_CLARIFICATION)
    if generated_answer.status == ResponseStatus.ABSTAINED:
        # LLMError or provider failure
        _log_guardrail(GuardrailCategory.GENERATION_FAILURE, "retrieval", language, "generation failed")
        return {
            **state,
            "response": AssistantResponse(
                text=generated_answer.text,
                language=language,
                status=ResponseStatus.ABSTAINED,
                citations=(),
            ),
        }

    if generated_answer.status == ResponseStatus.NEEDS_CLARIFICATION:
        # MalformedOutputError or GroundingError
        _log_guardrail(GuardrailCategory.GENERATION_MALFORMED, "retrieval", language, "malformed or ungrounded output")
        return {
            **state,
            "response": AssistantResponse(
                text=generated_answer.text,
                language=language,
                status=ResponseStatus.NEEDS_CLARIFICATION,
                citations=(),
            ),
        }

    # Gate 3: Citation validation - generated answer cites IDs not in validated_citations
    if generated_answer.citation_ids:
        validated_chunk_ids = {c.chunk_id for c in validated_citations if c.chunk_id}
        invented_ids = set(generated_answer.citation_ids) - validated_chunk_ids
        if invented_ids:
            _log_guardrail(GuardrailCategory.CITATION_INVENTED_ID, "retrieval", language, f"invented citation IDs: {invented_ids}")
            return {
                **state,
                "response": _build_abstained_response(query, language, "I could not verify the sources for my answer"),
            }

    # Gate 4: ANSWERED requires at least one validated citation
    if not validated_citations:
        _log_guardrail(GuardrailCategory.CITATION_FAILURE, "retrieval", language, "no validated citations for answered response")
        return {
            **state,
            "response": _build_abstained_response(query, language, "I could not verify the sources for my answer"),
        }

    # All gates passed - return the generated answer with validated citations
    response = AssistantResponse(
        text=generated_answer.text,
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=validated_citations,
    )
    return {**state, "response": response}


def _handle_fallback_route(
    state: dict[str, Any],
    language: Language,
    generated_answer: GeneratedAnswer | None,
    validated_citations: tuple,
    citation_batch: CitationBatch,
    query: str,
) -> dict[str, Any]:
    """Fallback for unknown routes or FINALIZE without pre-existing response."""
    if generated_answer is not None:
        return _handle_generated_answer(state, language, generated_answer, validated_citations, citation_batch, query)

    # No generator, no specific route - clarify
    _log_guardrail(GuardrailCategory.UNDERSPECIFIED_QUERY, "fallback", language, "no generator and no specific route")
    return {
        **state,
        "response": _build_clarification_response(query, language, "I need more specific details to help you"),
    }