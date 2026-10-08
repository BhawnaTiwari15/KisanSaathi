"""Pure presentation helpers for the KisanSaathi Streamlit UI.

This module must not import Streamlit. Everything here is deterministic and
unit-testable, so the Streamlit layer stays a thin shell over the application
boundaries: these helpers shape input into graph state and shape graph output
into display strings, and they never decide what an answer means.

Business rules (retrieval, eligibility, citations, weather, guardrails) are
never duplicated here; this module only formats, validates presence, and maps
display choices onto the existing typed contracts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from kisansathi.domain.schemas import Citation, Language, ResponseStatus, UserMessage
from kisansathi.eligibility.models import FACT_NAMES, EligibilityRequest

logger = logging.getLogger(__name__)

SAFE_ERROR_MESSAGE = (
    "Something went wrong while processing your request. "
    "Please try again or rephrase your question."
)

LANGUAGE_LABELS: dict[Language, str] = {
    Language.ENGLISH: "English (en)",
    Language.HINDI: "Hindi (hi)",
    Language.KANNADA: "Kannada (kn)",
    Language.TELUGU: "Telugu (te)",
}

STATUS_LABELS: dict[ResponseStatus, str] = {
    ResponseStatus.ANSWERED: "Answered",
    ResponseStatus.NEEDS_CLARIFICATION: "Needs clarification",
    ResponseStatus.ABSTAINED: "Abstained",
}

_STATUS_FOOTERS: dict[ResponseStatus, str] = {
    ResponseStatus.ABSTAINED: (
        "I cannot provide a reliable answer because the system could not "
        "establish sufficient grounded evidence. Please adjust your request "
        "or provide additional information."
    ),
    ResponseStatus.NEEDS_CLARIFICATION: (
        "I need more information to help you. Please provide the missing "
        "details and try again."
    ),
}

FACT_LABELS: dict[str, str] = {
    "landholding_in_own_name": "Landholding in own name",
    "land_is_cultivable": "Land is cultivable",
    "land_used_for_non_agricultural_purpose": "Land used for non-agricultural purpose",
    "family_member_paid_income_tax_last_assessment_year": (
        "Family member paid income tax last assessment year"
    ),
}

FACT_SELECTIONS: tuple[str, ...] = ("Not sure", "Yes", "No")

_SELECTION_TO_BOOL: dict[str, bool | None] = {
    "Yes": True,
    "No": False,
    "Not sure": None,
}

_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png"})


def validate_text(text: str) -> bool:
    """Return True when the farmer's question contains non-whitespace text."""
    return isinstance(text, str) and bool(text.strip())


def status_label(status: ResponseStatus) -> str:
    """Return the short display label for a response status."""
    return STATUS_LABELS.get(status, "Status")


def status_footer(status: ResponseStatus) -> str | None:
    """Return the status-specific footer caption, or None when there is none."""
    return _STATUS_FOOTERS.get(status)


def render_citation_text(citation: Citation) -> str:
    """Render one Citation into a single display line.

    Every field comes verbatim from the citation itself; nothing is invented
    when a field is absent.
    """
    parts: list[str] = []
    if citation.title:
        parts.append(citation.title)
    if citation.page_number is not None:
        if citation.page_end is not None and citation.page_end != citation.page_number:
            parts.append(f"pp. {citation.page_number}-{citation.page_end}")
        else:
            parts.append(f"p. {citation.page_number}")
    authority = citation.issuing_authority
    if authority is not None and authority.strip():
        parts.append(f"*{authority.strip()}*")
    if citation.url and citation.url.strip():
        parts.append(f"[{citation.url}]({citation.url})")
    return "  ".join(parts) if parts else "Source"


def parse_location(latitude_text: str, longitude_text: str) -> tuple[float, float] | None:
    """Parse explicit coordinate inputs, or return None when they are unusable.

    Blank, non-numeric, out-of-range, and non-finite values all yield None;
    the UI never guesses or defaults a location.
    """
    if not isinstance(latitude_text, str) or not isinstance(longitude_text, str):
        return None
    latitude_raw = latitude_text.strip()
    longitude_raw = longitude_text.strip()
    if not latitude_raw or not longitude_raw:
        return None
    try:
        latitude = float(latitude_raw)
        longitude = float(longitude_raw)
    except ValueError:
        return None
    if not (-90.0 <= latitude <= 90.0):
        return None
    if not (-180.0 <= longitude <= 180.0):
        return None
    return latitude, longitude


def fact_selection_to_bool(selection: str) -> bool | None:
    """Map a display selection onto a tri-state fact value.

    Unknown selections are treated as not supplied rather than as a value.
    """
    if not isinstance(selection, str):
        return None
    return _SELECTION_TO_BOOL.get(selection)


def build_eligibility_request(
    selections: Mapping[str, str],
    *,
    scheme: str = "pm-kisan",
) -> EligibilityRequest | None:
    """Build an EligibilityRequest from display selections, or None if empty.

    Returns None when every fact is "Not sure" (nothing was supplied), so the
    eligibility engine receives either explicit facts or no request at all.
    """
    values = {
        fact: fact_selection_to_bool(selections.get(fact, "Not sure")) for fact in FACT_NAMES
    }
    if all(value is None for value in values.values()):
        return None
    return EligibilityRequest(scheme=scheme, **values)


def image_content_type_for(mime_type: object) -> str:
    """Return a content type the vision boundary accepts, defaulting to JPEG."""
    if isinstance(mime_type, str):
        normalized = mime_type.split(";", 1)[0].strip().lower()
        if normalized == "image/jpg":
            normalized = "image/jpeg"
        if normalized in _IMAGE_CONTENT_TYPES:
            return normalized
    return "image/jpeg"


def build_request_state(
    question: str,
    language: Language,
    *,
    image_bytes: bytes | None = None,
    image_content_type: str | None = None,
    image_filename: str | None = None,
    audio_bytes: bytes | None = None,
    audio_content_type: str | None = None,
    location: tuple[float, float] | None = None,
    eligibility: EligibilityRequest | None = None,
) -> dict[str, Any]:
    """Shape UI inputs into an OrchestrationState for ``graph.invoke``.

    Coordinates are passed through only when the user supplied them, and
    modality content types are dropped when their bytes are absent, so the
    graph never sees partial modality data.
    """
    latitude, longitude = location if location is not None else (None, None)
    message = UserMessage(
        text=question,
        language=language,
        latitude=latitude,
        longitude=longitude,
    )
    return {
        "message": message,
        "route": "",
        "retrieved_chunks": (),
        "response": None,
        "weather": None,
        "citations": None,
        "eligibility_decision": None,
        "eligibility_request": eligibility,
        "generated_answer": None,
        "validated_citations": (),
        "detected_language": None,
        "audio_data": audio_bytes,
        "audio_content_type": audio_content_type if audio_bytes is not None else None,
        "image_data": image_bytes,
        "image_content_type": image_content_type if image_bytes is not None else None,
        "image_filename": image_filename if image_bytes is not None else None,
        "vision_result": None,
    }


def invoke_graph_safely(graph: Any, state: Mapping[str, Any]) -> tuple[Any, str | None]:
    """Invoke the graph boundary and never let an error reach the UI.

    Returns ``(response, None)`` on success, or ``(None, SAFE_ERROR_MESSAGE)``
    when the graph raises, returns a non-mapping result, or finishes without a
    response. Failures are logged server-side; the UI only sees the safe text.
    """
    try:
        result = graph.invoke(state)
    except Exception:
        logger.exception("graph invocation failed")
        return None, SAFE_ERROR_MESSAGE
    if not isinstance(result, Mapping):
        logger.error("graph returned unexpected result type %s", type(result).__name__)
        return None, SAFE_ERROR_MESSAGE
    response = result.get("response")
    if response is None:
        logger.error("graph finished without a response")
        return None, SAFE_ERROR_MESSAGE
    return response, None
