"""Prompt construction and response parsing for answer generation.

All functions are pure: no I/O, no external dependencies, fully testable.
"""

import re
from kisansathi.citations.models import CitationBatch
from kisansathi.domain.schemas import Language, ResponseStatus, UserMessage
from kisansathi.eligibility.models import EligibilityDecision
from kisansathi.generation.models import GeneratedAnswer, GroundingError, MalformedOutputError
from kisansathi.weather.models import WeatherResponse
from kisansathi.vision.models import VisionResult, VisionStatus

# Maximum characters of evidence to include in the prompt (protects against context
# overflow and limits the attack surface for prompt injection via retrieved text).
_MAX_CONTEXT_CHARS = 8000

# Maximum length of a verbatim excerpt from a retrieved chunk.
_MAX_EXCERPT_CHARS = 300

# Control characters that could be used for prompt injection.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """Remove control characters and limit length to mitigate prompt injection."""
    if not text:
        return ""
    cleaned = _CONTROL_CHARS.sub("", text)
    return cleaned[:_MAX_EXCERPT_CHARS]


def _format_citations(batch: CitationBatch) -> str:
    """Format citations as a numbered list for the prompt.

    Only metadata and a short verbatim excerpt are included — never the full
    chunk text. The excerpt is sanitized and truncated.
    """
    if not batch.citations:
        return "No evidence available."

    lines = []
    for i, citation in enumerate(batch.citations, 1):
        page_info = (
            f"page {citation.page_number}"
            if citation.page_end is None or citation.page_end == citation.page_number
            else f"pages {citation.page_number}-{citation.page_end}"
        )
        safe_title = _sanitize_text(citation.title)
        safe_authority = _sanitize_text(citation.issuing_authority)
        lines.append(
            f"[{i}] {citation.chunk_id} ({page_info})\n"
            f"    Title: {safe_title}\n"
            f"    Authority: {safe_authority}\n"
            f"    Excerpt: {safe_title}"
        )
    return "\n\n".join(lines)


def _format_eligibility(decision: EligibilityDecision | None) -> str:
    if decision is None:
        return ""
    if decision.status.value == "unsupported_scheme":
        return f"\nELIGIBILITY: {decision.summary}"
    if decision.status.value == "insufficient_information":
        missing = ", ".join(decision.missing_facts) if decision.missing_facts else "unknown"
        return f"\nELIGIBILITY: Insufficient information. Missing facts: {missing}."
    # eligible or ineligible
    evidence_summary = ", ".join(e.chunk_id for e in decision.evidence) if decision.evidence else "none"
    return (
        f"\nELIGIBILITY: {decision.summary} "
        f"(status: {decision.status.value}, evidence: {evidence_summary})"
    )


def _format_weather(weather: WeatherResponse | None) -> str:
    if weather is None:
        return ""
    current = weather.current
    parts = []
    if current.temperature_c is not None:
        parts.append(f"temperature {current.temperature_c}°C")
    if current.precipitation_mm is not None:
        parts.append(f"precipitation {current.precipitation_mm} mm")
    if current.wind_speed_mps is not None:
        parts.append(f"wind {current.wind_speed_mps} m/s")
    if current.weather_code is not None:
        parts.append(f"weather code {current.weather_code}")
    weather_str = ", ".join(parts) if parts else "no data"
    return f"\nWEATHER: {weather_str} (at {weather.latitude}, {weather.longitude})"


def _format_vision(vision: VisionResult | None) -> str:
    if vision is None:
        return ""
    if vision.status != VisionStatus.SUCCESS:
        return f"\nVISION: {vision.status.value.replace('_', ' ').title()}."
    
    if not vision.observations:
        return "\nVISION: No usable visual information."
    
    lines = ["\nVISUAL OBSERVATIONS:"]
    for i, obs in enumerate(vision.observations, 1):
        conf_pct = int(obs.confidence * 100)
        lines.append(
            f"[{i}] {obs.label} ({obs.category}, {conf_pct}% confidence): {obs.description}"
        )
    lines.append(
        f"\nNOTE: Visual observations are uncertain (overall confidence: {vision.confidence_overall:.0%}). "
        "Do not treat as definitive diagnosis. Use cautious language: 'appears to be', 'consistent with', 'suggests'."
    )
    return "\n".join(lines)


def build_system_prompt(language: Language) -> str:
    """Return the system prompt for the requested language."""
    lang_name = {
        Language.ENGLISH: "English",
        Language.HINDI: "Hindi",
    }.get(language, "English")

    return (
        f"You are KisanSaathi, an assistant for Indian farmers. Answer ONLY from the evidence "
        f"provided below. Rules:\n"
        f"1. Do not use any external knowledge.\n"
        f"2. Do not invent facts, numbers, or citation IDs.\n"
        f"3. Every claim must be supported by a citation ID from the EVIDENCE section.\n"
        f"4. Citation IDs must be written exactly as shown, e.g. [pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76].\n"
        f"5. If the evidence is insufficient, say so clearly and do not guess.\n"
        f"6. Answer in {lang_name}.\n"
        f"7. Keep answers concise and actionable for a farmer.\n"
    )


def build_user_prompt(
    message: UserMessage,
    citations: CitationBatch,
    eligibility_decision: EligibilityDecision | None,
    weather: WeatherResponse | None,
    vision_result: VisionResult | None = None,
) -> str:
    """Assemble the grounded context into a user prompt."""
    evidence = _format_citations(citations)
    eligibility = _format_eligibility(eligibility_decision)
    weather_str = _format_weather(weather)
    vision_str = _format_vision(vision_result)

    # Truncate if exceeding max context
    full_prompt = (
        f"USER QUESTION:\n{message.text}\n\n"
        f"EVIDENCE:\n{evidence}"
        f"{eligibility}"
        f"{weather_str}"
        f"{vision_str}"
    )

    if len(full_prompt) > _MAX_CONTEXT_CHARS:
        # Truncate evidence section, keep question and system sections
        evidence = evidence[:_MAX_CONTEXT_CHARS - len(full_prompt) + len(evidence)]
        evidence += "\n\n[EVIDENCE TRUNCATED DUE TO LENGTH LIMIT]"
        full_prompt = (
            f"USER QUESTION:\n{message.text}\n\n"
            f"EVIDENCE:\n{evidence}"
            f"{eligibility}"
            f"{weather_str}"
            f"{vision_str}"
        )

    return full_prompt


def _extract_citation_ids(raw: str) -> tuple[str, ...]:
    """Extract citation IDs from the CITATIONS section.

    Expected format: one ID per line, optionally with brackets.
    """
    citations_section = ""
    if "CITATIONS:" in raw:
        _, citations_section = raw.split("CITATIONS:", 1)

    ids = []
    for line in citations_section.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept [id] or id
        if line.startswith("[") and line.endswith("]"):
            line = line[1:-1]
        ids.append(line.strip())
    return tuple(ids)


def parse_generated_answer(
    raw: str,
    allowed_citation_ids: frozenset[str],
    language: Language = Language.ENGLISH,
) -> GeneratedAnswer:
    """Parse and validate model output.

    Expected format:
    ANSWER:
    <free-text answer, may contain [citation_id] inline>

    CITATIONS:
    [citation_id_1]
    [citation_id_2]
    ...

    Raises:
        MalformedOutputError: If ANSWER section is missing or empty.
        GroundingError: If any cited ID is not in allowed_citation_ids.
    """
    if "ANSWER:" not in raw:
        raise MalformedOutputError("Missing ANSWER section in model output")

    answer_part, _, citations_part = raw.partition("CITATIONS:")
    answer_text = answer_part.replace("ANSWER:", "").strip()

    if not answer_text:
        raise MalformedOutputError("ANSWER section is empty")

    cited_ids = _extract_citation_ids(raw)

    # Validate every cited ID against the allowlist
    for cid in cited_ids:
        if cid not in allowed_citation_ids:
            raise GroundingError(f"Model cited unknown citation ID: {cid}")

    # Determine status from answer text (heuristic: if answer says insufficient/unknown)
    status = ResponseStatus.ANSWERED
    low = answer_text.lower()
    insufficient_markers = (
        "insufficient",
        "not enough",
        "cannot determine",
        "unable to determine",
        "not available",
        "no evidence",
        "काफी नहीं",
        "पर्याप्त नहीं",
        "निर्धारित नहीं",
        "उपलब्ध नहीं",
        # Longer Hindi phrases used in tests
        "पर्याप्त सबूत नहीं",
        "काफी सबूत नहीं",
        "निर्धारित नहीं किया जा सकता",
        "उपलब्ध नहीं है",
    )
    if any(marker in low for marker in insufficient_markers):
        status = ResponseStatus.NEEDS_CLARIFICATION

    return GeneratedAnswer(
        text=answer_text,
        citation_ids=cited_ids,
        status=status,
        language=language,
    )