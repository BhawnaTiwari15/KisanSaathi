"""Small, cross-cutting message and citation contracts."""

from dataclasses import dataclass, field
from enum import StrEnum
import re


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHUNK_ID = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*):([0-9a-f]{24})$")


class Language(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"
    TELUGU = "te"


class ResponseStatus(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    ABSTAINED = "abstained"


class Route(StrEnum):
    """The routes the request may take out of routing."""

    RETRIEVAL = "retrieval"
    ELIGIBILITY = "eligibility"
    WEATHER = "weather"
    CLARIFY = "clarify"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class UserMessage:
    text: str
    language: Language | None = None
    detected_language: Language | None = None
    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Message text must not be empty")
        _validate_optional_coordinate(self.latitude, "latitude", -90.0, 90.0)
        _validate_optional_coordinate(self.longitude, "longitude", -180.0, 180.0)


def _validate_optional_coordinate(
    value: float | None,
    field_name: str,
    minimum: float,
    maximum: float,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number or None")
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be within [{minimum:g}, {maximum:g}]")


@dataclass(frozen=True, slots=True)
class Citation:
    """A public, user-facing reference to a specific place in a registered source.

    Every field is copied verbatim from registered source metadata or from stored chunk
    provenance. Nothing here is constructed, guessed, or repaired: a value that the
    registry does not supply is simply absent.

    ``page_number`` is the first page of the cited span; ``page_end`` is set only when
    the span crosses a page boundary. ``issuing_authority`` keeps any attribution caveat
    recorded in the source manifest. ``chunk_id`` is internal traceability, so an auditor
    can go from a rendered citation back to the exact stored chunk.
    """

    source_id: str
    title: str
    url: str
    page_number: int | None = None
    page_end: int | None = None
    issuing_authority: str | None = None
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _SAFE_ID.fullmatch(self.source_id):
            raise ValueError("Citation source_id must be a non-empty path-safe identifier")
        if not self.title.strip():
            raise ValueError("Citation title must not be empty")
        if not self.url.strip():
            raise ValueError("Citation url must not be empty")
        for name in ("page_number", "page_end"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Citation {name} must be a positive integer")
        if self.page_number is not None and self.page_end is not None:
            if self.page_end < self.page_number:
                raise ValueError("Citation page_end must not be before page_number")
        if self.issuing_authority is not None and not self.issuing_authority.strip():
            raise ValueError("Citation issuing_authority must not be blank")
        if self.chunk_id is not None:
            if not isinstance(self.chunk_id, str) or _CHUNK_ID.fullmatch(self.chunk_id) is None:
                raise ValueError("Citation chunk_id must be '<source_id>:<24 hex>'")
            if self.chunk_id.split(":", 1)[0] != self.source_id:
                raise ValueError("Citation chunk_id must belong to its source_id")

    @property
    def page_span(self) -> tuple[int, int] | None:
        """Return the cited (first, last) page, or None when no page is recorded."""
        if self.page_number is None:
            return None
        return self.page_number, self.page_end or self.page_number


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    text: str
    language: Language
    status: ResponseStatus = ResponseStatus.ANSWERED
    citations: tuple[Citation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Response text must not be empty")