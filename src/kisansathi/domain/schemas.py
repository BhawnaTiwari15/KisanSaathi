"""Small, cross-cutting message and citation contracts."""

from dataclasses import dataclass, field
from enum import StrEnum


class Language(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"


class ResponseStatus(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class UserMessage:
    text: str
    language: Language | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Message text must not be empty")


@dataclass(frozen=True, slots=True)
class Citation:
    source_id: str
    title: str
    url: str
    page_number: int | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Citation source_id must not be empty")
        if not self.title.strip():
            raise ValueError("Citation title must not be empty")
        if not self.url.strip():
            raise ValueError("Citation url must not be empty")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("Citation page_number must be at least 1")


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    text: str
    language: Language
    status: ResponseStatus = ResponseStatus.ANSWERED
    citations: tuple[Citation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Response text must not be empty")