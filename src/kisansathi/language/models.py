"""Language detection protocols and exceptions."""

from dataclasses import dataclass
from typing import Protocol

from kisansathi.domain.schemas import Language


class LanguageDetector(Protocol):
    """Protocol for language detection.

    Implementations should return the detected language for the given text,
    or None if the language cannot be determined with confidence.
    """

    def detect(self, text: str) -> Language | None:
        """Detect the language of the given text.

        Args:
            text: The text to analyze.

        Returns:
            The detected Language, or None if uncertain.
        """
        ...


class DetectionError(Exception):
    """Raised when language detection fails due to infrastructure issues."""

    pass


class AmbiguousLanguageError(DetectionError):
    """Raised when text contains multiple scripts with no clear majority."""

    pass


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Result of language detection with confidence."""

    language: Language
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")