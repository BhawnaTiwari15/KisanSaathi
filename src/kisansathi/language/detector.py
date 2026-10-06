"""Deterministic language detection based on Unicode code point ranges.

This implementation uses Unicode code point ranges to identify the
primary script of the input text. It supports:
- Devanagari (Hindi): U+0900–U+097F
- Kannada: U+0C80–U+0CFF
- Telugu: U+0C00–U+0C7F
- Latin (English): U+0000–U+024F (Basic Latin through Latin Extended-B)

The detector is deterministic, requires no external dependencies, and is
suitable for use in tests and as a production fallback.
"""

from collections import Counter
from typing import Counter as CounterType

from kisansathi.domain.schemas import Language
from kisansathi.language.models import (
    AmbiguousLanguageError,
    DetectionError,
    DetectionResult,
    LanguageDetector,
)

# Unicode code point ranges for each supported script
# Each entry is (start, end, Language)
_SCRIPT_RANGES = [
    (0x0900, 0x097F, Language.HINDI),        # Devanagari
    (0x0C80, 0x0CFF, Language.KANNADA),       # Kannada
    (0x0C00, 0x0C7F, Language.TELUGU),        # Telugu
    (0x0000, 0x024F, Language.ENGLISH),       # Latin (Basic Latin through Latin Extended-B)
]

# Minimum proportion of dominant script for confident detection
_MIN_SCRIPT_RATIO = 0.5

# Minimum number of alphabetic characters required for detection
_MIN_ALPHA_CHARS = 3


def _get_script_for_char(char: str) -> Language | None:
    """Return the Language for a character based on its Unicode code point.

    For Latin (English), only alphabetic characters count (excludes digits,
    punctuation, spaces). For Indic scripts (Devanagari, Kannada, Telugu),
    all characters in the script's Unicode block are counted, including
    combining marks which are not `isalpha()` but are part of the script.

    Args:
        char: A single character string.

    Returns:
        The Language enum value, or None if the character is not in a
        recognized script range.
    """
    if not char or len(char) != 1:
        return None

    codepoint = ord(char)

    # Latin (English): only alphabetic characters count
    if 0x0000 <= codepoint <= 0x024F:
        return Language.ENGLISH if char.isalpha() else None

    # Devanagari (Hindi): U+0900–U+097F
    if 0x0900 <= codepoint <= 0x097F:
        return Language.HINDI

    # Kannada: U+0C80–U+0CFF
    if 0x0C80 <= codepoint <= 0x0CFF:
        return Language.KANNADA

    # Telugu: U+0C00–U+0C7F
    if 0x0C00 <= codepoint <= 0x0C7F:
        return Language.TELUGU

    return None


class DeterministicLanguageDetector:
    """Deterministic language detector using Unicode code point ranges.

    This detector counts characters belonging to each supported script
    and returns the dominant script's language if it exceeds the
    confidence threshold.
    """

    def __init__(
        self,
        *,
        min_script_ratio: float = _MIN_SCRIPT_RATIO,
        min_alpha_chars: int = _MIN_ALPHA_CHARS,
    ) -> None:
        """Initialize the detector.

        Args:
            min_script_ratio: Minimum proportion of dominant script required
                for confident detection (0.0 to 1.0).
            min_alpha_chars: Minimum number of alphabetic characters required
                to attempt detection.
        """
        if not 0.0 <= min_script_ratio <= 1.0:
            raise ValueError("min_script_ratio must be between 0.0 and 1.0")
        if min_alpha_chars < 1:
            raise ValueError("min_alpha_chars must be at least 1")
        self._min_script_ratio = min_script_ratio
        self._min_alpha_chars = min_alpha_chars

    def detect(self, text: str) -> Language | None:
        """Detect the language of the given text.

        Args:
            text: The text to analyze.

        Returns:
            The detected Language, or None if uncertain or no alphabetic
            characters found.

        Raises:
            AmbiguousLanguageError: If text contains multiple scripts with
                no clear majority.
        """
        if not text or not text.strip():
            return None

        script_counts = self._count_scripts(text)
        if not script_counts:
            return None

        total = sum(script_counts.values())
        if total < self._min_alpha_chars:
            return None

        dominant_lang, count = script_counts.most_common(1)[0]
        ratio = count / total

        if ratio < self._min_script_ratio:
            raise AmbiguousLanguageError(
                f"No dominant script found (top: {dominant_lang.value}={ratio:.2f})"
            )

        return dominant_lang

    def detect_with_confidence(self, text: str) -> DetectionResult | None:
        """Detect language with confidence score.

        Args:
            text: The text to analyze.

        Returns:
            DetectionResult with language and confidence, or None if uncertain.
        """
        if not text or not text.strip():
            return None

        script_counts = self._count_scripts(text)
        if not script_counts:
            return None

        total = sum(script_counts.values())
        if total < self._min_alpha_chars:
            return None

        dominant_lang, count = script_counts.most_common(1)[0]
        ratio = count / total

        if ratio < self._min_script_ratio:
            return None

        return DetectionResult(language=dominant_lang, confidence=ratio)

    def _count_scripts(self, text: str) -> Counter[Language]:
        """Count languages of alphabetic characters in text."""
        counts: Counter[Language] = Counter()
        for char in text:
            lang = _get_script_for_char(char)
            if lang is not None:
                counts[lang] += 1
        return counts


class _FakeLanguageDetector:
    """Fake language detector for testing.

    Returns a canned language or raises a configured exception.
    """

    def __init__(
        self,
        result: Language | Exception | None = Language.ENGLISH,
    ) -> None:
        self._result = result
        self.calls: list[str] = []

    def detect(self, text: str) -> Language | None:
        self.calls.append(text)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def detect_with_confidence(self, text: str) -> DetectionResult | None:
        self.calls.append(text)
        if isinstance(self._result, Exception):
            raise self._result
        if self._result is None:
            return None
        return DetectionResult(language=self._result, confidence=1.0)


def make_fake_detector(
    language: Language | Exception | None = Language.ENGLISH,
) -> _FakeLanguageDetector:
    """Create a fake language detector for testing."""
    return _FakeLanguageDetector(language)