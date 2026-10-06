"""Language detection package.

Provides protocols and implementations for detecting the language of user input.
"""

from kisansathi.language.detector import (
    DeterministicLanguageDetector,
    make_fake_detector,
)
from kisansathi.language.models import (
    AmbiguousLanguageError,
    DetectionError,
    DetectionResult,
    LanguageDetector,
)

__all__ = [
    "AmbiguousLanguageError",
    "DetectionError",
    "DetectionResult",
    "DeterministicLanguageDetector",
    "LanguageDetector",
    "make_fake_detector",
]