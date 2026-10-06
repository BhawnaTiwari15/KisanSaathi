"""Speech-to-text contracts and protocols.

This module defines the domain-level interface for speech recognition.
Concrete implementations (local Whisper, hosted API) live in infrastructure,
not here. The domain depends only on the SpeechToText protocol.
"""

from dataclasses import dataclass
from typing import Protocol

from kisansathi.domain.schemas import Language


class SpeechToTextError(Exception):
    """Base exception for speech-to-text failures."""
    pass


class TranscriptionError(SpeechToTextError):
    """Raised when transcription fails (model error, timeout, etc.)."""
    pass


class EmptyTranscriptError(SpeechToTextError):
    """Raised when transcription returns empty or whitespace-only text."""
    pass


class InvalidAudioError(SpeechToTextError):
    """Raised when audio input is invalid, corrupted, or unsupported."""
    pass


class UnsupportedLanguageError(SpeechToTextError):
    """Raised when the detected language is not supported by the system."""
    pass


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Result of speech-to-text transcription.

    Attributes:
        text: The transcribed text. Never empty for successful results.
        language: The language detected by the speech model, if available.
        confidence: Model confidence score (0.0 to 1.0), if available.
        duration_seconds: Duration of the processed audio, if available.
    """
    text: str
    language: Language | None = None
    confidence: float | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("TranscriptionResult text must not be empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")


class SpeechToText(Protocol):
    """Protocol for speech-to-text transcription.

    Implementations must be stateless and thread-safe.
    The domain layer depends only on this protocol.
    """

    def transcribe(self, audio_data: bytes, *, content_type: str | None = None) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes.
            content_type: MIME type of the audio (e.g., "audio/wav", "audio/mp3"),
                if known. Used for format validation and routing.

        Returns:
            TranscriptionResult with the transcribed text and metadata.

        Raises:
            InvalidAudioError: If audio format is unsupported, corrupted, or invalid.
            TranscriptionError: If the transcription service/model fails.
            EmptyTranscriptError: If transcription returns empty text.
        """
        ...


@dataclass(frozen=True, slots=True)
class AudioInput:
    """Validated audio input ready for transcription.

    This is the normalized form passed between layers.
    """
    data: bytes
    content_type: str
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("AudioInput data must not be empty")
        if not self.content_type:
            raise ValueError("AudioInput content_type must not be empty")