"""Fake SpeechToText implementations for testing.

These fakes allow testing voice integration without network calls or model weights.
"""

from kisansathi.domain.schemas import Language
from kisansathi.voice.models import (
    SpeechToText,
    SpeechToTextError,
    TranscriptionResult,
    TranscriptionError,
    EmptyTranscriptError,
    InvalidAudioError,
)


class FakeSpeechToText:
    """Fake SpeechToText implementation for testing.

    Returns canned responses or raises configured exceptions.
    """

    def __init__(
        self,
        result: TranscriptionResult | SpeechToTextError | None = None,
    ) -> None:
        self._result = result or TranscriptionResult(
            text="Fake transcript",
            language=Language.ENGLISH,
            confidence=1.0,
        )
        self.calls: list[dict] = []

    def transcribe(self, audio_data: bytes, *, content_type: str | None = None) -> TranscriptionResult:
        self.calls.append({
            "audio_size": len(audio_data),
            "content_type": content_type,
        })
        if isinstance(self._result, SpeechToTextError):
            raise self._result
        return self._result


def make_fake_speech_to_text(
    text: str = "Fake transcript",
    language: Language = Language.ENGLISH,
    confidence: float = 1.0,
    error: SpeechToTextError | None = None,
) -> FakeSpeechToText:
    """Create a FakeSpeechToText with a canned result or error."""
    if error is not None:
        return FakeSpeechToText(error)
    return FakeSpeechToText(
        TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
        )
    )


class TrackingFakeSpeechToText:
    """Fake SpeechToText that tracks calls and can be configured per call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._default_result: TranscriptionResult = TranscriptionResult(
            text="Default transcript",
            language=Language.ENGLISH,
            confidence=1.0,
        )
        self._overrides: dict[str, TranscriptionResult | SpeechToTextError] = {}

    def set_result(self, key: str, result: TranscriptionResult | SpeechToTextError) -> None:
        """Configure a specific result for a given audio key."""
        self._overrides[key] = result

    def set_default(self, result: TranscriptionResult) -> None:
        self._default_result = result

    def transcribe(self, audio_data: bytes, *, content_type: str | None = None) -> TranscriptionResult:
        call_info = {
            "audio_size": len(audio_data),
            "content_type": content_type,
        }
        self.calls.append(call_info)

        # Check for override based on audio size (simple key)
        key = str(len(audio_data))
        if key in self._overrides:
            result = self._overrides[key]
        else:
            result = self._default_result

        if isinstance(result, SpeechToTextError):
            raise result
        return result


def make_tracking_fake() -> TrackingFakeSpeechToText:
    """Create a TrackingFakeSpeechToText."""
    return TrackingFakeSpeechToText()