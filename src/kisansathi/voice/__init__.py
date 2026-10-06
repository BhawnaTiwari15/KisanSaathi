"""Voice input package.

Provides speech-to-text functionality for converting farmer voice input to text.
"""

from kisansathi.voice.audio import (
    MAX_AUDIO_SIZE_BYTES,
    MAX_DURATION_SECONDS,
    MIN_AUDIO_SIZE_BYTES,
    MIN_DURATION_SECONDS,
    SUPPORTED_AUDIO_TYPES,
    map_whisper_language_to_supported,
    validate_audio_input,
    validate_transcription_result,
)
from kisansathi.voice.fake import (
    FakeSpeechToText,
    TrackingFakeSpeechToText,
    make_fake_speech_to_text,
    make_tracking_fake,
)
from kisansathi.voice.models import (
    AudioInput,
    EmptyTranscriptError,
    InvalidAudioError,
    SpeechToText,
    SpeechToTextError,
    TranscriptionError,
    TranscriptionResult,
    UnsupportedLanguageError,
)

__all__ = [
    "AudioInput",
    "EmptyTranscriptError",
    "InvalidAudioError",
    "SpeechToText",
    "SpeechToTextError",
    "TranscriptionError",
    "TranscriptionResult",
    "UnsupportedLanguageError",
    "SUPPORTED_AUDIO_TYPES",
    "MAX_AUDIO_SIZE_BYTES",
    "MAX_DURATION_SECONDS",
    "MIN_AUDIO_SIZE_BYTES",
    "MIN_DURATION_SECONDS",
    "validate_audio_input",
    "validate_transcription_result",
    "map_whisper_language_to_supported",
    "FakeSpeechToText",
    "TrackingFakeSpeechToText",
    "make_fake_speech_to_text",
    "make_tracking_fake",
]