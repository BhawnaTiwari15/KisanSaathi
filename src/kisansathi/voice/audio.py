"""Audio input validation and normalization.

Provides utilities to validate and normalize audio input before transcription.
"""

from kisansathi.domain.schemas import Language
from kisansathi.voice.models import (
    AudioInput,
    EmptyTranscriptError,
    InvalidAudioError,
    SpeechToTextError,
)

# Supported audio MIME types
SUPPORTED_AUDIO_TYPES = frozenset({
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
    "audio/aac",
    "audio/mp4",
    "audio/m4a",
})

# Maximum audio size (10 MB)
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024

# Minimum audio size (100 bytes)
MIN_AUDIO_SIZE_BYTES = 100

# Maximum duration (30 seconds)
MAX_DURATION_SECONDS = 30.0

# Minimum duration (0.5 seconds)
MIN_DURATION_SECONDS = 0.5


def validate_audio_input(
    audio_data: bytes,
    content_type: str,
    *,
    max_size: int = MAX_AUDIO_SIZE_BYTES,
    min_size: int = MIN_AUDIO_SIZE_BYTES,
    max_duration: float = MAX_DURATION_SECONDS,
    min_duration: float = MIN_DURATION_SECONDS,
) -> AudioInput:
    """Validate and normalize audio input for transcription.

    Args:
        audio_data: Raw audio bytes.
        content_type: MIME type of the audio.
        max_size: Maximum allowed size in bytes.
        min_size: Minimum allowed size in bytes.
        max_duration: Maximum allowed duration in seconds.
        min_duration: Minimum allowed duration in seconds.

    Returns:
        Validated AudioInput object.

    Raises:
        InvalidAudioError: If validation fails.
    """
    if not audio_data:
        raise InvalidAudioError("Audio data is empty")

    if len(audio_data) < min_size:
        raise InvalidAudioError(
            f"Audio data too small: {len(audio_data)} bytes (minimum {min_size})"
        )

    if len(audio_data) > max_size:
        raise InvalidAudioError(
            f"Audio data too large: {len(audio_data)} bytes (maximum {max_size})"
        )

    if not content_type:
        raise InvalidAudioError("Content type is required")

    normalized_content_type = content_type.lower().split(";")[0].strip()
    if normalized_content_type not in SUPPORTED_AUDIO_TYPES:
        raise InvalidAudioError(
            f"Unsupported audio format: {content_type}. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_TYPES))}"
        )

    return AudioInput(
        data=audio_data,
        content_type=normalized_content_type,
        duration_seconds=None,  # Duration unknown without decoding
    )


def validate_transcription_result(
    text: str,
    *,
    min_length: int = 1,
) -> str:
    """Validate transcription result text.

    Args:
        text: Transcribed text from speech model.
        min_length: Minimum allowed length after stripping.

    Returns:
        Stripped text.

    Raises:
        EmptyTranscriptError: If text is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise EmptyTranscriptError("Transcription returned empty text")
    if len(text.strip()) < min_length:
        raise EmptyTranscriptError(
            f"Transcription too short: {len(text.strip())} characters"
        )
    return text.strip()


def map_whisper_language_to_supported(
    whisper_language: str | Language | None,
    supported_languages: frozenset[Language] | None = None,
) -> Language | None:
    """Map Whisper's language code to our supported Language enum.

    Whisper uses ISO 639-1 codes (e.g., "en", "hi", "kn", "te").
    Returns None if the language is not supported or not detected.

    Args:
        whisper_language: Language code from Whisper (e.g., "en", "hi") or
            already-resolved Language enum.
        supported_languages: Set of supported languages. Defaults to all four.

    Returns:
        Corresponding Language enum, or None if unsupported/unknown.
    """
    if whisper_language is None:
        return None

    from kisansathi.domain.schemas import Language

    # If already a Language enum, use directly
    if isinstance(whisper_language, Language):
        lang = whisper_language
    else:
        # Normalize string code to lowercase
        code = whisper_language.lower().strip()

        # Map ISO 639-1 codes to our Language enum
        mapping = {
            "en": Language.ENGLISH,
            "hi": Language.HINDI,
            "kn": Language.KANNADA,
            "te": Language.TELUGU,
        }

        lang = mapping.get(code)
        if lang is None:
            return None
        lang = lang

    # Check against supported languages if provided
    if supported_languages is not None and lang not in supported_languages:
        return None

    return lang