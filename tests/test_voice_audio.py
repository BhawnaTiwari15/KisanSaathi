"""Unit tests for audio validation and language mapping."""

import unittest
from kisansathi.domain.schemas import Language
from kisansathi.voice.audio import (
    SUPPORTED_AUDIO_TYPES,
    MAX_AUDIO_SIZE_BYTES,
    MIN_AUDIO_SIZE_BYTES,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    validate_audio_input,
    validate_transcription_result,
    map_whisper_language_to_supported,
)
from kisansathi.voice.models import (
    InvalidAudioError,
    EmptyTranscriptError,
)


class TestValidateAudioInput(unittest.TestCase):
    def test_valid_wav(self) -> None:
        audio = b"RIFF...." + b"x" * 1000  # fake WAV header + data
        result = validate_audio_input(audio, "audio/wav")
        self.assertEqual(result.content_type, "audio/wav")
        self.assertEqual(result.data, audio)

    def test_valid_mp3(self) -> None:
        audio = b"ID3..." + b"x" * 1000  # fake MP3 header + data
        result = validate_audio_input(audio, "audio/mp3")
        self.assertEqual(result.content_type, "audio/mp3")

    def test_valid_ogg(self) -> None:
        audio = b"OggS..." + b"x" * 1000
        result = validate_audio_input(audio, "audio/ogg")
        self.assertEqual(result.content_type, "audio/ogg")

    def test_valid_webm(self) -> None:
        audio = b"\x1a\x45\xdf\xa3" + b"x" * 1000  # WebM header
        result = validate_audio_input(audio, "audio/webm")
        self.assertEqual(result.content_type, "audio/webm")

    def test_case_insensitive_content_type(self) -> None:
        audio = b"x" * 1000
        result = validate_audio_input(audio, "AUDIO/WAV")
        self.assertEqual(result.content_type, "audio/wav")

    def test_content_type_with_parameters(self) -> None:
        audio = b"x" * 1000
        result = validate_audio_input(audio, "audio/wav; charset=binary")
        self.assertEqual(result.content_type, "audio/wav")

    def test_empty_audio_raises(self) -> None:
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(b"", "audio/wav")

    def test_audio_too_small_raises(self) -> None:
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(b"x" * 50, "audio/wav")

    def test_audio_too_large_raises(self) -> None:
        large_audio = b"x" * (MAX_AUDIO_SIZE_BYTES + 1)
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(large_audio, "audio/wav")

    def test_unsupported_format_raises(self) -> None:
        audio = b"x" * 1000
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(audio, "audio/amr")  # not in SUPPORTED_AUDIO_TYPES

    def test_unknown_format_raises(self) -> None:
        audio = b"x" * 1000
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(audio, "audio/unknown")

    def test_missing_content_type_raises(self) -> None:
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(b"x" * 1000, "")

    def test_none_content_type_raises(self) -> None:
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(b"x" * 1000, None)  # type: ignore

    def test_custom_limits(self) -> None:
        # Custom min size
        audio = b"x" * 200
        result = validate_audio_input(audio, "audio/wav", min_size=100)
        self.assertEqual(len(result.data), 200)

        # Custom max size
        large_audio = b"x" * 5000
        with self.assertRaises(InvalidAudioError):
            validate_audio_input(large_audio, "audio/wav", max_size=1000)


class TestValidateTranscriptionResult(unittest.TestCase):
    def test_valid_text(self) -> None:
        result = validate_transcription_result("Hello world")
        self.assertEqual(result, "Hello world")

    def test_whitespace_stripped(self) -> None:
        result = validate_transcription_result("  Hello world  ")
        self.assertEqual(result, "Hello world")

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(EmptyTranscriptError):
            validate_transcription_result("")

    def test_whitespace_only_raises(self) -> None:
        with self.assertRaises(EmptyTranscriptError):
            validate_transcription_result("   \n\t  ")

    def test_min_length(self) -> None:
        result = validate_transcription_result("Hi", min_length=2)
        self.assertEqual(result, "Hi")

    def test_too_short_raises(self) -> None:
        with self.assertRaises(EmptyTranscriptError):
            validate_transcription_result("H", min_length=2)


class TestMapWhisperLanguageToSupported(unittest.TestCase):
    def test_english(self) -> None:
        result = map_whisper_language_to_supported("en")
        self.assertEqual(result, Language.ENGLISH)

    def test_hindi(self) -> None:
        result = map_whisper_language_to_supported("hi")
        self.assertEqual(result, Language.HINDI)

    def test_kannada(self) -> None:
        result = map_whisper_language_to_supported("kn")
        self.assertEqual(result, Language.KANNADA)

    def test_telugu(self) -> None:
        result = map_whisper_language_to_supported("te")
        self.assertEqual(result, Language.TELUGU)

    def test_case_insensitive(self) -> None:
        result = map_whisper_language_to_supported("EN")
        self.assertEqual(result, Language.ENGLISH)

    def test_whitespace_handling(self) -> None:
        result = map_whisper_language_to_supported("  hi  ")
        self.assertEqual(result, Language.HINDI)

    def test_none_returns_none(self) -> None:
        result = map_whisper_language_to_supported(None)
        self.assertIsNone(result)

    def test_empty_string_returns_none(self) -> None:
        result = map_whisper_language_to_supported("")
        self.assertIsNone(result)

    def test_unsupported_language_returns_none(self) -> None:
        result = map_whisper_language_to_supported("fr")
        self.assertIsNone(result)

    def test_unsupported_language_with_custom_set(self) -> None:
        from kisansathi.domain.schemas import Language
        supported = frozenset({Language.ENGLISH, Language.HINDI})
        result = map_whisper_language_to_supported("kn", supported)
        self.assertIsNone(result)

    def test_supported_language_with_custom_set(self) -> None:
        from kisansathi.domain.schemas import Language
        supported = frozenset({Language.ENGLISH, Language.HINDI})
        result = map_whisper_language_to_supported("hi", supported)
        self.assertEqual(result, Language.HINDI)


if __name__ == "__main__":
    unittest.main()