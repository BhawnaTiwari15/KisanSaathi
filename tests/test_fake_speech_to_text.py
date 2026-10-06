"""Unit tests for fake SpeechToText implementations."""

import unittest
from kisansathi.domain.schemas import Language
from kisansathi.voice.fake import (
    FakeSpeechToText,
    TrackingFakeSpeechToText,
    make_fake_speech_to_text,
    make_tracking_fake,
)
from kisansathi.voice.models import (
    SpeechToTextError,
    TranscriptionResult,
    TranscriptionError,
    EmptyTranscriptError,
    InvalidAudioError,
)
from kisansathi.voice.models import Language as VoiceLanguage


class TestFakeSpeechToText(unittest.TestCase):
    def test_returns_canned_transcription_result(self) -> None:
        fake = FakeSpeechToText()
        result = fake.transcribe(b"fake audio", content_type="audio/wav")
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Fake transcript")
        self.assertEqual(result.language, Language.ENGLISH)

    def test_returns_canned_result_with_custom_language(self) -> None:
        fake = make_fake_speech_to_text(text="नमस्ते", language=Language.HINDI)
        result = fake.transcribe(b"audio")
        self.assertEqual(result.text, "नमस्ते")
        self.assertEqual(result.language, Language.HINDI)

    def test_returns_canned_result_with_custom_confidence(self) -> None:
        fake = make_fake_speech_to_text(confidence=0.95)
        result = fake.transcribe(b"audio")
        self.assertEqual(result.confidence, 0.95)

    def test_raises_configured_error(self) -> None:
        fake = FakeSpeechToText(TranscriptionError("model failed"))
        with self.assertRaises(TranscriptionError):
            fake.transcribe(b"audio")

    def test_records_calls(self) -> None:
        fake = FakeSpeechToText()
        fake.transcribe(b"audio1", content_type="audio/wav")
        fake.transcribe(b"audio2", content_type="audio/mp3")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["audio_size"], 6)
        self.assertEqual(fake.calls[0]["content_type"], "audio/wav")
        self.assertEqual(fake.calls[1]["audio_size"], 6)
        self.assertEqual(fake.calls[1]["content_type"], "audio/mp3")

    def test_error_propagates(self) -> None:
        fake = FakeSpeechToText(InvalidAudioError("bad format"))
        with self.assertRaises(InvalidAudioError):
            fake.transcribe(b"audio")

    def test_empty_transcript_error(self) -> None:
        fake = FakeSpeechToText(EmptyTranscriptError("empty"))
        with self.assertRaises(EmptyTranscriptError):
            fake.transcribe(b"audio")


class TestMakeFakeSpeechToText(unittest.TestCase):
    def test_default_creates_valid_result(self) -> None:
        fake = make_fake_speech_to_text()
        result = fake.transcribe(b"audio")
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Fake transcript")
        self.assertEqual(result.language, Language.ENGLISH)
        self.assertEqual(result.confidence, 1.0)

    def test_custom_text(self) -> None:
        fake = make_fake_speech_to_text(text="Custom transcript")
        result = fake.transcribe(b"audio")
        self.assertEqual(result.text, "Custom transcript")

    def test_custom_language(self) -> None:
        fake = make_fake_speech_to_text(language=Language.KANNADA)
        result = fake.transcribe(b"audio")
        self.assertEqual(result.language, Language.KANNADA)

    def test_error_shortcut(self) -> None:
        from kisansathi.voice.models import TranscriptionError
        fake = make_fake_speech_to_text(error=TranscriptionError("boom"))
        with self.assertRaises(TranscriptionError):
            fake.transcribe(b"audio")


class TestTrackingFakeSpeechToText(unittest.TestCase):
    def test_default_result(self) -> None:
        fake = TrackingFakeSpeechToText()
        result = fake.transcribe(b"audio")
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Default transcript")

    def test_override_by_audio_size(self) -> None:
        fake = TrackingFakeSpeechToText()
        custom = TranscriptionResult(text="Custom for 100 bytes", language=Language.HINDI)
        fake.set_result("100", custom)
        result = fake.transcribe(b"x" * 100)
        self.assertEqual(result.text, "Custom for 100 bytes")
        self.assertEqual(result.language, Language.HINDI)

    def test_default_used_when_no_override(self) -> None:
        fake = TrackingFakeSpeechToText()
        fake.set_result("100", TranscriptionResult(text="For 100", language=Language.HINDI))
        result = fake.transcribe(b"x" * 200)  # Different size
        self.assertEqual(result.text, "Default transcript")

    def test_set_default(self) -> None:
        fake = TrackingFakeSpeechToText()
        fake.set_default(TranscriptionResult(text="New default", language=Language.TELUGU))
        result = fake.transcribe(b"audio")
        self.assertEqual(result.text, "New default")
        self.assertEqual(result.language, Language.TELUGU)

    def test_records_all_calls(self) -> None:
        fake = TrackingFakeSpeechToText()
        fake.transcribe(b"audio1", content_type="audio/wav")
        fake.transcribe(b"audio2", content_type="audio/mp3")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["audio_size"], 6)
        self.assertEqual(fake.calls[0]["content_type"], "audio/wav")
        self.assertEqual(fake.calls[1]["audio_size"], 6)
        self.assertEqual(fake.calls[1]["content_type"], "audio/mp3")

    def test_override_with_error(self) -> None:
        fake = TrackingFakeSpeechToText()
        fake.set_result("50", InvalidAudioError("corrupt"))
        with self.assertRaises(InvalidAudioError):
            fake.transcribe(b"x" * 50)


class TestMakeTrackingFake(unittest.TestCase):
    def test_creates_tracking_fake(self) -> None:
        fake = make_tracking_fake()
        self.assertIsInstance(fake, TrackingFakeSpeechToText)
        result = fake.transcribe(b"audio")
        self.assertEqual(result.text, "Default transcript")


if __name__ == "__main__":
    unittest.main()