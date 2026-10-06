"""Unit tests for the deterministic language detector."""

import unittest
from kisansathi.domain.schemas import Language
from kisansathi.language import DeterministicLanguageDetector, AmbiguousLanguageError


class TestDeterministicLanguageDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = DeterministicLanguageDetector()

    def test_hindi_devanagari(self) -> None:
        text = "कृपया मुझे PM-KISAN के बारे में बताएं"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.HINDI)

    def test_kannada(self) -> None:
        text = "ದಯವಿಟ್ಟು ನನ್ನ ಸಹಾಯ ಮಾಡಿ"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.KANNADA)

    def test_telugu(self) -> None:
        text = "దయచేసి నాకు సహాయం చేయండి"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.TELUGU)

    def test_english_latin(self) -> None:
        text = "Please tell me about PM-KISAN"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.ENGLISH)

    def test_mixed_script_hindi_dominant(self) -> None:
        # Mixed Devanagari and Latin - Hindi has more characters (including combining marks)
        text = "Help \u0915\u0943\u092a\u092f\u093e"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.HINDI)

    def test_empty_text(self) -> None:
        result = self.detector.detect("")
        self.assertIsNone(result)

    def test_whitespace_only(self) -> None:
        result = self.detector.detect("   \n\t  ")
        self.assertIsNone(result)

    def test_insufficient_alpha_chars(self) -> None:
        # Only 2 alphabetic characters
        text = "Hi"
        result = self.detector.detect(text)
        self.assertIsNone(result)

    def test_hindi_with_numbers_and_punctuation(self) -> None:
        text = "PM-KISAN 2024 के नियम क्या हैं?"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.HINDI)

    def test_kannada_with_english_words(self) -> None:
        text = "PM-KISAN ಯೋಜನೆಯ ವಿವರಗಳನ್ನು ತಿಳಿ"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.KANNADA)

    def test_telugu_with_english_words(self) -> None:
        text = "PM-KISAN యోజన వివరాలు తెలియజేయండి"
        result = self.detector.detect(text)
        self.assertEqual(result, Language.TELUGU)

    def test_detect_with_confidence(self) -> None:
        text = "कृपया सहायता करें"
        result = self.detector.detect_with_confidence(text)
        self.assertIsNotNone(result)
        self.assertEqual(result.language, Language.HINDI)
        self.assertGreater(result.confidence, 0.5)

    def test_english_confidence(self) -> None:
        text = "Please help me"
        result = self.detector.detect_with_confidence(text)
        self.assertIsNotNone(result)
        self.assertEqual(result.language, Language.ENGLISH)
        self.assertGreater(result.confidence, 0.5)

    def test_custom_thresholds(self) -> None:
        # Higher threshold - should fail on mixed text
        detector = DeterministicLanguageDetector(min_script_ratio=0.8)
        text = "Help \u0915\u0943\u092a\u092f\u093e"
        with self.assertRaises(Exception):
            detector.detect(text)

    def test_min_alpha_chars(self) -> None:
        detector = DeterministicLanguageDetector(min_alpha_chars=10)
        text = "नमस्ते"
        result = detector.detect(text)
        self.assertIsNone(result)


class TestFakeLanguageDetector(unittest.TestCase):
    def test_returns_canned_language(self) -> None:
        from kisansathi.language import make_fake_detector
        fake = make_fake_detector(Language.HINDI)
        result = fake.detect("any text")
        self.assertEqual(result, Language.HINDI)

    def test_returns_none(self) -> None:
        from kisansathi.language import make_fake_detector
        fake = make_fake_detector(None)
        result = fake.detect("any text")
        self.assertIsNone(result)

    def test_raises_exception(self) -> None:
        from kisansathi.language import make_fake_detector
        from kisansathi.language import DetectionError
        fake = make_fake_detector(DetectionError("detector failed"))
        with self.assertRaises(DetectionError):
            fake.detect("any text")

    def test_records_calls(self) -> None:
        from kisansathi.language import make_fake_detector
        fake = make_fake_detector(Language.KANNADA)
        fake.detect("text1")
        fake.detect("text2")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0], "text1")
        self.assertEqual(fake.calls[1], "text2")

    def test_detect_with_confidence(self) -> None:
        from kisansathi.language import make_fake_detector
        from kisansathi.language import DetectionResult
        fake = make_fake_detector(Language.TELUGU)
        result = fake.detect_with_confidence("test")
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.language, Language.TELUGU)
        self.assertEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()