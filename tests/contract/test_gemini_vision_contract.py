"""
Contract tests for real Gemini vision provider.

These tests REQUIRE:
- RUN_REAL_VISION_TESTS=1 environment variable
- GEMINI_API_KEY environment variable with a valid API key

They are SKIPPED by default in normal CI runs.

To run manually:
    RUN_REAL_VISION_TESTS=1 GEMINI_API_KEY=your-key .venv\\Scripts\\python.exe -m pytest tests/contract/test_gemini_vision_contract.py -v -s

These tests validate the end-to-end integration with the real Gemini API.
They do NOT mock any HTTP calls.
"""

import os
import unittest

import pytest

from kisansathi.config import Settings
from kisansathi.vision.image import ValidatedImage
from kisansathi.vision.models import VisionResult, VisionStatus
from kisansathi.vision.providers.gemini import GeminiVisionAnalyzer


# Valid test image data (JPEG > 100 bytes)
VALID_JPEG = (
    b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0C\x14\r\x0C\x0B\x0B\x0C\x19\x12\x13\x0F\x14\x1D\x1A"
    b"\x1F\x1E\x1D\x1A\x1C\x1C $.' \",#\x1C\x1C(7),01444\x1F'9=82<.342"
    b"\xFF\xC0\x00\x0B\x08\x00\x01\x00\x01\x01\x11\x00\xFF\xD9"
)


def _real_test_enabled() -> bool:
    """Check if real vision tests should run."""
    return os.getenv("RUN_REAL_VISION_TESTS") == "1" and os.getenv("GEMINI_API_KEY") is not None


def _get_real_settings() -> Settings:
    """Create settings for real provider test."""
    return Settings(
        vision_provider="gemini",
        vision_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest"),
        vision_api_key=os.getenv("GEMINI_API_KEY"),
        vision_timeout_seconds=30.0,
        vision_connect_timeout_seconds=10.0,
        vision_max_retries=1,
        vision_retry_backoff_base=2.0,
        vision_max_image_bytes=10 * 1024 * 1024,
        vision_temperature=0.0,
        vision_max_output_tokens=1024,
    )


@pytest.mark.skipif(not _real_test_enabled(), reason="Real vision tests require RUN_REAL_VISION_TESTS=1 and GEMINI_API_KEY")
class TestGeminiVisionContract(unittest.TestCase):
    """Contract tests against real Gemini API."""

    def setUp(self):
        self.settings = _get_real_settings()
        self.analyzer = GeminiVisionAnalyzer(self.settings)

    def tearDown(self):
        self.analyzer.close()

    def test_real_provider_returns_valid_vision_result(self):
        """Test that real provider returns a valid VisionResult structure."""
        result = self.analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        # Basic structure validation
        self.assertIsInstance(result, VisionResult)
        self.assertIn(result.status, VisionStatus)
        self.assertIsInstance(result.observations, tuple)
        self.assertIsInstance(result.language, str)
        self.assertIsInstance(result.confidence_overall, float)
        self.assertTrue(0.0 <= result.confidence_overall <= 1.0)
        self.assertIsInstance(result.analyzer_metadata, dict)
        self.assertEqual(result.analyzer_metadata["provider"], "gemini")

        # If successful, should have observations
        if result.status == VisionStatus.SUCCESS:
            self.assertGreater(len(result.observations), 0)
            for obs in result.observations:
                self.assertIsInstance(obs.label, str)
                self.assertTrue(len(obs.label) > 0)
                self.assertIn(obs.category, ("crop", "disease", "pest", "stress", "healthy", "other"))
                self.assertIsInstance(obs.confidence, float)
                self.assertTrue(0.0 <= obs.confidence <= 1.0)
                self.assertIsInstance(obs.description, str)
                self.assertTrue(len(obs.description) > 0)

        # If low confidence, should still have observations (clarification status)
        if result.status == VisionStatus.LOW_CONFIDENCE:
            self.assertGreater(len(result.observations), 0)

        # If no usable information, should have no observations
        if result.status == VisionStatus.NO_USABLE_INFORMATION:
            self.assertEqual(len(result.observations), 0)

    def test_real_provider_handles_crop_image(self):
        """Test provider response on a typical crop image."""
        result = self.analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        # Should not crash and should return a valid result
        self.assertIsInstance(result, VisionResult)
        self.assertIn(result.status, (
            VisionStatus.SUCCESS,
            VisionStatus.LOW_CONFIDENCE,
            VisionStatus.UNABLE_TO_ANALYZE,
            VisionStatus.NO_USABLE_INFORMATION,
        ))

    def test_real_provider_confidence_threshold_filtering(self):
        """Test that confidence_threshold parameter filters observations."""
        # High threshold - should get fewer or no observations
        result_high = self.analyzer.analyze(
            VALID_JPEG,
            content_type="image/jpeg",
            confidence_threshold=0.9,
        )

        # Low threshold - should get more observations
        result_low = self.analyzer.analyze(
            VALID_JPEG,
            content_type="image/jpeg",
            confidence_threshold=0.1,
        )

        # Both should be valid results
        self.assertIsInstance(result_high, VisionResult)
        self.assertIsInstance(result_low, VisionResult)

        # High threshold should have <= observations than low threshold
        # (may be equal if all observations have high confidence)
        self.assertLessEqual(len(result_high.observations), len(result_low.observations))

    def test_real_provider_max_observations_limit(self):
        """Test that max_observations parameter limits results."""
        result = self.analyzer.analyze(
            VALID_JPEG,
            content_type="image/jpeg",
            max_observations=2,
        )

        self.assertIsInstance(result, VisionResult)
        self.assertLessEqual(len(result.observations), 2)

    def test_real_provider_invalid_image_format_rejected(self):
        """Test that invalid image format is rejected before provider call."""
        from kisansathi.vision.models import UnsupportedFormatError

        with self.assertRaises(UnsupportedFormatError):
            self.analyzer.analyze(VALID_JPEG, content_type="image/gif")

    def test_real_provider_oversized_image_rejected(self):
        """Test that oversized image is rejected before provider call."""
        from kisansathi.vision.models import ImageTooLargeError

        oversized = b"x" * (10 * 1024 * 1024 + 1)
        with self.assertRaises(ImageTooLargeError):
            self.analyzer.analyze(oversized, content_type="image/jpeg")

    def test_real_provider_empty_image_rejected(self):
        """Test that empty image is rejected before provider call."""
        from kisansathi.vision.models import ImageValidationError

        with self.assertRaises(ImageValidationError):
            self.analyzer.analyze(b"", content_type="image/jpeg")

    def test_real_provider_latency_reasonable(self):
        """Test that provider latency is within expected bounds."""
        import time
        start = time.monotonic()
        result = self.analyzer.analyze(VALID_JPEG, content_type="image/jpeg")
        latency = time.monotonic() - start

        # Should complete within timeout (30s) with margin
        self.assertLess(latency, 25.0, f"Provider latency {latency:.1f}s exceeded 25s")
        self.assertIsInstance(result, VisionResult)

    def test_real_provider_response_has_expected_metadata(self):
        """Test that response metadata contains expected fields."""
        result = self.analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("provider", result.analyzer_metadata)
        self.assertEqual(result.analyzer_metadata["provider"], "gemini")
        self.assertIn("model", result.analyzer_metadata)
        self.assertTrue(len(result.analyzer_metadata["model"]) > 0)


@pytest.mark.skipif(not _real_test_enabled(), reason="Real vision tests require RUN_REAL_VISION_TESTS=1 and GEMINI_API_KEY")
class TestGeminiVisionContractEdgeCases(unittest.TestCase):
    """Edge case contract tests."""

    def setUp(self):
        self.settings = _get_real_settings()
        self.analyzer = GeminiVisionAnalyzer(self.settings)

    def tearDown(self):
        self.analyzer.close()

    def test_real_provider_multiple_calls_consistent(self):
        """Test that multiple calls with same image produce consistent results."""
        result1 = self.analyzer.analyze(VALID_JPEG, content_type="image/jpeg")
        result2 = self.analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        # Both should be valid
        self.assertIsInstance(result1, VisionResult)
        self.assertIsInstance(result2, VisionResult)

        # Status should be consistent (though exact observations may vary)
        self.assertEqual(result1.status, result2.status)

    def test_real_provider_different_image_formats(self):
        """Test provider handles different valid image formats."""
        # Test PNG
        VALID_PNG = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff"
            b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd5\x8b\xd3\x00\x00\x00\x00IEND\xaeB`\x82"
            + b"\x00" * 40
        )

        result = self.analyzer.analyze(VALID_PNG, content_type="image/png")
        self.assertIsInstance(result, VisionResult)
        self.assertIn(result.status, VisionStatus)


if __name__ == "__main__":
    # Allow running directly
    if _real_test_enabled():
        unittest.main()
    else:
        print("Real vision tests disabled. Set RUN_REAL_VISION_TESTS=1 and GEMINI_API_KEY to run.")