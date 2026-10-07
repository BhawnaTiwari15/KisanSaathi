"""Unit tests for vision providers (using mocked HTTP)."""

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx

from kisansathi.config import Settings
from kisansathi.vision.fake import FakeVisionAnalyzer
from kisansathi.vision.image import ValidatedImage
from kisansathi.vision.models import (
    VisionResult,
    VisionStatus,
    AnalyzerUnavailableError,
    AnalyzerTimeoutError,
    ImageValidationError,
    ImageTooLargeError,
)
from kisansathi.vision.providers.gemini import GeminiVisionAnalyzer
from kisansathi.vision.providers.base import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    InvalidRequestError,
    MalformedResponseError,
)


# Valid test image data (JPEG > 100 bytes)
VALID_JPEG = (
    b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0C\x14\r\x0C\x0B\x0B\x0C\x19\x12\x13\x0F\x14\x1D\x1A"
    b"\x1F\x1E\x1D\x1A\x1C\x1C $.' \",#\x1C\x1C(7),01444\x1F'9=82<.342"
    b"\xFF\xC0\x00\x0B\x08\x00\x01\x00\x01\x01\x11\x00\xFF\xD9"
)


def make_settings(**overrides) -> Settings:
    """Create test settings with defaults."""
    defaults = {
        "vision_provider": "gemini",
        "vision_model_name": "gemini-1.5-flash-latest",
        "vision_api_key": "test-api-key",
        "vision_timeout_seconds": 15.0,
        "vision_connect_timeout_seconds": 5.0,
        "vision_max_retries": 2,
        "vision_retry_backoff_base": 1.0,
        "vision_max_image_bytes": 10 * 1024 * 1024,
        "vision_temperature": 0.0,
        "vision_max_output_tokens": 1024,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_validated_image(data: bytes = VALID_JPEG) -> ValidatedImage:
    """Create a pre-validated image for testing."""
    return ValidatedImage(
        data=data,
        content_type="image/jpeg",
        size_bytes=len(data),
        detected_format="jpeg",
        filename="test.jpg",
        width=1,
        height=1,
    )


class TestGeminiVisionAnalyzer(unittest.TestCase):
    """Tests for GeminiVisionAnalyzer with mocked HTTP."""

    def setUp(self):
        self.settings = make_settings()
        self.validated_image = make_validated_image()

    def _make_mock_response(self, status_code: int, json_data: dict | None = None, text: str = "") -> httpx.Response:
        """Create a mock httpx.Response."""
        mock = MagicMock(spec=httpx.Response)
        mock.status_code = status_code
        if json_data is not None:
            mock.json.return_value = json_data
        mock.text = text or json.dumps(json_data) if json_data else ""
        return mock

    def _make_success_response(self, observations: list[dict] | None = None) -> dict:
        """Create a successful Gemini response payload."""
        if observations is None:
            observations = [
                {
                    "label": "tomato plant",
                    "category": "crop",
                    "confidence": 0.92,
                    "description": "Green tomato plant with fruit clusters visible.",
                },
                {
                    "label": "late blight symptoms",
                    "category": "disease",
                    "confidence": 0.65,
                    "description": "Brown lesions with white margins; CONSISTENT WITH late blight but not confirmed.",
                },
            ]
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "observations": observations,
                                    "language": "en",
                                    "overall_confidence": 0.785,
                                    "notes": None,
                                })
                            }
                        ]
                    }
                }
            ]
        }

    @patch("httpx.Client.post")
    def test_successful_analysis(self, mock_post):
        """Test successful provider response parsing."""
        mock_post.return_value = self._make_mock_response(200, self._make_success_response())

        analyzer = GeminiVisionAnalyzer(self.settings)
        result = analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertEqual(result.status, VisionStatus.SUCCESS)
        self.assertEqual(len(result.observations), 2)
        self.assertEqual(result.observations[0].label, "tomato plant")
        self.assertEqual(result.observations[0].category, "crop")
        self.assertAlmostEqual(result.observations[0].confidence, 0.92)
        self.assertEqual(result.observations[1].category, "disease")
        self.assertAlmostEqual(result.confidence_overall, 0.785, places=2)
        self.assertEqual(result.analyzer_metadata["provider"], "gemini")

    @patch("httpx.Client.post")
    def test_low_confidence_observations_filtered(self, mock_post):
        """Test that observations below threshold are filtered."""
        observations = [
            {"label": "clear crop", "category": "crop", "confidence": 0.9, "description": "Clear crop"},
            {"label": "uncertain pest", "category": "pest", "confidence": 0.3, "description": "Maybe pest"},
        ]
        mock_post.return_value = self._make_mock_response(200, self._make_success_response(observations))

        analyzer = GeminiVisionAnalyzer(self.settings)
        result = analyzer.analyze(VALID_JPEG, content_type="image/jpeg", confidence_threshold=0.5)

        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.observations[0].label, "clear crop")

    @patch("httpx.Client.post")
    def test_no_observations_after_filter_returns_no_usable_information(self, mock_post):
        """Test that empty observations after filtering gives NO_USABLE_INFORMATION."""
        observations = [
            {"label": "uncertain", "category": "other", "confidence": 0.2, "description": "Uncertain"},
        ]
        mock_post.return_value = self._make_mock_response(200, self._make_success_response(observations))

        analyzer = GeminiVisionAnalyzer(self.settings)
        result = analyzer.analyze(VALID_JPEG, content_type="image/jpeg", confidence_threshold=0.5)

        self.assertEqual(result.status, VisionStatus.NO_USABLE_INFORMATION)
        self.assertEqual(len(result.observations), 0)

    @patch("httpx.Client.post")
    def test_confidence_clamping(self, mock_post):
        """Test that confidence values are clamped to [0, 1]."""
        observations = [
            {"label": "over", "category": "crop", "confidence": 1.5, "description": "Over"},
            {"label": "under", "category": "crop", "confidence": -0.2, "description": "Under"},
            {"label": "normal", "category": "crop", "confidence": 0.8, "description": "Normal"},
        ]
        mock_post.return_value = self._make_mock_response(200, self._make_success_response(observations))

        analyzer = GeminiVisionAnalyzer(self.settings)
        # Use threshold lower than original values to include clamped ones
        result = analyzer.analyze(VALID_JPEG, content_type="image/jpeg", confidence_threshold=-0.5)

        # Find the observations by label
        over_obs = next(o for o in result.observations if o.label == "over")
        under_obs = next(o for o in result.observations if o.label == "under")
        normal_obs = next(o for o in result.observations if o.label == "normal")

        self.assertEqual(over_obs.confidence, 1.0)
        self.assertEqual(under_obs.confidence, 0.0)
        self.assertEqual(normal_obs.confidence, 0.8)

    @patch("httpx.Client.post")
    def test_malformed_json_response(self, mock_post):
        """Test handling of invalid JSON from provider."""
        mock = MagicMock(spec=httpx.Response)
        mock.status_code = 200
        mock.text = "not valid json"
        mock.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("invalid response", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_missing_candidates_in_response(self, mock_post):
        """Test handling of response without candidates."""
        mock_post.return_value = self._make_mock_response(200, {"candidates": []})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("no candidates", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_missing_observations_array(self, mock_post):
        """Test handling of response without observations."""
        bad_text = json.dumps({"observations": "not an array"})
        mock_post.return_value = self._make_mock_response(200, {
            "candidates": [{"content": {"parts": [{"text": bad_text}]}}]
        })

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("observations", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_invalid_confidence_in_observation(self, mock_post):
        """Test handling of non-numeric confidence."""
        observations = [
            {"label": "test", "category": "crop", "confidence": "high", "description": "Test"},
        ]
        mock_post.return_value = self._make_mock_response(200, self._make_success_response(observations))

        analyzer = GeminiVisionAnalyzer(self.settings)
        result = analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        # Should skip invalid observation, result in NO_USABLE_INFORMATION
        self.assertEqual(result.status, VisionStatus.NO_USABLE_INFORMATION)

    @patch("httpx.Client.post")
    def test_invalid_category_in_observation(self, mock_post):
        """Test handling of unknown category."""
        observations = [
            {"label": "test", "category": "unknown_category", "confidence": 0.8, "description": "Test"},
        ]
        mock_post.return_value = self._make_mock_response(200, self._make_success_response(observations))

        analyzer = GeminiVisionAnalyzer(self.settings)
        result = analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        # Should skip invalid observation
        self.assertEqual(result.status, VisionStatus.NO_USABLE_INFORMATION)

    @patch("httpx.Client.post")
    def test_authentication_failure_401(self, mock_post):
        """Test 401 authentication error."""
        mock_post.return_value = self._make_mock_response(401, {"error": "Unauthorized"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("authentication", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_authentication_failure_403(self, mock_post):
        """Test 403 forbidden error."""
        mock_post.return_value = self._make_mock_response(403, {"error": "Forbidden"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("authorized", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_rate_limit_429(self, mock_post):
        """Test 429 rate limit error."""
        mock_post.return_value = self._make_mock_response(429, {"error": "Rate limited"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("rate limit", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_server_error_500(self, mock_post):
        """Test 500 server error."""
        mock_post.return_value = self._make_mock_response(500, {"error": "Internal error"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("unavailable", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_bad_request_400(self, mock_post):
        """Test 400 bad request error."""
        mock_post.return_value = self._make_mock_response(400, {"error": "Bad request"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(ImageValidationError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("invalid vision request", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_timeout_error(self, mock_post):
        """Test provider timeout."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerTimeoutError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("timeout", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_connection_error(self, mock_post):
        """Test connection error."""
        mock_post.side_effect = httpx.ConnectError("Connection failed")

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerTimeoutError) as cm:
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertIn("timeout", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_retry_on_transient_error(self, mock_post):
        """Test retry on transient error (500 then success)."""
        mock_post.side_effect = [
            self._make_mock_response(500, {"error": "Server error"}),
            self._make_mock_response(200, self._make_success_response()),
        ]

        analyzer = GeminiVisionAnalyzer(self.settings)
        result = analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertEqual(result.status, VisionStatus.SUCCESS)
        self.assertEqual(mock_post.call_count, 2)

    @patch("httpx.Client.post")
    def test_no_retry_on_auth_error(self, mock_post):
        """Test no retry on authentication error."""
        mock_post.return_value = self._make_mock_response(401, {"error": "Unauthorized"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError):
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertEqual(mock_post.call_count, 1)

    @patch("httpx.Client.post")
    def test_no_retry_on_invalid_request(self, mock_post):
        """Test no retry on 400 bad request."""
        mock_post.return_value = self._make_mock_response(400, {"error": "Bad request"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(ImageValidationError):
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        self.assertEqual(mock_post.call_count, 1)

    @patch("httpx.Client.post")
    def test_max_retries_exceeded(self, mock_post):
        """Test giving up after max retries."""
        mock_post.return_value = self._make_mock_response(500, {"error": "Server error"})

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(AnalyzerUnavailableError):
            analyzer.analyze(VALID_JPEG, content_type="image/jpeg")

        # Initial + 2 retries = 3 calls
        self.assertEqual(mock_post.call_count, 3)

    def test_oversized_image_rejected_before_provider_call(self):
        """Test that oversized images are rejected before calling provider."""
        oversized = b"x" * (10 * 1024 * 1024 + 1)
        validated = make_validated_image(oversized)

        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(ImageTooLargeError):
            analyzer.analyze(oversized, content_type="image/jpeg")

    def test_invalid_image_rejected_before_provider_call(self):
        """Test that invalid images are rejected before calling provider."""
        # Empty image
        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(ImageValidationError):
            analyzer.analyze(b"", content_type="image/jpeg")

    def test_unsupported_format_rejected_before_provider_call(self):
        """Test that unsupported formats are rejected."""
        from kisansathi.vision.models import UnsupportedFormatError
        analyzer = GeminiVisionAnalyzer(self.settings)
        with self.assertRaises(UnsupportedFormatError):
            analyzer.analyze(VALID_JPEG, content_type="image/gif")

    def test_provider_name_logged(self):
        """Test that provider name is accessible."""
        analyzer = GeminiVisionAnalyzer(self.settings)
        self.assertEqual(analyzer.provider_name, "gemini")

    def test_close_closes_client(self):
        """Test that close() closes the HTTP client."""
        analyzer = GeminiVisionAnalyzer(self.settings)
        analyzer.close()
        # Should not raise
        analyzer.close()


class TestFactoryFunction(unittest.TestCase):
    """Tests for create_vision_analyzer factory."""

    def test_fake_provider_when_no_api_key(self):
        """Test factory returns fake when no API key."""
        settings = make_settings(vision_api_key=None)
        from kisansathi.vision.providers import create_vision_analyzer
        analyzer = create_vision_analyzer(settings)
        self.assertIsInstance(analyzer, FakeVisionAnalyzer)

    def test_fake_provider_when_fake_configured(self):
        """Test factory returns fake when provider is fake."""
        settings = make_settings(vision_provider="fake", vision_api_key="some-key")
        from kisansathi.vision.providers import create_vision_analyzer
        analyzer = create_vision_analyzer(settings)
        self.assertIsInstance(analyzer, FakeVisionAnalyzer)

    def test_gemini_provider_when_configured(self):
        """Test factory returns Gemini when configured."""
        settings = make_settings(vision_provider="gemini", vision_api_key="test-key")
        from kisansathi.vision.providers import create_vision_analyzer
        analyzer = create_vision_analyzer(settings)
        self.assertIsInstance(analyzer, GeminiVisionAnalyzer)

    def test_unknown_provider_raises(self):
        """Test factory raises for unknown provider."""
        settings = make_settings(vision_provider="unknown", vision_api_key="test-key")
        from kisansathi.vision.providers import create_vision_analyzer
        with self.assertRaises(ValueError):
            create_vision_analyzer(settings)


if __name__ == "__main__":
    unittest.main()