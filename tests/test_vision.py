"""Comprehensive tests for the vision package."""

import unittest
from kisansathi.vision import (
    VisionResult,
    VisualObservation,
    VisionStatus,
    VisionError,
    ImageValidationError,
    AnalyzerUnavailableError,
    AnalyzerTimeoutError,
    UnsupportedFormatError,
    ImageTooLargeError,
    LowConfidenceError,
    VisionAnalyzer,
    validate_image_input,
    ValidatedImage,
    FakeVisionAnalyzer,
    make_fake_vision_analyzer,
    TrackingFakeVisionAnalyzer,
    make_tracking_fake_vision,
)
from kisansathi.vision.image import SUPPORTED_IMAGE_TYPES, MAX_IMAGE_SIZE_BYTES


# Valid test image data (minimal valid JPEG)
VALID_JPEG = (
    b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0C\x14\r\x0C\x0B\x0B\x0C\x19\x12\x13\x0F\x14\x1D\x1A"
    b"\x1F\x1E\x1D\x1A\x1C\x1C $.' \",#\x1C\x1C(7),01444\x1F'9=82<.342"
    b"\xFF\xC0\x00\x0B\x08\x00\x01\x00\x01\x01\x01\x11\x00\xFF\xD9"
)

# Valid test image data (minimal valid PNG - padded to > 100 bytes)
VALID_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd5\x8b\xd3\x00\x00\x00\x00IEND\xaeB`\x82"
    + b"\x00" * 40  # Pad to > 100 bytes
)

# Valid test image data (minimal valid WebP - padded to > 100 bytes)
VALID_WEBP = (
    b"RIFF\x2c\x00\x00\x00WEBPVP8 \x14\x00\x00\x00\x01\x00\x00\x01\x00"
    b"\x00\x9d\x01\x2a\x01\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    + b"\x00" * 50  # Pad to > 100 bytes
)

# Invalid/corrupted image data
INVALID_JPEG = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50  # Truncated JPEG
INVALID_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50  # Truncated PNG
CORRUPTED_IMAGE = b"not an image at all"
EMPTY_IMAGE = b""

# Oversized image (simulated)
OVERSIZED_IMAGE = b"\xFF\xD8\xFF\xE0" + b"\x00" * (MAX_IMAGE_SIZE_BYTES + 100)

# JPEG with mismatched content type
JPEG_AS_PNG = VALID_JPEG  # JPEG data but will be sent as image/png


class TestVisionModels(unittest.TestCase):
    """Tests for vision domain models."""

    def test_visual_observation_creation(self) -> None:
        obs = VisualObservation(
            label="tomato plant",
            category="crop",
            confidence=0.9,
            description="Healthy tomato plant",
        )
        self.assertEqual(obs.label, "tomato plant")
        self.assertEqual(obs.category, "crop")
        self.assertEqual(obs.confidence, 0.9)
        self.assertFalse(obs.is_low_confidence)
        self.assertFalse(obs.is_uncertain)

    def test_visual_observation_low_confidence(self) -> None:
        obs = VisualObservation(
            label="possible pest",
            category="pest",
            confidence=0.4,
            description="May indicate pest damage",
        )
        self.assertTrue(obs.is_low_confidence)
        self.assertTrue(obs.is_uncertain)

    def test_visual_observation_validation(self) -> None:
        with self.assertRaises(ValueError):
            VisualObservation(label="", category="crop", confidence=0.5, description="test")
        with self.assertRaises(ValueError):
            VisualObservation(label="test", category="crop", confidence=1.5, description="test")
        with self.assertRaises(ValueError):
            VisualObservation(label="test", category="crop", confidence=0.5, description="")

    def test_vision_result_success(self) -> None:
        obs = VisualObservation(
            label="tomato plant", category="crop", confidence=0.9, description="Healthy plant"
        )
        result = VisionResult(status=VisionStatus.SUCCESS, observations=(obs,))
        self.assertTrue(result.is_success)
        self.assertFalse(result.is_failure)
        self.assertFalse(result.should_abstain)
        self.assertFalse(result.should_clarify)

    def test_vision_result_failure_statuses(self) -> None:
        for status in (
            VisionStatus.INVALID_IMAGE,
            VisionStatus.UNSUPPORTED_FORMAT,
            VisionStatus.IMAGE_TOO_LARGE,
            VisionStatus.ANALYZER_UNAVAILABLE,
            VisionStatus.ANALYZER_TIMEOUT,
            VisionStatus.NO_USABLE_INFORMATION,
        ):
            result = VisionResult(status=status, observations=())
            self.assertTrue(result.should_abstain)
            self.assertFalse(result.is_success)

    def test_vision_result_clarify_statuses(self) -> None:
        for status in (VisionStatus.LOW_CONFIDENCE, VisionStatus.UNABLE_TO_ANALYZE):
            result = VisionResult(status=status, observations=())
            self.assertTrue(result.should_clarify)
            self.assertFalse(result.should_abstain)

    def test_vision_result_uncertain_observations(self) -> None:
        obs1 = VisualObservation(label="tomato", category="crop", confidence=0.9, description="Tomato plant")
        obs2 = VisualObservation(label="pest", category="pest", confidence=0.5, description="Possible pest")
        result = VisionResult(status=VisionStatus.SUCCESS, observations=(obs1, obs2))
        self.assertTrue(result.has_uncertain_observations)
        self.assertEqual(len(result.get_uncertain_observations()), 1)
        self.assertEqual(len(result.get_high_confidence_observations()), 1)

    def test_vision_result_disease_pest_detection(self) -> None:
        obs = VisualObservation(label="blight", category="disease", confidence=0.8, description="Leaf blight detected")
        result = VisionResult(status=VisionStatus.SUCCESS, observations=(obs,))
        self.assertTrue(result.has_disease_or_pest)

    def test_vision_result_crop_detection(self) -> None:
        obs = VisualObservation(label="tomato", category="crop", confidence=0.8, description="Tomato plant identified")
        result = VisionResult(status=VisionStatus.SUCCESS, observations=(obs,))
        self.assertTrue(result.has_crop_observation)

    def test_vision_result_format_for_retrieval(self) -> None:
        obs1 = VisualObservation(label="tomato", category="crop", confidence=0.9, description="Tomato plant")
        obs2 = VisualObservation(label="blight", category="disease", confidence=0.6, description="Possible blight")
        obs3 = VisualObservation(label="noise", category="other", confidence=0.3, description="Background noise")
        result = VisionResult(status=VisionStatus.SUCCESS, observations=(obs1, obs2, obs3))
        retrieval_text = result.format_for_retrieval()
        self.assertIn("tomato", retrieval_text)
        self.assertIn("blight", retrieval_text)
        self.assertNotIn("noise", retrieval_text)  # Below 0.5 threshold

    def test_vision_result_format_for_generation(self) -> None:
        obs1 = VisualObservation(label="tomato", category="crop", confidence=0.9, description="Tomato plant")
        obs2 = VisualObservation(label="pest", category="pest", confidence=0.5, description="Possible pest")
        result = VisionResult(status=VisionStatus.SUCCESS, observations=(obs1, obs2))
        gen_text = result.format_for_generation()
        self.assertIn("tomato", gen_text)
        self.assertIn("possibly pest", gen_text)  # Uncertain marked with "possibly"

    def test_vision_result_validation_errors(self) -> None:
        # SUCCESS with empty observations should fail
        with self.assertRaises(ValueError):
            VisionResult(status=VisionStatus.SUCCESS, observations=())
        # Failure with observations should fail
        with self.assertRaises(ValueError):
            VisionResult(status=VisionStatus.INVALID_IMAGE, observations=(
                VisualObservation(label="test", category="crop", confidence=0.5, description="test"),
            ))


class TestImageValidation(unittest.TestCase):
    """Tests for image validation."""

    def test_valid_jpeg(self) -> None:
        validated = validate_image_input(VALID_JPEG, "image/jpeg")
        self.assertIsInstance(validated, ValidatedImage)
        self.assertEqual(validated.content_type, "image/jpeg")
        self.assertEqual(validated.detected_format, "jpeg")
        self.assertIsNotNone(validated.width)
        self.assertIsNotNone(validated.height)

    def test_valid_png(self) -> None:
        validated = validate_image_input(VALID_PNG, "image/png")
        self.assertEqual(validated.content_type, "image/png")
        self.assertEqual(validated.detected_format, "png")

    def test_valid_webp(self) -> None:
        validated = validate_image_input(VALID_WEBP, "image/webp")
        self.assertEqual(validated.content_type, "image/webp")
        self.assertEqual(validated.detected_format, "webp")

    def test_empty_image_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_input(EMPTY_IMAGE, "image/jpeg")

    def test_too_small_image_rejected(self) -> None:
        tiny = b"\xFF\xD8\xFF"  # 3 bytes
        with self.assertRaises(ImageValidationError):
            validate_image_input(tiny, "image/jpeg")

    def test_oversized_image_rejected(self) -> None:
        with self.assertRaises(ImageTooLargeError):
            validate_image_input(OVERSIZED_IMAGE, "image/jpeg")

    def test_invalid_content_type_rejected(self) -> None:
        with self.assertRaises(UnsupportedFormatError):
            validate_image_input(VALID_JPEG, "image/gif")

    def test_missing_content_type_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_input(VALID_JPEG, "")

    def test_corrupted_jpeg_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_input(INVALID_JPEG, "image/jpeg")

    def test_corrupted_png_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_input(INVALID_PNG, "image/png")

    def test_non_image_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_input(CORRUPTED_IMAGE, "image/jpeg")

    def test_mime_mismatch_rejected(self) -> None:
        # JPEG data sent as PNG
        with self.assertRaises(ImageValidationError):
            validate_image_input(VALID_JPEG, "image/png")

    def test_png_sent_as_jpeg_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_input(VALID_PNG, "image/jpeg")

    def test_case_insensitive_content_type(self) -> None:
        validated = validate_image_input(VALID_JPEG, "IMAGE/JPEG")
        self.assertEqual(validated.content_type, "image/jpeg")

    def test_content_type_with_charset(self) -> None:
        validated = validate_image_input(VALID_JPEG, "image/jpeg; charset=binary")
        self.assertEqual(validated.content_type, "image/jpeg")

    def test_dimension_validation(self) -> None:
        # The test images are 1x1, should pass
        validated = validate_image_input(VALID_JPEG, "image/jpeg")
        self.assertEqual(validated.width, 1)
        self.assertEqual(validated.height, 1)


class TestFakeVisionAnalyzer(unittest.TestCase):
    """Tests for fake vision analyzer."""

    def test_default_success(self) -> None:
        fake = FakeVisionAnalyzer()
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.SUCCESS)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(len(fake.calls), 1)

    def test_configured_error(self) -> None:
        fake = FakeVisionAnalyzer(result=AnalyzerUnavailableError("service down"))
        with self.assertRaises(AnalyzerUnavailableError):
            fake.analyze(VALID_JPEG, content_type="image/jpeg")

    def test_make_fake_success(self) -> None:
        fake = make_fake_vision_analyzer(status="success", confidence=0.8)
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.SUCCESS)
        self.assertEqual(result.confidence_overall, 0.8)

    def test_make_fake_low_confidence(self) -> None:
        fake = make_fake_vision_analyzer(status="low_confidence")
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.LOW_CONFIDENCE)
        self.assertEqual(result.observations, ())

    def test_make_fake_unable_to_analyze(self) -> None:
        fake = make_fake_vision_analyzer(status="unable_to_analyze")
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.UNABLE_TO_ANALYZE)

    def test_make_fake_analyzer_unavailable(self) -> None:
        fake = make_fake_vision_analyzer(status="analyzer_unavailable")
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.ANALYZER_UNAVAILABLE)

    def test_make_fake_no_usable_information(self) -> None:
        fake = make_fake_vision_analyzer(status="no_usable_information")
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.NO_USABLE_INFORMATION)

    def test_make_fake_custom_observations(self) -> None:
        obs = (
            VisualObservation(label="wheat", category="crop", confidence=0.85, description="Wheat field"),
            VisualObservation(label="rust", category="disease", confidence=0.7, description="Leaf rust"),
        )
        fake = make_fake_vision_analyzer(observations=obs)
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(len(result.observations), 2)
        self.assertTrue(result.has_disease_or_pest)
        self.assertTrue(result.has_crop_observation)

    def test_set_result_dynamic(self) -> None:
        fake = FakeVisionAnalyzer()
        fake.set_result(
            VisionResult(
                status=VisionStatus.SUCCESS,
                observations=(
                    VisualObservation(label="new", category="crop", confidence=0.95, description="New observation"),
                ),
            )
        )
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.observations[0].label, "new")


class TestTrackingFakeVisionAnalyzer(unittest.TestCase):
    """Tests for tracking fake vision analyzer."""

    def test_default_result(self) -> None:
        fake = make_tracking_fake_vision()
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.SUCCESS)

    def test_override_by_size(self) -> None:
        fake = make_tracking_fake_vision()
        fake.set_result(
            "100",
            VisionResult(
                status=VisionStatus.LOW_CONFIDENCE,
                observations=(),
            ),
        )
        # 100 bytes -> LOW_CONFIDENCE
        small_image = b"\xFF\xD8\xFF" + b"\x00" * 97
        result = fake.analyze(small_image, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.LOW_CONFIDENCE)

        # Different size -> default
        result = fake.analyze(VALID_JPEG, content_type="image/jpeg")
        self.assertEqual(result.status, VisionStatus.SUCCESS)

    def test_calls_tracked(self) -> None:
        fake = make_tracking_fake_vision()
        fake.analyze(VALID_JPEG, content_type="image/jpeg")
        fake.analyze(VALID_PNG, content_type="image/png")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["content_type"], "image/jpeg")
        self.assertEqual(fake.calls[1]["content_type"], "image/png")


class TestVisionIntegrationHelpers(unittest.TestCase):
    """Tests for vision result integration helpers."""

    def test_should_abstain_mapping(self) -> None:
        """Test that failure statuses correctly map to abstain behavior."""
        abstain_statuses = [
            VisionStatus.INVALID_IMAGE,
            VisionStatus.UNSUPPORTED_FORMAT,
            VisionStatus.IMAGE_TOO_LARGE,
            VisionStatus.ANALYZER_UNAVAILABLE,
            VisionStatus.ANALYZER_TIMEOUT,
            VisionStatus.NO_USABLE_INFORMATION,
        ]
        for status in abstain_statuses:
            result = VisionResult(status=status, observations=())
            self.assertTrue(result.should_abstain, f"{status} should abstain")

    def test_should_clarify_mapping(self) -> None:
        """Test that uncertain statuses correctly map to clarify behavior."""
        clarify_statuses = [VisionStatus.LOW_CONFIDENCE, VisionStatus.UNABLE_TO_ANALYZE]
        for status in clarify_statuses:
            result = VisionResult(status=status, observations=())
            self.assertTrue(result.should_clarify, f"{status} should clarify")

    def test_success_with_uncertain_observations(self) -> None:
        """SUCCESS with low-confidence observations should not abstain but should flag uncertainty."""
        obs = VisualObservation(label="possible pest", category="pest", confidence=0.4, description="Possible pest detected")
        result = VisionResult(status=VisionStatus.SUCCESS, observations=(obs,))
        self.assertFalse(result.should_abstain)
        self.assertTrue(result.has_uncertain_observations)


if __name__ == "__main__":
    unittest.main()