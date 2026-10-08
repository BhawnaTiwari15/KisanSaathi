"""Unit tests for the real Gemini text-generation provider (mocked HTTP).

These tests never call the real Gemini API. They use a mocked
``httpx.Client.post`` to cover successful parsing, provider errors, malformed
responses, timeouts, retries, missing configuration, secret leakage, and the
exact request contract (grounded prompts, temperature, max tokens, header).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx

from kisansathi.citations.models import CitationBatch
from kisansathi.config import LLMProvider, Settings
from kisansathi.domain.schemas import Citation, Language, UserMessage
from kisansathi.generation import DefaultAnswerGenerator
from kisansathi.generation.fake import FakeLLMClient, make_fake_generation_context
from kisansathi.generation.gemini import GeminiTextClient
from kisansathi.generation.models import (
    GenerationContext,
    GroundingError,
    LLMClient,
    LLMError,
    MalformedOutputError,
)
from kisansathi.generation.prompts import build_system_prompt, build_user_prompt
from kisansathi.ui.composition_root import _default_answer_generator, build_application_service


def make_settings(**overrides) -> Settings:
    """Create test settings with a fake API key by default."""
    defaults = {
        "llm_provider": "gemini",
        "llm_model_name": "gemini-1.5-flash-latest",
        "llm_api_key": "test-api-key",
        "llm_timeout_seconds": 15.0,
        "llm_connect_timeout_seconds": 5.0,
        "llm_max_retries": 2,
        "llm_temperature": 0.0,
        "llm_max_tokens": 512,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_mock_response(
    status_code: int, json_data: dict | None = None, text: str = ""
) -> MagicMock:
    """Create a mock httpx.Response."""
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    if json_data is not None:
        mock.json.return_value = json_data
    mock.text = text or (json.dumps(json_data) if json_data else "")
    return mock


def _make_success_response(raw_text: str) -> dict:
    """Create a successful Gemini generateContent payload."""
    return {"candidates": [{"content": {"parts": [{"text": raw_text}]}}]}


class TestGeminiTextClient(unittest.TestCase):
    """Tests for GeminiTextClient with mocked HTTP."""

    def setUp(self) -> None:
        self.settings = make_settings()

    def test_implements_llm_client_protocol(self) -> None:
        client = GeminiTextClient(self.settings)
        self.assertIsInstance(client, LLMClient)

    @patch("httpx.Client.post")
    def test_successful_parse_returns_text(self, mock_post) -> None:
        raw = "ANSWER:\nThe farmer is eligible.\nCITATIONS:\n[chunk-1]"
        mock_post.return_value = _make_mock_response(200, _make_success_response(raw))

        client = GeminiTextClient(self.settings)
        result = client.generate(system_prompt="system", user_prompt="user")

        self.assertEqual(result, raw)

    @patch("httpx.Client.post")
    def test_request_contains_grounded_prompts_and_config(self, mock_post) -> None:
        """The request carries the grounded prompts verbatim plus config."""
        mock_post.return_value = _make_mock_response(200, _make_success_response("ok"))

        context = make_fake_generation_context()
        system_prompt = build_system_prompt(context.message.language or Language.ENGLISH)
        user_prompt = build_user_prompt(
            context.message,
            context.citations,
            context.eligibility_decision,
            context.weather,
        )

        client = GeminiTextClient(self.settings)
        client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.25,
            max_tokens=700,
        )

        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"], {"x-goog-api-key": "test-api-key"})
        # The key travels in the header, never in the URL or query params
        self.assertNotIn("params", kwargs)
        self.assertNotIn("test-api-key", args[0])
        self.assertTrue(args[0].endswith(":generateContent"))

        payload = kwargs["json"]
        parts = payload["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"], system_prompt)
        self.assertEqual(parts[1]["text"], user_prompt)
        self.assertEqual(payload["generationConfig"]["temperature"], 0.25)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 700)

    @patch("httpx.Client.post")
    def test_api_key_in_header_not_query_string(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(200, _make_success_response("ok"))

        client = GeminiTextClient(self.settings)
        client.generate(system_prompt="s", user_prompt="u")

        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"], {"x-goog-api-key": "test-api-key"})
        self.assertNotIn("params", kwargs)
        self.assertNotIn("test-api-key", args[0])

    @patch("httpx.Client.post")
    def test_authentication_failure_401(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(401, {"error": "Unauthorized"})

        client = GeminiTextClient(self.settings)
        with self.assertRaises(LLMError) as cm:
            client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("authentication", str(cm.exception).lower())
        self.assertEqual(mock_post.call_count, 1)

    @patch("httpx.Client.post")
    def test_rate_limit_429(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(429, {"error": "Rate limited"})

        client = GeminiTextClient(self.settings)
        with patch("time.sleep"):
            with self.assertRaises(LLMError) as cm:
                client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("rate limit", str(cm.exception).lower())
        self.assertEqual(mock_post.call_count, 3)

    @patch("httpx.Client.post")
    def test_no_retry_on_bad_request_400(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(400, {"error": "Bad request"})

        client = GeminiTextClient(self.settings)
        with self.assertRaises(LLMError) as cm:
            client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("invalid llm request", str(cm.exception).lower())
        self.assertEqual(mock_post.call_count, 1)

    @patch("httpx.Client.post")
    def test_invalid_json_response(self, mock_post) -> None:
        mock = MagicMock(spec=httpx.Response)
        mock.status_code = 200
        mock.text = "not json"
        mock.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_post.return_value = mock

        client = GeminiTextClient(self.settings)
        with self.assertRaises(LLMError) as cm:
            client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("invalid response", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_empty_candidates_in_response(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(200, {"candidates": []})

        client = GeminiTextClient(self.settings)
        with self.assertRaises(LLMError) as cm:
            client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("candidates", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_empty_text_part(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(
            200, {"candidates": [{"content": {"parts": [{"text": ""}]}}]}
        )

        client = GeminiTextClient(self.settings)
        with self.assertRaises(LLMError) as cm:
            client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("invalid response", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_missing_parts_in_response(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(
            200, {"candidates": [{"content": {"role": "model"}}]}
        )

        client = GeminiTextClient(self.settings)
        with self.assertRaises(LLMError) as cm:
            client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("parts", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_timeout_error(self, mock_post) -> None:
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        client = GeminiTextClient(self.settings)
        with patch("time.sleep"):
            with self.assertRaises(LLMError) as cm:
                client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("timeout", str(cm.exception).lower())
        self.assertEqual(mock_post.call_count, 3)

    @patch("httpx.Client.post")
    def test_connection_error(self, mock_post) -> None:
        mock_post.side_effect = httpx.ConnectError("Connection failed")

        client = GeminiTextClient(self.settings)
        with patch("time.sleep"):
            with self.assertRaises(LLMError) as cm:
                client.generate(system_prompt="s", user_prompt="u")

        self.assertIn("timeout", str(cm.exception).lower())

    @patch("httpx.Client.post")
    def test_retry_on_transient_error(self, mock_post) -> None:
        mock_post.side_effect = [
            _make_mock_response(500, {"error": "Server error"}),
            _make_mock_response(200, _make_success_response("ok")),
        ]

        client = GeminiTextClient(self.settings)
        with patch("time.sleep"):
            result = client.generate(system_prompt="s", user_prompt="u")

        self.assertEqual(result, "ok")
        self.assertEqual(mock_post.call_count, 2)

    @patch("httpx.Client.post")
    def test_max_retries_exceeded(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(500, {"error": "Server error"})

        client = GeminiTextClient(self.settings)
        with patch("time.sleep"):
            with self.assertRaises(LLMError):
                client.generate(system_prompt="s", user_prompt="u")

        # Initial attempt + 2 retries
        self.assertEqual(mock_post.call_count, 3)

    def test_missing_api_key_raises_on_construction(self) -> None:
        with self.assertRaises(ValueError):
            GeminiTextClient(make_settings(llm_api_key=None))

    @patch("httpx.Client.post")
    def test_no_secret_leakage_in_error(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(401, {"error": "Unauthorized"})

        client = GeminiTextClient(self.settings)
        with self.assertRaises(LLMError) as cm:
            client.generate(system_prompt="s", user_prompt="u")

        self.assertNotIn("test-api-key", str(cm.exception))

    @patch("httpx.Client.post")
    def test_no_secret_leakage_in_logs(self, mock_post) -> None:
        mock_post.return_value = _make_mock_response(500, {"error": "Server error"})

        client = GeminiTextClient(self.settings)
        with patch("time.sleep"):
            with self.assertLogs("kisansathi.generation.gemini", level="WARNING") as captured:
                with self.assertRaises(LLMError):
                    client.generate(system_prompt="s", user_prompt="u")

        joined = "\n".join(captured.output)
        self.assertNotIn("test-api-key", joined)

    def test_provider_name(self) -> None:
        client = GeminiTextClient(self.settings)
        self.assertEqual(client.provider_name, "gemini")
        client.close()

    def test_close_closes_client(self) -> None:
        client = GeminiTextClient(self.settings)
        client.close()
        client.close()

    def test_context_manager_closes_client(self) -> None:
        with GeminiTextClient(self.settings) as client:
            self.assertEqual(client.provider_name, "gemini")


class TestLLMSettings(unittest.TestCase):
    """Tests for the LLM configuration fields."""

    def test_defaults_are_safe(self) -> None:
        settings = Settings.from_env({})

        self.assertEqual(settings.llm_provider, LLMProvider.FAKE)
        self.assertIsNone(settings.llm_api_key)
        self.assertEqual(settings.llm_model_name, "gemini-3.5-flash-lite")
        self.assertEqual(settings.llm_temperature, 0.0)
        self.assertEqual(settings.llm_max_tokens, 512)

    def test_reads_llm_settings_from_env(self) -> None:
        settings = Settings.from_env(
            {
                "KISANSAATHI_LLM_PROVIDER": "gemini",
                "KISANSAATHI_LLM_API_KEY": "env-key",
                "KISANSAATHI_LLM_MODEL_NAME": "test-model",
                "KISANSAATHI_LLM_TIMEOUT_SECONDS": "30.5",
                "KISANSAATHI_LLM_CONNECT_TIMEOUT_SECONDS": "10.5",
                "KISANSAATHI_LLM_MAX_RETRIES": "3",
                "KISANSAATHI_LLM_TEMPERATURE": "0.4",
                "KISANSAATHI_LLM_MAX_TOKENS": "800",
            }
        )

        self.assertEqual(settings.llm_provider, LLMProvider.GEMINI)
        self.assertEqual(settings.llm_api_key, "env-key")
        self.assertEqual(settings.llm_model_name, "test-model")
        self.assertEqual(settings.llm_timeout_seconds, 30.5)
        self.assertEqual(settings.llm_connect_timeout_seconds, 10.5)
        self.assertEqual(settings.llm_max_retries, 3)
        self.assertEqual(settings.llm_temperature, 0.4)
        self.assertEqual(settings.llm_max_tokens, 800)

    def test_blank_api_key_becomes_none(self) -> None:
        settings = Settings.from_env({"KISANSAATHI_LLM_API_KEY": "   "})
        self.assertIsNone(settings.llm_api_key)

    def test_rejects_unknown_llm_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            Settings.from_env({"KISANSAATHI_LLM_PROVIDER": "openai"})

    def test_rejects_invalid_llm_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM temperature"):
            Settings.from_env({"KISANSAATHI_LLM_TEMPERATURE": "3"})

    def test_rejects_negative_llm_max_retries(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM max retries"):
            Settings.from_env({"KISANSAATHI_LLM_MAX_RETRIES": "-1"})


class _StubRetriever:
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[object, ...]:
        return ()


class TestAnswerGeneratorWiring(unittest.TestCase):
    """Tests for composition-root wiring of the real answer generator."""

    def test_missing_key_wires_none_not_fake(self) -> None:
        settings = make_settings(llm_api_key=None)
        self.assertIsNone(_default_answer_generator(settings))

    def test_fake_provider_wires_none(self) -> None:
        settings = make_settings(llm_provider="fake", llm_api_key="test-key")
        self.assertIsNone(_default_answer_generator(settings))

    def test_gemini_configured_builds_default_generator(self) -> None:
        generator = _default_answer_generator(make_settings())
        self.assertIsInstance(generator, DefaultAnswerGenerator)

    def test_application_builds_without_api_key(self) -> None:
        with patch("kisansathi.ui.composition_root.build_graph") as mock_graph:
            build_application_service(settings=Settings(), retriever=_StubRetriever())

        self.assertIsNone(mock_graph.call_args.kwargs["answer_generator"])

    def test_application_builds_with_key_uses_real_generator(self) -> None:
        with patch("kisansathi.ui.composition_root.build_graph") as mock_graph:
            build_application_service(settings=make_settings(), retriever=_StubRetriever())

        generator = mock_graph.call_args.kwargs["answer_generator"]
        self.assertIsInstance(generator, DefaultAnswerGenerator)


ALLOWED_CHUNK_ID = "pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76"
UNKNOWN_CHUNK_ID = "pm-kisan-revised-faq:deadbeeffeedfacecafed00d"


def _citation_context() -> GenerationContext:
    """Context whose citation batch allowlists exactly one chunk id."""
    citation = Citation(
        source_id="pm-kisan-revised-faq",
        title="PM-KISAN FAQ",
        url="https://example.com/pm-kisan",
        page_number=1,
        chunk_id=ALLOWED_CHUNK_ID,
    )
    return GenerationContext(
        message=UserMessage(
            text="What documents are needed to apply for PM-KISAN?",
            language=Language.ENGLISH,
        ),
        citations=CitationBatch(citations=(citation,)),
        eligibility_decision=None,
        weather=None,
        vision_result=None,
    )


class TestRejectionDiagnosticLogging(unittest.TestCase):
    """Diagnostic logging must distinguish malformed vs grounding rejections.

    These tests assert the generator still raises the original exception (no
    behavior change) and that the safe metadata logged never includes the raw
    model response text or any secret-like content.
    """

    GENERATOR_LOGGER = "kisansathi.generation.generator"

    def test_missing_answer_marker_logs_malformed_diagnostic(self) -> None:
        raw = "The farmer needs documents but the ANSWER marker is missing."
        generator = DefaultAnswerGenerator(FakeLLMClient(response=raw))

        with self.assertLogs(self.GENERATOR_LOGGER, level="WARNING") as captured:
            with self.assertRaises(MalformedOutputError):
                generator.generate(_citation_context())

        log_text = " ".join(captured.output)
        self.assertIn("exception=MalformedOutputError", log_text)
        self.assertIn("answer_marker=False", log_text)
        self.assertIn("citations_marker=False", log_text)
        self.assertIn("citation_ids=0", log_text)
        self.assertIn("allowed_ids=1", log_text)

    def test_empty_answer_section_logs_malformed_diagnostic(self) -> None:
        raw = "ANSWER:\n\nCITATIONS:\n"
        generator = DefaultAnswerGenerator(FakeLLMClient(response=raw))

        with self.assertLogs(self.GENERATOR_LOGGER, level="WARNING") as captured:
            with self.assertRaises(MalformedOutputError):
                generator.generate(_citation_context())

        log_text = " ".join(captured.output)
        self.assertIn("exception=MalformedOutputError", log_text)
        self.assertIn("answer_marker=True", log_text)
        self.assertIn("answer_empty=True", log_text)
        self.assertIn("citations_marker=True", log_text)
        self.assertIn("citation_ids=0", log_text)
        self.assertIn("allowed_ids=1", log_text)

    def test_unknown_citation_logs_grounding_diagnostic(self) -> None:
        raw = (
            "ANSWER:\nThe farmer needs documents.\nCITATIONS:\n"
            f"[{UNKNOWN_CHUNK_ID}]"
        )
        generator = DefaultAnswerGenerator(FakeLLMClient(response=raw))

        with self.assertLogs(self.GENERATOR_LOGGER, level="WARNING") as captured:
            with self.assertRaises(GroundingError):
                generator.generate(_citation_context())

        log_text = " ".join(captured.output)
        self.assertIn("exception=GroundingError", log_text)
        self.assertIn("answer_marker=True", log_text)
        self.assertIn("answer_empty=False", log_text)
        self.assertIn("citations_marker=True", log_text)
        self.assertIn("citation_ids=1", log_text)
        self.assertIn("allowed_ids=1", log_text)

    def test_diagnostic_log_never_leaks_raw_output_or_secrets(self) -> None:
        secret_snippet = "AIzaFAKE0SECRETOCTETS"
        raw = (
            "ANSWER:\nThe farmer needs documents containing " + secret_snippet + ".\n"
            f"CITATIONS:\n[{UNKNOWN_CHUNK_ID}]"
        )
        generator = DefaultAnswerGenerator(FakeLLMClient(response=raw))

        with self.assertLogs(self.GENERATOR_LOGGER, level="WARNING") as captured:
            with self.assertRaises(GroundingError):
                generator.generate(_citation_context())

        log_text = " ".join(captured.output)
        self.assertNotIn("farmer needs documents", log_text)
        self.assertNotIn(secret_snippet, log_text)


if __name__ == "__main__":
    unittest.main()