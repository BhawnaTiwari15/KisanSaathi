"""Gemini text-generation provider implementing the ``LLMClient`` protocol.

Mirrors the HTTP provider patterns used by the real Gemini vision provider:
``generateContent`` endpoint construction, ``x-goog-api-key`` header, read/write
timeouts, retry of transient failures only, and provider error mapping.

This is a pure text client. It transmits the already-grounded system and user
prompts verbatim and returns the raw model text; it never performs retrieval
and never invents citations. Citation grounding and output validation are the
responsibility of ``DefaultAnswerGenerator`` / ``parse_generated_answer``.
"""

import json
import logging
import time

import httpx

from kisansathi.config import Settings
from kisansathi.generation.models import LLMError

logger = logging.getLogger(__name__)

# Gemini REST endpoint for generateContent (same host as the vision provider).
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Exponential backoff base between transient-failure retries (seconds).
_RETRY_BACKOFF_BASE = 1.0


class ProviderError(Exception):
    """Base class for provider errors (mirrors the vision provider base)."""

    def __init__(
        self, message: str, *, retryable: bool = False, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class AuthenticationError(ProviderError):
    """Provider authentication failed (401, 403)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message, retryable=False, status_code=status_code)


class RateLimitError(ProviderError):
    """Provider rate limit exceeded (429)."""

    def __init__(self, message: str, status_code: int = 429) -> None:
        super().__init__(message, retryable=True, status_code=status_code)


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True, status_code=None)


class ProviderUnavailableError(ProviderError):
    """Provider service unavailable (5xx)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message, retryable=True, status_code=status_code)


class InvalidRequestError(ProviderError):
    """Invalid request to provider (400)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message, retryable=False, status_code=status_code)


class MalformedResponseError(ProviderError):
    """Provider returned malformed or empty content."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, status_code=None)


class GeminiTextClient:
    """Gemini text-generation client implementing the ``LLMClient`` protocol.

    The client is intentionally unaware of prompt content: it sends the system
    and user prompts it is given (assembled and grounded by
    ``DefaultAnswerGenerator``) and returns the raw model text. It performs no
    retrieval and never fabricates citations.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("Gemini API key is required for GeminiTextClient")
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model_name
        self._max_retries = settings.llm_max_retries
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=settings.llm_connect_timeout_seconds,
                read=settings.llm_timeout_seconds,
                write=settings.llm_timeout_seconds,
                pool=settings.llm_timeout_seconds,
            ),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )

    @property
    def provider_name(self) -> str:
        """Provider name for logging and telemetry."""
        return "gemini"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Send the grounded prompts to Gemini and return the raw model text.

        Retries transient failures (rate limits, timeouts, 5xx) up to the
        configured maximum; authentication, bad-request, and malformed
        responses are never retried. Raises ``LLMError`` on failure — the
        message never contains the API key.
        """
        retry_count = 0
        while retry_count <= self._max_retries:
            try:
                return self._call_provider(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except ProviderError as error:
                if not error.retryable or retry_count >= self._max_retries:
                    raise self._map_error(error) from error
                retry_count += 1
                self._wait_before_retry(retry_count)
                logger.warning(
                    "Gemini text provider error (retry %d/%d): %s",
                    retry_count,
                    self._max_retries,
                    error,
                )
            except httpx.TimeoutException as error:
                mapped = ProviderTimeoutError(f"Provider request timed out: {error}")
                if retry_count >= self._max_retries:
                    raise self._map_error(mapped) from error
                retry_count += 1
                self._wait_before_retry(retry_count)
                logger.warning(
                    "Gemini text provider timeout (retry %d/%d): %s",
                    retry_count,
                    self._max_retries,
                    error,
                )
            except httpx.HTTPError as error:
                mapped = ProviderUnavailableError(f"Provider request failed: {error}", 500)
                if retry_count >= self._max_retries:
                    raise self._map_error(mapped) from error
                retry_count += 1
                self._wait_before_retry(retry_count)
                logger.warning(
                    "Gemini text provider request error (retry %d/%d): %s",
                    retry_count,
                    self._max_retries,
                    error,
                )

        raise self._map_error(ProviderUnavailableError("Provider request failed", 500))

    def _call_provider(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """POST the grounded prompts to Gemini and return the raw text."""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": system_prompt},
                        {"text": user_prompt},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        url = f"{_GEMINI_BASE_URL}/{self._model}:generateContent"
        headers = {"x-goog-api-key": self._api_key}

        try:
            response = self._client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError(f"Gemini request timed out: {error}") from error
        except httpx.ConnectError as error:
            raise ProviderTimeoutError(f"Gemini connection failed: {error}") from error

        if response.status_code == 401:
            raise AuthenticationError("Invalid or missing API key", 401)
        if response.status_code == 403:
            raise AuthenticationError("API key not authorized for this model", 403)
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded", 429)
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Gemini service error: {response.status_code}", response.status_code
            )
        if response.status_code == 400:
            raise InvalidRequestError("Bad request to Gemini", 400)
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"Gemini error {response.status_code}", response.status_code
            )

        try:
            data = response.json()
        except json.JSONDecodeError as error:
            raise MalformedResponseError(f"Gemini returned invalid JSON: {error}") from error

        return self._parse_text(data)

    def _parse_text(self, data: object) -> str:
        """Extract the generated text from a Gemini ``candidates`` payload."""
        try:
            candidates = data.get("candidates") if isinstance(data, dict) else None
            if not candidates:
                raise MalformedResponseError("No candidates in Gemini response")

            candidate = candidates[0]
            content = candidate.get("content") if isinstance(candidate, dict) else None
            if not content:
                raise MalformedResponseError("No content in Gemini candidate")

            parts = content.get("parts") if isinstance(content, dict) else None
            if not parts:
                raise MalformedResponseError("No parts in Gemini content")

            first_part = parts[0]
            text_part = first_part.get("text") if isinstance(first_part, dict) else None
            if not text_part or not str(text_part).strip():
                raise MalformedResponseError("No text in Gemini response part")

            return str(text_part)
        except MalformedResponseError:
            raise
        except Exception as error:
            raise MalformedResponseError(f"Failed to parse Gemini response: {error}") from error

    def _map_error(self, error: ProviderError) -> LLMError:
        """Map provider errors to the domain ``LLMError`` without leaking secrets."""
        if isinstance(error, AuthenticationError):
            message = "LLM provider authentication failed"
        elif isinstance(error, RateLimitError):
            message = "LLM provider rate limited"
        elif isinstance(error, ProviderTimeoutError):
            message = "LLM provider timeout"
        elif isinstance(error, ProviderUnavailableError):
            message = f"LLM provider unavailable: {error}"
        elif isinstance(error, InvalidRequestError):
            message = f"Invalid LLM request: {error}"
        else:
            message = f"LLM provider returned an invalid response: {error}"
        return LLMError(message)

    def _wait_before_retry(self, attempt: int) -> None:
        """Wait with exponential backoff before retrying."""
        time.sleep(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "GeminiTextClient":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()