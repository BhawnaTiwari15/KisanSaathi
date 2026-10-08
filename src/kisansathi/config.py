"""Environment-backed settings for the KisanSaathi application."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import os

from kisansathi.domain.schemas import Language

DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_QDRANT_COLLECTION_NAME = "kisansathi_chunks"
DEFAULT_QDRANT_STORAGE_PATH = "data/qdrant"
DEFAULT_BM25_STORAGE_PATH = "data/bm25.sqlite3"

# Vision provider defaults
DEFAULT_VISION_PROVIDER = "fake"
DEFAULT_VISION_MODEL_NAME = "gemini-1.5-flash-latest"
DEFAULT_VISION_TIMEOUT_SECONDS = 15.0
DEFAULT_VISION_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_VISION_MAX_RETRIES = 2
DEFAULT_VISION_RETRY_BACKOFF_BASE = 1.0
DEFAULT_VISION_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_VISION_TEMPERATURE = 0.0
DEFAULT_VISION_MAX_OUTPUT_TOKENS = 1024

DEFAULT_SOURCES_MANIFEST_PATH = "data/sources.json"


class AppEnvironment(StrEnum):
    """Supported application environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"

    @classmethod
    def parse(cls, value: str) -> "AppEnvironment":
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(f"Unsupported application environment: {value!r}") from error


class VisionProvider(StrEnum):
    """Supported vision analysis providers."""

    FAKE = "fake"
    GEMINI = "gemini"

    @classmethod
    def parse(cls, value: str) -> "VisionProvider":
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(f"Unsupported vision provider: {value!r}") from error


@dataclass(frozen=True, slots=True)
class Settings:
    """Settings read from process environment variables."""

    environment: AppEnvironment = AppEnvironment.DEVELOPMENT
    default_language: Language = Language.ENGLISH
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    qdrant_collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME
    qdrant_storage_path: str = DEFAULT_QDRANT_STORAGE_PATH
    bm25_storage_path: str = DEFAULT_BM25_STORAGE_PATH

    # Vision provider settings
    vision_provider: VisionProvider = VisionProvider.FAKE
    vision_model_name: str = DEFAULT_VISION_MODEL_NAME
    vision_api_key: str | None = None
    vision_timeout_seconds: float = DEFAULT_VISION_TIMEOUT_SECONDS
    vision_connect_timeout_seconds: float = DEFAULT_VISION_CONNECT_TIMEOUT_SECONDS
    vision_max_retries: int = DEFAULT_VISION_MAX_RETRIES
    vision_retry_backoff_base: float = DEFAULT_VISION_RETRY_BACKOFF_BASE
    vision_max_image_bytes: int = DEFAULT_VISION_MAX_IMAGE_BYTES
    vision_temperature: float = DEFAULT_VISION_TEMPERATURE
    vision_max_output_tokens: int = DEFAULT_VISION_MAX_OUTPUT_TOKENS
    sources_manifest_path: str | None = None

    def __post_init__(self) -> None:
        if not self.embedding_model_name.strip():
            raise ValueError("Embedding model name must not be empty")
        if not self.reranker_model_name.strip():
            raise ValueError("Reranker model name must not be empty")
        if (
            isinstance(self.embedding_dimension, bool)
            or not isinstance(self.embedding_dimension, int)
            or self.embedding_dimension <= 0
        ):
            raise ValueError("Embedding dimension must be a positive integer")
        if not self.qdrant_collection_name.strip():
            raise ValueError("Qdrant collection name must not be empty")
        if not self.qdrant_storage_path.strip():
            raise ValueError("Qdrant storage path must not be empty")
        if not self.bm25_storage_path.strip():
            raise ValueError("BM25 storage path must not be empty")
        if not self.vision_model_name.strip():
            raise ValueError("Vision model name must not be empty")
        if self.vision_timeout_seconds <= 0:
            raise ValueError("Vision timeout must be positive")
        if self.vision_connect_timeout_seconds <= 0:
            raise ValueError("Vision connect timeout must be positive")
        if self.vision_max_retries < 0:
            raise ValueError("Vision max retries must be non-negative")
        if self.vision_retry_backoff_base <= 0:
            raise ValueError("Vision retry backoff base must be positive")
        if self.vision_max_image_bytes <= 0:
            raise ValueError("Vision max image bytes must be positive")
        if not (0.0 <= self.vision_temperature <= 2.0):
            raise ValueError("Vision temperature must be between 0.0 and 2.0")
        if self.vision_max_output_tokens <= 0:
            raise ValueError("Vision max output tokens must be positive")
        if self.sources_manifest_path is not None and not self.sources_manifest_path.strip():
            raise ValueError("Sources manifest path must not be blank")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Create settings from the given mapping or the process environment."""
        values = os.environ if environ is None else environ
        environment = AppEnvironment.parse(
            values.get("KISANSAATHI_ENV", AppEnvironment.DEVELOPMENT)
        )
        language_value = values.get("KISANSAATHI_DEFAULT_LANGUAGE", Language.ENGLISH.value)
        try:
            default_language = Language(language_value.strip().lower())
        except ValueError as error:
            raise ValueError(
                f"Unsupported default language: {language_value!r}; expected 'en' or 'hi'"
            ) from error

        embedding_model_name = values.get(
            "KISANSAATHI_EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME
        ).strip()
        if not embedding_model_name:
            raise ValueError("Embedding model name must not be empty")
        reranker_model_name = values.get(
            "KISANSAATHI_RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL_NAME
        ).strip()
        if not reranker_model_name:
            raise ValueError("Reranker model name must not be empty")

        embedding_dimension_value = values.get(
            "KISANSAATHI_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)
        )
        try:
            embedding_dimension = int(embedding_dimension_value)
        except ValueError as error:
            raise ValueError("Embedding dimension must be a positive integer") from error
        qdrant_collection_name = values.get(
            "KISANSAATHI_QDRANT_COLLECTION_NAME", DEFAULT_QDRANT_COLLECTION_NAME
        ).strip()
        qdrant_storage_path = values.get(
            "KISANSAATHI_QDRANT_STORAGE_PATH", DEFAULT_QDRANT_STORAGE_PATH
        ).strip()
        bm25_storage_path = values.get(
            "KISANSAATHI_BM25_STORAGE_PATH", DEFAULT_BM25_STORAGE_PATH
        ).strip()

        vision_provider = VisionProvider.parse(
            values.get("KISANSAATHI_VISION_PROVIDER", DEFAULT_VISION_PROVIDER)
        )
        vision_model_name = values.get(
            "KISANSAATHI_VISION_MODEL_NAME", DEFAULT_VISION_MODEL_NAME
        ).strip()
        vision_api_key = values.get("KISANSAATHI_VISION_API_KEY")
        if vision_api_key is not None:
            vision_api_key = vision_api_key.strip() or None
        vision_timeout_seconds = float(
            values.get("KISANSAATHI_VISION_TIMEOUT_SECONDS", str(DEFAULT_VISION_TIMEOUT_SECONDS))
        )
        vision_connect_timeout_seconds = float(
            values.get(
                "KISANSAATHI_VISION_CONNECT_TIMEOUT_SECONDS", str(DEFAULT_VISION_CONNECT_TIMEOUT_SECONDS)
            )
        )
        vision_max_retries = int(
            values.get("KISANSAATHI_VISION_MAX_RETRIES", str(DEFAULT_VISION_MAX_RETRIES))
        )
        vision_retry_backoff_base = float(
            values.get(
                "KISANSAATHI_VISION_RETRY_BACKOFF_BASE", str(DEFAULT_VISION_RETRY_BACKOFF_BASE)
            )
        )
        vision_max_image_bytes = int(
            values.get("KISANSAATHI_VISION_MAX_IMAGE_BYTES", str(DEFAULT_VISION_MAX_IMAGE_BYTES))
        )
        vision_temperature = float(
            values.get("KISANSAATHI_VISION_TEMPERATURE", str(DEFAULT_VISION_TEMPERATURE))
        )
        vision_max_output_tokens = int(
            values.get("KISANSAATHI_VISION_MAX_OUTPUT_TOKENS", str(DEFAULT_VISION_MAX_OUTPUT_TOKENS))
        )

        sources_manifest_path = values.get("KISANSAATHI_SOURCES_MANIFEST")
        if sources_manifest_path is not None:
            sources_manifest_path = sources_manifest_path.strip() or None

        return cls(
            environment=environment,
            default_language=default_language,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
            embedding_dimension=embedding_dimension,
            qdrant_collection_name=qdrant_collection_name,
            qdrant_storage_path=qdrant_storage_path,
            bm25_storage_path=bm25_storage_path,
            vision_provider=vision_provider,
            vision_model_name=vision_model_name,
            vision_api_key=vision_api_key,
            vision_timeout_seconds=vision_timeout_seconds,
            vision_connect_timeout_seconds=vision_connect_timeout_seconds,
            vision_max_retries=vision_max_retries,
            vision_retry_backoff_base=vision_retry_backoff_base,
            vision_max_image_bytes=vision_max_image_bytes,
            vision_temperature=vision_temperature,
            vision_max_output_tokens=vision_max_output_tokens,
            sources_manifest_path=sources_manifest_path,
        )