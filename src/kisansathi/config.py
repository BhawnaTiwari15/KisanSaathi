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

        return cls(
            environment=environment,
            default_language=default_language,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
            embedding_dimension=embedding_dimension,
            qdrant_collection_name=qdrant_collection_name,
            qdrant_storage_path=qdrant_storage_path,
            bm25_storage_path=bm25_storage_path,
        )