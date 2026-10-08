import unittest

from kisansathi.config import (
    DEFAULT_BM25_STORAGE_PATH,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_RERANKER_MODEL_NAME,
    DEFAULT_QDRANT_COLLECTION_NAME,
    DEFAULT_QDRANT_STORAGE_PATH,
    AppEnvironment,
    Settings,
)
from kisansathi.domain.schemas import Language


class SettingsTests(unittest.TestCase):
    def test_defaults_are_safe_and_local(self) -> None:
        settings = Settings.from_env({})

        self.assertEqual(settings.environment, AppEnvironment.DEVELOPMENT)
        self.assertEqual(settings.default_language, Language.ENGLISH)
        self.assertEqual(settings.embedding_model_name, DEFAULT_EMBEDDING_MODEL_NAME)
        self.assertEqual(settings.reranker_model_name, DEFAULT_RERANKER_MODEL_NAME)
        self.assertEqual(settings.embedding_dimension, DEFAULT_EMBEDDING_DIMENSION)
        self.assertEqual(settings.qdrant_collection_name, DEFAULT_QDRANT_COLLECTION_NAME)
        self.assertEqual(settings.qdrant_storage_path, DEFAULT_QDRANT_STORAGE_PATH)
        self.assertEqual(settings.bm25_storage_path, DEFAULT_BM25_STORAGE_PATH)

    def test_reads_supported_environment_values(self) -> None:
        settings = Settings.from_env(
            {
                "KISANSAATHI_ENV": "test",
                "KISANSAATHI_DEFAULT_LANGUAGE": "hi",
            }
        )

        self.assertEqual(settings.environment, AppEnvironment.TEST)
        self.assertEqual(settings.default_language, Language.HINDI)

    def test_rejects_unknown_environment(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported application environment"):
            Settings.from_env({"KISANSAATHI_ENV": "staging"})

    def test_rejects_unknown_language(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported default language"):
            Settings.from_env({"KISANSAATHI_DEFAULT_LANGUAGE": "fr"})

    def test_reads_embedding_model_name(self) -> None:
        settings = Settings.from_env({"KISANSAATHI_EMBEDDING_MODEL_NAME": "test/model"})

        self.assertEqual(settings.embedding_model_name, "test/model")

    def test_rejects_blank_embedding_model_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Embedding model name must not be empty"):
            Settings.from_env({"KISANSAATHI_EMBEDDING_MODEL_NAME": "  "})

    def test_reads_reranker_model_name(self) -> None:
        settings = Settings.from_env(
            {"KISANSAATHI_RERANKER_MODEL_NAME": "test/reranker"}
        )

        self.assertEqual(settings.reranker_model_name, "test/reranker")

    def test_rejects_blank_reranker_model_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reranker model name must not be empty"):
            Settings.from_env({"KISANSAATHI_RERANKER_MODEL_NAME": "  "})

    def test_reads_vector_store_configuration(self) -> None:
        settings = Settings.from_env(
            {
                "KISANSAATHI_EMBEDDING_DIMENSION": "768",
                "KISANSAATHI_QDRANT_COLLECTION_NAME": "test_chunks",
                "KISANSAATHI_QDRANT_STORAGE_PATH": "tmp/qdrant",
            }
        )

        self.assertEqual(settings.embedding_dimension, 768)
        self.assertEqual(settings.qdrant_collection_name, "test_chunks")
        self.assertEqual(settings.qdrant_storage_path, "tmp/qdrant")

    def test_reads_bm25_storage_path(self) -> None:
        settings = Settings.from_env({"KISANSAATHI_BM25_STORAGE_PATH": "tmp/bm25.sqlite3"})

        self.assertEqual(settings.bm25_storage_path, "tmp/bm25.sqlite3")

    def test_rejects_invalid_embedding_dimension(self) -> None:
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "Embedding dimension must be a positive integer"
            ):
                Settings.from_env({"KISANSAATHI_EMBEDDING_DIMENSION": value})

    def test_sources_manifest_path_defaults_to_none(self) -> None:
        settings = Settings.from_env({})

        self.assertIsNone(settings.sources_manifest_path)

    def test_reads_sources_manifest_path(self) -> None:
        settings = Settings.from_env(
            {"KISANSAATHI_SOURCES_MANIFEST": "/app/data/sources.json"}
        )

        self.assertEqual(settings.sources_manifest_path, "/app/data/sources.json")

    def test_blank_sources_manifest_path_becomes_none(self) -> None:
        settings = Settings.from_env({"KISANSAATHI_SOURCES_MANIFEST": "   "})

        self.assertIsNone(settings.sources_manifest_path)

    def test_rejects_blank_direct_sources_manifest_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "Sources manifest path must not be blank"):
            Settings(sources_manifest_path="  ")


if __name__ == "__main__":
    unittest.main()