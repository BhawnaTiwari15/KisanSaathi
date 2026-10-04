"""Environment-backed settings for the KisanSaathi application."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import os

from kisansathi.domain.schemas import Language


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

        return cls(environment=environment, default_language=default_language)