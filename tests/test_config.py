import unittest

from kisansathi.config import AppEnvironment, Settings
from kisansathi.domain.schemas import Language


class SettingsTests(unittest.TestCase):
    def test_defaults_are_safe_and_local(self) -> None:
        settings = Settings.from_env({})

        self.assertEqual(settings.environment, AppEnvironment.DEVELOPMENT)
        self.assertEqual(settings.default_language, Language.ENGLISH)

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


if __name__ == "__main__":
    unittest.main()