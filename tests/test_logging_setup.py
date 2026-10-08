"""Tests for the shared logging configuration."""

import logging
import os
import unittest
from unittest.mock import patch

from kisansathi import logging_setup
from kisansathi.logging_setup import configure_logging, resolve_log_level


class ResolveLogLevelTests(unittest.TestCase):
    def test_known_level_names_are_case_insensitive(self) -> None:
        self.assertEqual(resolve_log_level("debug"), logging.DEBUG)
        self.assertEqual(resolve_log_level("WARNING"), logging.WARNING)
        self.assertEqual(resolve_log_level(" Info "), logging.INFO)

    def test_unknown_level_falls_back_to_info(self) -> None:
        self.assertEqual(resolve_log_level("bogus"), logging.INFO)

    def test_blank_value_falls_back_to_info(self) -> None:
        self.assertEqual(resolve_log_level("   "), logging.INFO)

    def test_default_is_info(self) -> None:
        self.assertEqual(resolve_log_level(), logging.INFO)

    def test_reads_environment_when_no_value_given(self) -> None:
        with patch.dict(os.environ, {"KISANSAATHI_LOG_LEVEL": "error"}, clear=False):
            self.assertEqual(resolve_log_level(), logging.ERROR)


class ConfigureLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root_logger = logging.getLogger()
        self.saved_handlers = list(self.root_logger.handlers)
        self.saved_level = self.root_logger.level

    def tearDown(self) -> None:
        self.root_logger.handlers = self.saved_handlers
        self.root_logger.setLevel(self.saved_level)

    def test_configures_root_and_returns_applied_level(self) -> None:
        self.assertEqual(configure_logging("DEBUG"), logging.DEBUG)
        self.assertNotEqual(self.root_logger.level, logging.NOTSET)

    def test_does_not_duplicate_handlers_on_second_call(self) -> None:
        configure_logging("DEBUG")
        handler_count = len(self.root_logger.handlers)
        configure_logging("INFO")
        self.assertEqual(len(self.root_logger.handlers), handler_count)
        self.assertEqual(self.root_logger.level, logging.INFO)

    def test_unknown_level_logs_warning_and_uses_info(self) -> None:
        with self.assertLogs(logging_setup.__name__, level="WARNING"):
            applied = configure_logging("nonsense")
        self.assertEqual(applied, logging.INFO)
        self.assertEqual(self.root_logger.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()