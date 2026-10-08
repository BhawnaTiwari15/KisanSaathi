"""Single logging configuration helper for entry points.

Entry points (the Streamlit app and the CLI modules) call
``configure_logging`` once at startup. Log output goes to stderr, which the
service manager (systemd, a container runtime, or a terminal) collects.
Levels are read from ``KISANSAATHI_LOG_LEVEL`` and never from secrets.
"""

from __future__ import annotations

import logging
import os

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_LEVEL_NAMES = logging.getLevelNamesMapping()


def _normalize_level_name(value: str | None) -> str:
    if value is None:
        value = os.environ.get("KISANSAATHI_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    normalized = str(value).strip().upper()
    return normalized or DEFAULT_LOG_LEVEL


def resolve_log_level(value: str | None = None) -> int:
    """Return the logging level for a level name.

    Unknown or blank names fall back to ``INFO`` so a mistyped environment
    variable can never crash an entry point.
    """
    return _LEVEL_NAMES.get(_normalize_level_name(value)) or logging.INFO


def configure_logging(level: str | None = None) -> int:
    """Configure the root logger once and return the resolved level.

    Safe to call more than once: existing handlers are kept and only the
    root level is updated, so repeated entry-point wiring never duplicates
    output.
    """
    name = _normalize_level_name(level)
    resolved = _LEVEL_NAMES.get(name)
    if resolved is None:
        resolved = logging.INFO
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(resolved)
    else:
        logging.basicConfig(level=resolved, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    if _LEVEL_NAMES.get(name) is None:
        logging.getLogger(__name__).warning(
            "Unsupported log level %r; using %s", name, logging.getLevelName(resolved)
        )
    return resolved
