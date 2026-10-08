"""Tests for the composition root's source-manifest override wiring."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kisansathi.config import Settings
from kisansathi.ui.composition_root import (
    _default_citation_resolver,
    build_application_service,
)


class _StubRetriever:
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[object, ...]:
        return ()


def _manifest_with_source(source_id: str = "tmp-source") -> dict[str, object]:
    return {
        "sources": [
            {
                "source_id": source_id,
                "scheme": "PM-KISAN",
                "jurisdiction": "IN",
                "language": "en",
                "issuing_authority": "Test authority",
                "source_url": "https://example.com/test.pdf",
                "title": "Test source",
                "accessed_at": "2026-10-08",
                "staging_path": "test.pdf",
                "publication_date": None,
                "publication_year": None,
                "effective_from": None,
                "effective_to": None,
            }
        ]
    }


class CitationResolverDefaultsTests(unittest.TestCase):
    def test_default_resolver_loads_the_tracked_manifest(self) -> None:
        resolver = _default_citation_resolver()
        self.assertIsNotNone(resolver)
        self.assertEqual(len(resolver.registry), 2)

    def test_invalid_override_path_disables_resolver(self) -> None:
        with patch("kisansathi.ui.composition_root.logger") as mock_logger:
            resolver = _default_citation_resolver(Path("missing-manifest.json"))
        self.assertIsNone(resolver)
        mock_logger.warning.assert_called_once()

    def test_uses_default_manifest_when_no_override(self) -> None:
        with (
            patch(
                "kisansathi.ui.composition_root._default_citation_resolver",
                return_value=object(),
            ) as mock_resolver,
            patch("kisansathi.ui.composition_root.build_graph") as mock_graph,
        ):
            build_application_service(retriever=_StubRetriever())

        mock_resolver.assert_called_once_with(None)
        passed_resolver = mock_graph.call_args.kwargs["citation_resolver"]
        self.assertEqual(passed_resolver, mock_resolver.return_value)

    def test_manifest_override_from_settings_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "sources.json"
            manifest.write_text(json.dumps(_manifest_with_source()), encoding="utf-8")
            settings = Settings(sources_manifest_path=str(manifest))
            with (
                patch(
                    "kisansathi.ui.composition_root._default_citation_resolver",
                    return_value=object(),
                ) as mock_resolver,
                patch("kisansathi.ui.composition_root.build_graph") as mock_graph,
            ):
                build_application_service(settings, retriever=_StubRetriever())

            mock_resolver.assert_called_once_with(manifest)
            passed_resolver = mock_graph.call_args.kwargs["citation_resolver"]
            self.assertEqual(passed_resolver, mock_resolver.return_value)


if __name__ == "__main__":
    unittest.main()