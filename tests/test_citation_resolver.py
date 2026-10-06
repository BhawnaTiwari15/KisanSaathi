"""Tests for citation resolution from retrieval and eligibility provenance.

Fixtures are fixed literals: a two-source registry and synthetic payloads. One class
resolves the real PM-KISAN eligibility evidence against the real tracked manifest.
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from dataclasses import dataclass

from kisansathi.citations.models import (
    ChunkSourceMismatchError,
    CitationBatch,
    CitationError,
    InvalidEvidenceError,
    InvalidPageError,
    RejectedCitation,
    SourceMetadataMismatchError,
    UnknownCitationIdError,
    UnknownSourceError,
)
from kisansathi.citations.registry import ManifestSourceRegistry
from kisansathi.citations.resolver import (
    CitationResolver,
    validate_referenced_citations,
)
from kisansathi.domain.schemas import Citation
from kisansathi.eligibility.models import EvidenceRef
from kisansathi.eligibility.pm_kisan_rules import PM_KISAN_RULES
from kisansathi.ingestion.models import ManifestEntry
from kisansathi.retrieval.vector_store import SearchResult

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = REPO_ROOT / "data" / "sources.json"

SOURCE_ID = "pm-kisan-revised-faq"
OTHER_SOURCE_ID = "mh-pdmc-notification-2026-27"
SHA = "95d4e2892b4b351e599bf903f8c6f89b20d4b74831eb8f1a0e61a2a66771ff86"
DIGEST = "33ff2db494c59ab1ca6c4f76"
CHUNK_ID = f"{SOURCE_ID}:{DIGEST}"
TITLE = "Example Scheme FAQ"
URL = "https://example.gov.in/faq.pdf"
AUTHORITY = "Example Ministry (portal-level attribution; issuer not established)"


def make_entry(**overrides: object) -> ManifestEntry:
    """Return a valid ManifestEntry, overriding any field under test."""
    values: dict[str, object] = {
        "source_id": SOURCE_ID,
        "scheme": "PM-KISAN",
        "jurisdiction": "IN",
        "language": "en",
        "issuing_authority": AUTHORITY,
        "source_url": URL,
        "title": TITLE,
        "accessed_at": "2026-10-05",
        "staging_path": "faq.pdf",
    }
    values.update(overrides)
    return ManifestEntry(**values)  # type: ignore[arg-type]


def make_registry(*entries: ManifestEntry) -> ManifestSourceRegistry:
    """Return a registry holding the primary source plus one unrelated source."""
    if not entries:
        entries = (
            make_entry(),
            make_entry(
                source_id=OTHER_SOURCE_ID,
                scheme="PM-RKVY-PDMC",
                jurisdiction="IN-MH",
                language="mr-en",
                title="NOTIFICATION-2026-27",
                source_url="https://example.gov.in/pdmc.pdf",
                issuing_authority="Directorate of Horticulture",
            ),
        )
    return ManifestSourceRegistry(entries)


def make_payload(**overrides: object) -> dict[str, object]:
    """Return a complete retrieval payload, overriding any field under test."""
    values: dict[str, object] = {
        "chunk_id": CHUNK_ID,
        "source_id": SOURCE_ID,
        "sha256": SHA,
        "scheme": "PM-KISAN",
        "jurisdiction": "IN",
        "language": "en",
        "title": TITLE,
        "page_start": 4,
        "page_end": 4,
        "heading": "18. Is there any restriction on Land Holding ?",
        "text": "Land holding is the sole criteria to avail the benefit under the Scheme",
    }
    values.update(overrides)
    return values


def make_search_result(**overrides: object) -> SearchResult:
    """Return a SearchResult wrapping a complete payload."""
    return SearchResult(score=0.87, payload=make_payload(**overrides))


def make_evidence(**overrides: object) -> EvidenceRef:
    """Return valid eligibility evidence, overriding any field under test."""
    values: dict[str, object] = {
        "source_id": SOURCE_ID,
        "sha256": SHA,
        "chunk_id": CHUNK_ID,
        "page_start": 4,
        "page_end": 4,
        "locator": "Revised FAQ, Q18",
    }
    values.update(overrides)
    return EvidenceRef(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StubEvidence:
    """Structural evidence that bypasses EvidenceRef's own validation.

    EvidenceRef already refuses foreign chunks and reversed page ranges, so exercising
    the resolver's defence-in-depth guards requires input that never survives that
    constructor. Field types are loose on purpose.
    """

    source_id: object = SOURCE_ID
    sha256: object = SHA
    chunk_id: object = CHUNK_ID
    page_start: object = 4
    page_end: object = 4


class _NoTextReadMapping(dict):
    """A payload mapping that fails loudly if source text is ever read."""

    def __getitem__(self, key: object) -> object:
        if key in {"text", "heading"}:
            raise AssertionError(f"citation layer must not read {key!r}")
        return super().__getitem__(key)


class SearchResultToCitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_valid_search_result_resolves_to_a_citation(self) -> None:
        citation = self.resolver.resolve_payload(make_search_result().payload)

        self.assertIsInstance(citation, Citation)
        self.assertEqual(citation.source_id, SOURCE_ID)
        self.assertEqual(citation.chunk_id, CHUNK_ID)

    def test_document_metadata_is_resolved_from_the_registry(self) -> None:
        citation = self.resolver.resolve_payload(make_payload())

        self.assertEqual(citation.title, TITLE)
        self.assertEqual(citation.issuing_authority, AUTHORITY)

    def test_source_url_comes_from_the_registry(self) -> None:
        citation = self.resolver.resolve_payload(make_payload())

        self.assertEqual(citation.url, URL)

    def test_single_page_span_leaves_page_end_unset(self) -> None:
        citation = self.resolver.resolve_payload(make_payload(page_start=4, page_end=4))

        self.assertEqual(citation.page_number, 4)
        self.assertIsNone(citation.page_end)
        self.assertEqual(citation.page_span, (4, 4))

    def test_multi_page_span_keeps_both_pages(self) -> None:
        citation = self.resolver.resolve_payload(make_payload(page_start=3, page_end=4))

        self.assertEqual(citation.page_number, 3)
        self.assertEqual(citation.page_end, 4)
        self.assertEqual(citation.page_span, (3, 4))

    def test_authority_caveat_survives_resolution(self) -> None:
        citation = self.resolver.resolve_payload(make_payload())

        self.assertIn("issuer not established", citation.issuing_authority)

    def test_resolver_requires_a_registry(self) -> None:
        with self.assertRaisesRegex(CitationError, "requires a SourceRegistry"):
            CitationResolver(object())

    def test_search_result_score_is_never_used(self) -> None:
        low = SearchResult(score=-99.0, payload=make_payload())
        high = SearchResult(score=1e9, payload=make_payload())

        self.assertEqual(
            self.resolver.resolve_payload(low.payload),
            self.resolver.resolve_payload(high.payload),
        )

    def test_citation_lands_in_an_assistant_response(self) -> None:
        from kisansathi.domain.schemas import AssistantResponse, Language, ResponseStatus

        citation = self.resolver.resolve_payload(make_payload())
        response = AssistantResponse(
            text="Land must be in the applicant's own name.",
            language=Language.ENGLISH,
            status=ResponseStatus.ANSWERED,
            citations=(citation,),
        )

        self.assertEqual(response.citations[0].page_number, 4)


class UnknownSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_unregistered_source_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(UnknownSourceError, "not registered"):
            self.resolver.resolve_payload(make_payload(source_id="unknown-source"))

    def test_unregistered_source_yields_no_partial_citation(self) -> None:
        with self.assertRaises(UnknownSourceError):
            self.resolver.resolve_payload(make_payload(source_id="unknown-source"))

    def test_blank_source_id_is_rejected(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=value):
                with self.assertRaises(InvalidEvidenceError):
                    self.resolver.resolve_payload(make_payload(source_id=value))

    def test_missing_source_id_is_rejected(self) -> None:
        payload = make_payload()
        del payload["source_id"]

        with self.assertRaisesRegex(InvalidEvidenceError, "source_id"):
            self.resolver.resolve_payload(payload)

    def test_each_missing_required_field_is_named(self) -> None:
        for field in ("chunk_id", "sha256", "page_start", "page_end"):
            with self.subTest(field=field):
                payload = make_payload()
                del payload[field]
                with self.assertRaisesRegex(InvalidEvidenceError, field):
                    self.resolver.resolve_payload(payload)

    def test_non_mapping_payload_is_rejected(self) -> None:
        for value in (None, "text", 5, [make_payload()]):
            with self.subTest(value=value):
                with self.assertRaises(InvalidEvidenceError):
                    self.resolver.resolve_payload(value)


class InvalidPageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_non_positive_pages_are_rejected(self) -> None:
        for page in (0, -1, -100):
            with self.subTest(page=page):
                with self.assertRaises(InvalidPageError):
                    self.resolver.resolve_payload(make_payload(page_start=page, page_end=page))

    def test_reversed_page_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidPageError, "must not be before"):
            self.resolver.resolve_payload(make_payload(page_start=7, page_end=3))

    def test_non_integer_pages_are_rejected(self) -> None:
        for page in ("4", 4.0, None, [4]):
            with self.subTest(page=page):
                with self.assertRaises(InvalidPageError):
                    self.resolver.resolve_payload(make_payload(page_start=page))

    def test_boolean_pages_are_rejected(self) -> None:
        with self.assertRaises(InvalidPageError):
            self.resolver.resolve_payload(make_payload(page_start=True, page_end=True))

    def test_invalid_page_end_is_rejected(self) -> None:
        with self.assertRaises(InvalidPageError):
            self.resolver.resolve_payload(make_payload(page_end="four"))


class ChunkSourceMismatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_chunk_from_another_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ChunkSourceMismatchError, "does not belong"):
            self.resolver.resolve_payload(
                make_payload(chunk_id=f"{OTHER_SOURCE_ID}:{DIGEST}")
            )

    def test_malformed_chunk_id_is_rejected(self) -> None:
        for chunk_id in ("no-prefix", f"{SOURCE_ID}:short", f"{SOURCE_ID}:" + "a" * 23, "A" * 30):
            with self.subTest(chunk_id=chunk_id):
                with self.assertRaises(ChunkSourceMismatchError):
                    self.resolver.resolve_payload(make_payload(chunk_id=chunk_id))

    def test_each_defect_is_detected_independently(self) -> None:
        """A foreign chunk is refused on source grounds, not on any co-occurring defect."""
        with self.assertRaises(ChunkSourceMismatchError):
            self.resolver.resolve_payload(
                make_payload(chunk_id=f"{OTHER_SOURCE_ID}:{DIGEST}", page_start=3, page_end=4)
            )

    def test_citation_schema_also_rejects_a_foreign_chunk(self) -> None:
        with self.assertRaisesRegex(ValueError, "must belong to its source_id"):
            Citation(
                source_id=SOURCE_ID,
                title=TITLE,
                url=URL,
                chunk_id=f"{OTHER_SOURCE_ID}:{DIGEST}",
            )


class SourceMetadataConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_stale_title_in_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(SourceMetadataMismatchError, "does not match"):
            self.resolver.resolve_payload(make_payload(title="Some Other Document"))

    def test_absent_title_skips_the_cross_check(self) -> None:
        payload = make_payload()
        del payload["title"]

        self.assertEqual(self.resolver.resolve_payload(payload).title, TITLE)

    def test_non_string_title_is_rejected(self) -> None:
        with self.assertRaises(InvalidEvidenceError):
            self.resolver.resolve_payload(make_payload(title=123))


class MalformedUrlTests(unittest.TestCase):
    def test_missing_url_is_refused_rather_than_fabricated(self) -> None:
        """A source with no registered URL cannot be cited at all; no placeholder exists."""
        resolver = CitationResolver(make_registry(make_entry()))

        with self.assertRaises(UnknownSourceError):
            resolver.resolve_payload(make_payload(source_id="unregistered"))

    def test_resolved_url_is_always_the_registered_url(self) -> None:
        resolver = CitationResolver(ManifestSourceRegistry.from_manifest(SOURCES_PATH))
        registry = resolver.registry

        for source_id in registry.source_ids():
            with self.subTest(source_id=source_id):
                entry = registry.get(source_id)
                payload = make_payload(
                    source_id=source_id,
                    chunk_id=f"{source_id}:{DIGEST}",
                    sha256="a" * 64,
                    page_start=1,
                    page_end=1,
                    title=entry.title,
                )
                self.assertEqual(resolver.resolve_payload(payload).url, entry.source_url)

    def test_registry_cannot_store_a_blank_url(self) -> None:
        with self.assertRaises(ValueError):
            make_entry(source_url="")


class EligibilityEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_valid_evidence_resolves_to_a_citation(self) -> None:
        citation = self.resolver.resolve_evidence(make_evidence())

        self.assertEqual(citation.source_id, SOURCE_ID)
        self.assertEqual(citation.chunk_id, CHUNK_ID)
        self.assertEqual(citation.page_number, 4)
        self.assertEqual(citation.url, URL)

    def test_evidence_and_payload_produce_the_same_citation(self) -> None:
        """Both provenance paths converge on one representation."""
        self.assertEqual(
            self.resolver.resolve_evidence(make_evidence()),
            self.resolver.resolve_payload(make_payload()),
        )

    def test_multi_page_evidence_is_cited_as_a_range(self) -> None:
        citation = self.resolver.resolve_evidence(make_evidence(page_start=3, page_end=4))

        self.assertEqual(citation.page_span, (3, 4))

    def test_evidence_for_an_unregistered_source_is_rejected(self) -> None:
        with self.assertRaises(UnknownSourceError):
            self.resolver.resolve_evidence(
                StubEvidence(source_id="unknown-source", chunk_id=f"unknown-source:{DIGEST}")
            )

    def test_evidence_resolves_against_the_matching_registered_source(self) -> None:
        """A second registered source must resolve to its own metadata, not the first."""
        citation = self.resolver.resolve_evidence(
            make_evidence(
                source_id=OTHER_SOURCE_ID,
                chunk_id=f"{OTHER_SOURCE_ID}:{DIGEST}",
                page_start=2,
                page_end=2,
            )
        )

        self.assertEqual(citation.source_id, OTHER_SOURCE_ID)
        self.assertEqual(citation.url, "https://example.gov.in/pdmc.pdf")
        self.assertEqual(citation.issuing_authority, "Directorate of Horticulture")
        self.assertEqual(citation.page_number, 2)

    def test_evidence_with_a_foreign_chunk_is_rejected(self) -> None:
        with self.assertRaises(ChunkSourceMismatchError):
            self.resolver.resolve_evidence(
                StubEvidence(chunk_id=f"{OTHER_SOURCE_ID}:{DIGEST}")
            )

    def test_evidence_with_a_malformed_chunk_is_rejected(self) -> None:
        for chunk_id in ("no-prefix", f"{SOURCE_ID}:short", f"{SOURCE_ID}:" + "a" * 23):
            with self.subTest(chunk_id=chunk_id):
                with self.assertRaises(ChunkSourceMismatchError):
                    self.resolver.resolve_evidence(StubEvidence(chunk_id=chunk_id))

    def test_evidence_with_an_invalid_page_is_rejected(self) -> None:
        with self.assertRaises(InvalidPageError):
            self.resolver.resolve_evidence(StubEvidence(page_start=5, page_end=2))

    def test_evidence_with_a_non_positive_page_is_rejected(self) -> None:
        for page in (0, -1, True):
            with self.subTest(page=page):
                with self.assertRaises(InvalidPageError):
                    self.resolver.resolve_evidence(
                        StubEvidence(page_start=page, page_end=page)
                    )

    def test_evidence_with_a_bad_sha256_is_rejected(self) -> None:
        for sha in ("abc", "A" * 64, "z" * 64, 12345):
            with self.subTest(sha=sha):
                with self.assertRaises(InvalidEvidenceError):
                    self.resolver.resolve_evidence(StubEvidence(sha256=sha))

    def test_evidence_with_blank_identity_is_rejected(self) -> None:
        for field in ("source_id", "chunk_id", "sha256"):
            with self.subTest(field=field):
                with self.assertRaises(InvalidEvidenceError):
                    self.resolver.resolve_evidence(StubEvidence(**{field: "  "}))

    def test_evidence_missing_a_required_attribute_is_rejected(self) -> None:
        class Partial:
            source_id = SOURCE_ID

        with self.assertRaises(InvalidEvidenceError):
            self.resolver.resolve_evidence(Partial())

    def test_evidence_ref_rejects_a_foreign_chunk_at_construction(self) -> None:
        """Eligibility already refuses this, so the resolver never sees it."""
        with self.assertRaises(ValueError):
            make_evidence(chunk_id=f"{OTHER_SOURCE_ID}:{DIGEST}")

    def test_evidence_ref_rejects_a_reversed_page_range_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            make_evidence(page_start=5, page_end=2)

    def test_real_pm_kisan_rules_resolve_against_the_real_manifest(self) -> None:
        resolver = CitationResolver(ManifestSourceRegistry.from_manifest(SOURCES_PATH))
        batch = resolver.resolve_evidence_batch(
            [reference for rule in PM_KISAN_RULES for reference in rule.evidence]
        )

        self.assertEqual(batch.rejected, ())
        self.assertEqual(len(batch.citations), len(PM_KISAN_RULES))
        for citation in batch.citations:
            with self.subTest(chunk_id=citation.chunk_id):
                self.assertEqual(citation.source_id, "pm-kisan-revised-faq")
                self.assertTrue(citation.url.startswith("https://pmkisan.gov.in/"))
                self.assertGreaterEqual(citation.page_number, 1)
        self.assertIn(
            "document-level issuer not established",
            batch.citations[0].issuing_authority,
        )


class CitationBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_batch_cites_valid_payloads_and_rejects_the_rest(self) -> None:
        batch = self.resolver.resolve_payloads(
            [
                make_payload(),
                make_payload(source_id="unknown-source"),
                make_payload(chunk_id=f"{OTHER_SOURCE_ID}:{DIGEST}"),
                make_payload(page_start=0),
            ]
        )

        self.assertEqual(len(batch.citations), 1)
        self.assertEqual(len(batch.rejected), 3)

    def test_batch_records_a_reason_for_every_refusal(self) -> None:
        batch = self.resolver.resolve_payloads([make_payload(source_id="unknown-source")])

        self.assertIn("not registered", batch.rejected[0].reason)
        self.assertEqual(batch.rejected[0].source_id, "unknown-source")

    def test_batch_is_empty_when_nothing_resolves(self) -> None:
        batch = self.resolver.resolve_payloads([make_payload(source_id="unknown-source")])

        self.assertEqual(batch.citations, ())
        self.assertEqual(len(batch.rejected), 1)

    def test_batch_is_empty_for_no_input(self) -> None:
        batch = self.resolver.resolve_payloads([])

        self.assertEqual(batch, CitationBatch())

    def test_batch_rejects_non_iterable_input(self) -> None:
        for value in ("payloads", 5, None):
            with self.subTest(value=value):
                with self.assertRaises(InvalidEvidenceError):
                    self.resolver.resolve_payloads(value)

    def test_batch_deduplicates_identical_chunks(self) -> None:
        batch = self.resolver.resolve_payloads([make_payload(), make_payload(), make_payload()])

        self.assertEqual(len(batch.citations), 1)
        self.assertEqual(batch.rejected, ())

    def test_batch_keeps_distinct_chunks(self) -> None:
        batch = self.resolver.resolve_payloads(
            [make_payload(), make_payload(chunk_id=f"{SOURCE_ID}:{'1' * 24}")]
        )

        self.assertEqual(len(batch.citations), 2)

    def test_one_chunk_id_yields_exactly_one_citation(self) -> None:
        """A chunk is content-addressed over its page span, so it cannot be cited twice."""
        batch = self.resolver.resolve_payloads(
            [make_payload(page_start=3, page_end=4), make_payload(page_start=4, page_end=4)]
        )

        self.assertEqual(len(batch.citations), 1)
        self.assertEqual(batch.citations[0].page_span, (3, 4))

    def test_batch_validates_its_member_types(self) -> None:
        with self.assertRaises(CitationError):
            CitationBatch(citations=("not-a-citation",))  # type: ignore[arg-type]

        with self.assertRaises(CitationError):
            CitationBatch(rejected=("not-a-rejection",))  # type: ignore[arg-type]

    def test_batch_validates_its_collection_types(self) -> None:
        stray = Citation(source_id="s", title="t", url="https://e.in")

        with self.assertRaises(CitationError):
            CitationBatch(citations=[stray])  # type: ignore[arg-type]

    def test_rejection_validates_its_fields(self) -> None:
        with self.assertRaises(CitationError):
            RejectedCitation(source_id="", chunk_id=None, reason="why")

        with self.assertRaises(CitationError):
            RejectedCitation(source_id="s", chunk_id=None, reason=" ")

    def test_batch_exposes_its_citation_ids(self) -> None:
        batch = self.resolver.resolve_payloads([make_payload()])

        self.assertEqual(batch.citation_ids, frozenset({CHUNK_ID}))
        self.assertEqual(batch.by_chunk_id()[CHUNK_ID].chunk_id, CHUNK_ID)


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())

    def test_repeated_resolution_is_identical(self) -> None:
        payload = make_payload(page_start=3, page_end=4)

        self.assertEqual(
            {self.resolver.resolve_payload(payload) for _ in range(50)},
            {self.resolver.resolve_payload(payload)},
        )

    def test_batch_generation_is_identical(self) -> None:
        payloads = [make_payload(), make_payload(page_start=3, page_end=4)]

        self.assertEqual(
            {self.resolver.resolve_payloads(payloads) for _ in range(50)},
            {self.resolver.resolve_payloads(payloads)},
        )

    def test_batch_preserves_first_seen_order(self) -> None:
        first = make_payload(chunk_id=f"{SOURCE_ID}:{'2' * 24}", page_start=3, page_end=4)
        second = make_payload(chunk_id=f"{SOURCE_ID}:{'3' * 24}", page_start=7, page_end=7)
        batch = self.resolver.resolve_payloads([first, second, first])

        self.assertEqual(
            [citation.page_number for citation in batch.citations],
            [3, 7],
        )

    def test_citation_is_frozen(self) -> None:
        citation = self.resolver.resolve_payload(make_payload())

        with self.assertRaises(AttributeError):
            citation.page_number = 99  # type: ignore[misc]

    def test_registry_order_does_not_change_citations(self) -> None:
        forward = CitationResolver(
            ManifestSourceRegistry((make_entry(), make_entry(source_id="other-source")))
        )
        reverse = CitationResolver(
            ManifestSourceRegistry((make_entry(source_id="other-source"), make_entry()))
        )

        self.assertEqual(
            forward.resolve_payload(make_payload()),
            reverse.resolve_payload(make_payload()),
        )


class SourceTextIsNeverReadTests(unittest.TestCase):
    def test_resolution_succeeds_without_reading_text_or_heading(self) -> None:
        resolver = CitationResolver(make_registry())

        self.assertEqual(
            resolver.resolve_payload(_NoTextReadMapping(make_payload())).chunk_id,
            CHUNK_ID,
        )

    def test_ocr_damage_is_never_repaired(self) -> None:
        resolver = CitationResolver(make_registry())
        damaged = "Micro land holdings, which are not cultrvable, are exclvded."
        payload = make_payload(text=damaged, title=TITLE)

        citation = resolver.resolve_payload(payload)

        self.assertNotIn("text", {field.name for field in Citation.__dataclass_fields__.values()})
        self.assertEqual(payload["text"], damaged)


class PostGenerationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = CitationResolver(make_registry())
        self.second_chunk_id = f"{SOURCE_ID}:{'1' * 24}"
        self.batch = self.resolver.resolve_payloads(
            [
                make_payload(page_start=3, page_end=4),
                make_payload(chunk_id=self.second_chunk_id, page_start=7, page_end=7),
            ]
        )

    def test_known_ids_resolve_to_their_citations(self) -> None:
        self.assertEqual(
            validate_referenced_citations([CHUNK_ID, self.second_chunk_id], self.batch),
            self.batch.citations,
        )

    def test_a_subset_of_ids_resolves_to_just_those_citations(self) -> None:
        self.assertEqual(
            validate_referenced_citations([CHUNK_ID], self.batch),
            (self.batch.citations[0],),
        )

    def test_unknown_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(UnknownCitationIdError, "unresolved citation id"):
            validate_referenced_citations([f"{SOURCE_ID}:{'0' * 24}"], self.batch)

    def test_blank_id_is_rejected(self) -> None:
        for identifier in ("", "   "):
            with self.subTest(identifier=identifier):
                with self.assertRaisesRegex(UnknownCitationIdError, "empty citation id"):
                    validate_referenced_citations([identifier], self.batch)

    def test_one_unknown_id_rejects_the_whole_answer(self) -> None:
        with self.assertRaises(UnknownCitationIdError):
            validate_referenced_citations([CHUNK_ID, "fabricated"], self.batch)

    def test_repeated_ids_are_deduplicated_in_order(self) -> None:
        resolved = validate_referenced_citations([CHUNK_ID, CHUNK_ID], self.batch)

        self.assertEqual(len(resolved), 1)

    def test_no_references_yields_no_citations(self) -> None:
        self.assertEqual(validate_referenced_citations([], self.batch), ())

    def test_ids_must_match_the_batch_exactly(self) -> None:
        with self.assertRaises(UnknownCitationIdError):
            validate_referenced_citations([SOURCE_ID], self.batch)

    def test_gate_requires_a_batch_and_an_iterable(self) -> None:
        with self.assertRaises(CitationError):
            validate_referenced_citations([CHUNK_ID], object())

        with self.assertRaises(CitationError):
            validate_referenced_citations(CHUNK_ID, self.batch)

    def test_gate_is_deterministic(self) -> None:
        self.assertEqual(
            validate_referenced_citations([CHUNK_ID], self.batch),
            validate_referenced_citations([CHUNK_ID], self.batch),
        )


class ImportIsolationTests(unittest.TestCase):
    def test_citations_import_neither_retrieval_nor_langgraph_nor_eligibility(self) -> None:
        source_root = REPO_ROOT / "src"
        code = """
import builtins, sys
original_import = builtins.__import__
FORBIDDEN = (
    'kisansathi.retrieval',
    'kisansathi.orchestration',
    'kisansathi.eligibility',
    'langgraph',
    'langchain_core',
)

def guarded_import(name, *args, **kwargs):
    if any(name == f or name.startswith(f + '.') for f in FORBIDDEN):
        raise AssertionError(f'forbidden import during module import: {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import kisansathi.citations
import kisansathi.citations.resolver
for name in FORBIDDEN:
    assert name not in sys.modules, f'{name} was imported'
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source_root)

        subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_evidence_protocol_matches_eligibility_evidence_without_importing_it(self) -> None:
        from kisansathi.citations.models import EvidenceLike

        self.assertIsInstance(make_evidence(), EvidenceLike)

        class NotEvidence:
            pass

        self.assertNotIsInstance(NotEvidence(), EvidenceLike)


if __name__ == "__main__":
    unittest.main()
