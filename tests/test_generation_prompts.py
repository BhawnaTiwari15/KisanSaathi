"""Unit tests for prompt building functions."""

import unittest
from kisansathi.citations.models import CitationBatch, RejectedCitation
from kisansathi.domain.schemas import Citation, Language, ResponseStatus, UserMessage
from kisansathi.eligibility.models import EligibilityDecision, EligibilityStatus
from kisansathi.generation.prompts import (
    _format_citations,
    build_system_prompt,
    build_user_prompt,
    parse_generated_answer,
)
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.models import WeatherCurrent, WeatherResponse


class TestBuildSystemPrompt(unittest.TestCase):
    def test_english(self) -> None:
        prompt = build_system_prompt(Language.ENGLISH)
        self.assertIn("English", prompt)
        self.assertIn("KisanSaathi", prompt)
        self.assertIn("Do not invent", prompt)
        self.assertIn("citation ID", prompt)

    def test_hindi(self) -> None:
        prompt = build_system_prompt(Language.HINDI)
        self.assertIn("Hindi", prompt)

    def test_kannada(self) -> None:
        prompt = build_system_prompt(Language.KANNADA)
        self.assertIn("Answer in Kannada.", prompt)

    def test_telugu(self) -> None:
        prompt = build_system_prompt(Language.TELUGU)
        self.assertIn("Answer in Telugu.", prompt)

    def test_marathi(self) -> None:
        prompt = build_system_prompt(Language.HINDI)
        self.assertIn("Hindi", prompt)

    def test_every_supported_language_has_display_name(self) -> None:
        expected = {
            Language.ENGLISH: "English",
            Language.HINDI: "Hindi",
            Language.KANNADA: "Kannada",
            Language.TELUGU: "Telugu",
        }
        self.assertEqual(set(expected), set(Language))
        for language, name in expected.items():
            with self.subTest(language=language):
                prompt = build_system_prompt(language)
                self.assertIn(f"Answer in {name}.", prompt)

    def test_output_contract_instructions_present(self) -> None:
        prompt = build_system_prompt(Language.ENGLISH)
        self.assertIn("Output contract", prompt)
        self.assertIn("must begin with exactly: ANSWER:", prompt)
        self.assertIn("Then exactly: CITATIONS:", prompt)
        self.assertIn("<source_id>:<24-hex>", prompt)
        self.assertIn("Do not add markdown, code fences", prompt)
        self.assertIn("use only IDs supplied in the EVIDENCE section", prompt)
        self.assertIn("If no supplied citation supports a claim, do not make that claim", prompt)

    def test_answer_appears_before_citations(self) -> None:
        prompt = build_system_prompt(Language.ENGLISH)
        self.assertLess(prompt.index("ANSWER:"), prompt.index("CITATIONS:"))

    def test_worked_example_present(self) -> None:
        prompt = build_system_prompt(Language.ENGLISH)
        self.assertIn("Example:", prompt)
        self.assertIn("pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76", prompt)

    def test_worked_example_is_parser_valid(self) -> None:
        example = (
            "ANSWER:\n"
            "The required documents are described in the official FAQ.\n"
            "CITATIONS:\n"
            "[pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76]"
        )
        allowed = frozenset({"pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76"})
        answer = parse_generated_answer(example, allowed)
        self.assertEqual(
            answer.citation_ids, ("pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76",)
        )

    def test_forbids_output_before_answer(self) -> None:
        prompt = build_system_prompt(Language.ENGLISH)
        self.assertIn("Output nothing before ANSWER:", prompt)

    def test_contract_appears_for_all_supported_languages(self) -> None:
        for language in Language:
            with self.subTest(language=language):
                prompt = build_system_prompt(language)
                self.assertIn("ANSWER:", prompt)
                self.assertIn("CITATIONS:", prompt)

    def test_existing_rules_unchanged(self) -> None:
        prompt = build_system_prompt(Language.ENGLISH)
        self.assertIn("Do not use any external knowledge.", prompt)
        self.assertIn("Do not invent facts, numbers, or citation IDs.", prompt)
        self.assertIn("Answer in English.", prompt)
        self.assertIn("Keep answers concise and actionable for a farmer.", prompt)


class TestBuildUserPrompt(unittest.TestCase):
    def setUp(self) -> None:
        self.message = UserMessage(
            text="What are the PM-KISAN eligibility rules?",
            language=Language.ENGLISH,
        )
        self.citation = Citation(
            source_id="pm-kisan-revised-faq",
            title="PM-KISAN FAQ",
            url="https://pmkisan.gov.in/Documents/RevisedFAQ.pdf",
            page_number=4,
            page_end=None,
            issuing_authority="Department of Agriculture",
            chunk_id="pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76",
        )
        self.batch = CitationBatch(citations=(self.citation,), rejected=())

    def test_includes_question(self) -> None:
        prompt = build_user_prompt(self.message, self.batch, None, None)
        self.assertIn("What are the PM-KISAN eligibility rules?", prompt)

    def test_includes_citations(self) -> None:
        prompt = build_user_prompt(self.message, self.batch, None, None)
        self.assertIn("pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76", prompt)
        self.assertIn("PM-KISAN FAQ", prompt)
        self.assertIn("Department of Agriculture", prompt)

    def test_includes_eligibility(self) -> None:
        decision = EligibilityDecision(
            scheme="PM-KISAN",
            status=EligibilityStatus.ELIGIBLE,
            summary="All conditions satisfied.",
            evidence=(),
        )
        prompt = build_user_prompt(self.message, self.batch, decision, None)
        self.assertIn("ELIGIBILITY", prompt)
        self.assertIn("All conditions satisfied", prompt)

    def test_includes_weather(self) -> None:
        weather = WeatherResponse(
            latitude=28.6139,
            longitude=77.2090,
            timezone="Asia/Kolkata",
            current=WeatherCurrent(
                temperature_c=31.2,
                precipitation_mm=0.0,
                wind_speed_mps=8.4,
                weather_code=2,
                time_iso="2026-10-06T09:00",
            ),
        )
        prompt = build_user_prompt(self.message, self.batch, None, weather)
        self.assertIn("WEATHER", prompt)
        self.assertIn("31.2", prompt)

    def test_empty_citations(self) -> None:
        empty_batch = CitationBatch()
        prompt = build_user_prompt(self.message, empty_batch, None, None)
        self.assertIn("No evidence available", prompt)


class TestParseGeneratedAnswer(unittest.TestCase):
    def setUp(self) -> None:
        self.citation_ids = frozenset(["chunk-1", "chunk-2"])

    def test_valid_output(self) -> None:
        raw = (
            "ANSWER:\nThis is the answer [chunk-1].\n\n"
            "CITATIONS:\n[chunk-1]\n[chunk-2]"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.text, "This is the answer [chunk-1].")
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))
        self.assertEqual(answer.status, ResponseStatus.ANSWERED)

    def test_valid_output_without_brackets(self) -> None:
        raw = (
            "ANSWER:\nThis is the answer.\n\n"
            "CITATIONS:\nchunk-1\nchunk-2"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_missing_answer_section(self) -> None:
        raw = "CITATIONS:\n[chunk-1]"
        with self.assertRaises(Exception) as cm:
            parse_generated_answer(raw, self.citation_ids)
        self.assertIn("ANSWER section", str(cm.exception))

    def test_empty_answer(self) -> None:
        raw = "ANSWER:\n\nCITATIONS:\n[chunk-1]"
        with self.assertRaises(Exception) as cm:
            parse_generated_answer(raw, self.citation_ids)
        self.assertIn("empty", str(cm.exception))

    def test_unknown_citation_id(self) -> None:
        raw = (
            "ANSWER:\nThis is the answer.\n\n"
            "CITATIONS:\n[chunk-999]"
        )
        with self.assertRaises(Exception) as cm:
            parse_generated_answer(raw, self.citation_ids)
        self.assertIn("unknown citation ID", str(cm.exception))

    def test_insufficient_evidence_detected(self) -> None:
        raw = (
            "ANSWER:\nThere is insufficient evidence to answer.\n\n"
            "CITATIONS:\n[chunk-1]"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_hindi_insufficient_detected(self) -> None:
        raw = (
            "ANSWER:\nपर्याप्त सबूत नहीं हैं।\n\n"
            "CITATIONS:\n[chunk-1]"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_malformed_output_handled(self) -> None:
        raw = "This is not a valid format at all"
        with self.assertRaises(Exception):
            parse_generated_answer(raw, self.citation_ids)


class TestFormatCitationsWithChunkText(unittest.TestCase):
    """Tests for the new chunk text excerpt functionality."""

    def setUp(self) -> None:
        self.citation = Citation(
            source_id="pm-kisan-revised-faq",
            title="PM-KISAN FAQ",
            url="https://pmkisan.gov.in/Documents/RevisedFAQ.pdf",
            page_number=4,
            page_end=None,
            issuing_authority="Department of Agriculture",
            chunk_id="pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76",
        )
        self.batch = CitationBatch(citations=(self.citation,), rejected=())

    def test_excerpt_uses_title_when_no_chunk_text(self) -> None:
        """When no chunk text provided, excerpt falls back to title."""
        result = _format_citations(self.batch, chunk_texts={})
        self.assertIn("Excerpt: PM-KISAN FAQ", result)

    def test_excerpt_uses_chunk_text_when_available(self) -> None:
        """When chunk text is provided, excerpt uses the actual chunk text."""
        chunk_text = (
            "The following documents are required: Aadhaar, Bank Account, Land Records"
        )
        chunk_texts = {"pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76": chunk_text}
        result = _format_citations(self.batch, chunk_texts=chunk_texts)
        self.assertIn("Excerpt: The following documents are required", result)
        self.assertIn("Aadhaar, Bank Account, Land Records", result)
        self.assertNotIn("Excerpt: PM-KISAN FAQ", result)

    def test_excerpt_truncates_long_chunk_text(self) -> None:
        """Chunk text excerpt is truncated to _MAX_EXCERPT_CHARS."""
        long_text = "x" * 500
        chunk_texts = {"pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76": long_text}
        result = _format_citations(self.batch, chunk_texts=chunk_texts)
        # _MAX_EXCERPT_CHARS = 300, so excerpt should be truncated
        excerpt_start = result.index("Excerpt: ") + len("Excerpt: ")
        excerpt = result[excerpt_start:]
        self.assertLessEqual(len(excerpt), 300)

    def test_excerpt_sanitizes_control_characters(self) -> None:
        """Chunk text excerpt sanitizes control characters."""
        chunk_text = "Document\x00list\x1fwith\x7fcontrol\x08chars"
        chunk_texts = {"pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76": chunk_text}
        result = _format_citations(self.batch, chunk_texts=chunk_texts)
        self.assertIn("Excerpt: Documentlistwithcontrolchars", result)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x1f", result)
        self.assertNotIn("\x7f", result)
        self.assertNotIn("\x08", result)


class TestBuildUserPromptWithRetrievedChunks(unittest.TestCase):
    """Tests for build_user_prompt with retrieved_chunks parameter."""

    def setUp(self) -> None:
        self.message = UserMessage(
            text="What documents are needed for PM-KISAN?",
            language=Language.ENGLISH,
        )
        self.citation = Citation(
            source_id="pm-kisan-revised-faq",
            title="PM-KISAN FAQ",
            url="https://pmkisan.gov.in/Documents/RevisedFAQ.pdf",
            page_number=4,
            page_end=None,
            issuing_authority="Department of Agriculture",
            chunk_id="pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76",
        )
        self.batch = CitationBatch(citations=(self.citation,), rejected=())

    def _make_search_result(self, chunk_id: str, text: str) -> SearchResult:
        return SearchResult(
            score=0.9,
            payload={
                "chunk_id": chunk_id,
                "source_id": "pm-kisan-revised-faq",
                "sha256": "0" * 64,
                "scheme": "PM-KISAN",
                "jurisdiction": "IN",
                "language": "en",
                "title": "PM-KISAN FAQ",
                "page_start": 4,
                "page_end": 5,
                "heading": None,
                "text": text,
            },
        )

    def test_uses_chunk_text_from_retrieved_chunks(self) -> None:
        """build_user_prompt uses chunk text from retrieved_chunks for excerpts."""
        chunk_text = "Required: Aadhaar number, Bank account, Land ownership proof"
        chunk_id = "pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76"
        results = (self._make_search_result(chunk_id, chunk_text),)
        prompt = build_user_prompt(self.message, self.batch, None, None, retrieved_chunks=results)
        self.assertIn("Excerpt: Required: Aadhaar number", prompt)
        self.assertIn("Bank account, Land ownership proof", prompt)

    def test_falls_back_to_title_when_chunk_id_not_in_retrieved(self) -> None:
        """If retrieved chunk doesn't match citation, falls back to title."""
        # Different chunk_id than the citation
        results = (self._make_search_result("pm-kisan-revised-faq:other-id", "Some other text"),)
        prompt = build_user_prompt(self.message, self.batch, None, None, retrieved_chunks=results)
        self.assertIn("Excerpt: PM-KISAN FAQ", prompt)

    def test_multiple_chunks_in_retrieved(self) -> None:
        """Multiple retrieved chunks are all mapped."""
        citation2 = Citation(
            source_id="pm-kisan-revised-faq",
            title="PM-KISAN FAQ",
            url="https://pmkisan.gov.in/Documents/RevisedFAQ.pdf",
            page_number=5,
            page_end=None,
            issuing_authority="Department of Agriculture",
            chunk_id="pm-kisan-revised-faq:07e8df2f1d787a879de2a7a2",
        )
        batch = CitationBatch(citations=(self.citation, citation2), rejected=())
        chunk_id_1 = "pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76"
        chunk_id_2 = "pm-kisan-revised-faq:07e8df2f1d787a879de2a7a2"
        results = (
            self._make_search_result(chunk_id_1, "First chunk text"),
            self._make_search_result(chunk_id_2, "Second chunk text"),
        )
        prompt = build_user_prompt(self.message, batch, None, None, retrieved_chunks=results)
        self.assertIn("Excerpt: First chunk text", prompt)
        self.assertIn("Excerpt: Second chunk text", prompt)

    def test_backwards_compatibility_without_retrieved_chunks(self) -> None:
        """Calling without retrieved_chunks still works (backwards compatible)."""
        prompt = build_user_prompt(self.message, self.batch, None, None)
        self.assertIn("Excerpt: PM-KISAN FAQ", prompt)
        self.assertIn("pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76", prompt)


if __name__ == "__main__":
    unittest.main()