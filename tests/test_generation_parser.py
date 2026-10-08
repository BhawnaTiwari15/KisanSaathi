"""Unit tests for the answer parser logic (separate from prompt building)."""

import unittest
from kisansathi.domain.schemas import ResponseStatus
from kisansathi.generation.prompts import parse_generated_answer
from kisansathi.generation.models import MalformedOutputError, GroundingError


class TestParseGeneratedAnswer(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed_ids = frozenset(["chunk-1", "chunk-2", "pm-kisan-revised-faq:abc123"])

    def test_valid_basic_format(self) -> None:
        raw = (
            "ANSWER:\nThis is the answer.\n\n"
            "CITATIONS:\n[chunk-1]\n[chunk-2]"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.text, "This is the answer.")
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_valid_without_brackets(self) -> None:
        raw = (
            "ANSWER:\nAnswer text.\n\n"
            "CITATIONS:\nchunk-1\nchunk-2"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_mixed_brackets_and_no_brackets(self) -> None:
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[chunk-1]\nchunk-2"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_citation_ids_with_special_chars(self) -> None:
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[pm-kisan-revised-faq:abc123]"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("pm-kisan-revised-faq:abc123",))

    def test_empty_citations_section(self) -> None:
        raw = "ANSWER:\nAnswer with no citations.\n\nCITATIONS:\n"
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ())

    def test_no_citations_section(self) -> None:
        raw = "ANSWER:\nAnswer with no citations section."
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ())

    def test_missing_answer_section_raises(self) -> None:
        raw = "CITATIONS:\n[chunk-1]"
        with self.assertRaises(MalformedOutputError) as cm:
            parse_generated_answer(raw, self.allowed_ids)
        self.assertIn("ANSWER section", str(cm.exception))

    def test_empty_answer_raises(self) -> None:
        raw = "ANSWER:\n\nCITATIONS:\n[chunk-1]"
        with self.assertRaises(MalformedOutputError) as cm:
            parse_generated_answer(raw, self.allowed_ids)
        self.assertIn("empty", str(cm.exception).lower())

    def test_unknown_citation_id_raises(self) -> None:
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[unknown-chunk]"
        )
        with self.assertRaises(GroundingError) as cm:
            parse_generated_answer(raw, self.allowed_ids)
        self.assertIn("unknown", str(cm.exception).lower())

    def test_partial_unknown_citation_id_raises(self) -> None:
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[chunk-1]\n[unknown-chunk]"
        )
        with self.assertRaises(GroundingError):
            parse_generated_answer(raw, self.allowed_ids)

    def test_whitespace_handling(self) -> None:
        raw = (
            "  ANSWER:  \n  Answer with spaces.  \n\n  "
            "CITATIONS:  \n  [chunk-1]  \n  [chunk-2]  \n"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.text, "Answer with spaces.")
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_case_sensitivity_of_citation_ids(self) -> None:
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[CHUNK-1]"
        )
        # Citation IDs are case-sensitive
        with self.assertRaises(GroundingError):
            parse_generated_answer(raw, self.allowed_ids)

    def test_insufficient_evidence_detected_english(self) -> None:
        markers = [
            "insufficient evidence",
            "not enough evidence",
            "cannot determine",
            "unable to determine",
            "not available",
            "no evidence",
        ]
        for marker in markers:
            with self.subTest(marker=marker):
                raw = (
                    f"ANSWER:\nThere is {marker}.\n\n"
                    "CITATIONS:\n[chunk-1]"
                )
                answer = parse_generated_answer(raw, self.allowed_ids)
                self.assertEqual(answer.status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_insufficient_evidence_detected_hindi(self) -> None:
        markers = [
            "पर्याप्त नहीं",
            "काफी नहीं",
            "निर्धारित नहीं",
            "उपलब्ध नहीं",
        ]
        for marker in markers:
            with self.subTest(marker=marker):
                raw = (
                    f"ANSWER:\n{marker}।\n\n"
                    "CITATIONS:\n[chunk-1]"
                )
                answer = parse_generated_answer(raw, self.allowed_ids)
                self.assertEqual(answer.status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_answered_when_sufficient(self) -> None:
        raw = (
            "ANSWER:\nBased on the evidence, the answer is yes.\n\n"
            "CITATIONS:\n[chunk-1]"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.status, ResponseStatus.ANSWERED)

    def test_default_language_is_english(self) -> None:
        raw = "ANSWER:\nAnswer.\n\nCITATIONS:\n[chunk-1]"
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.language.name, "ENGLISH")

    def test_multiple_empty_lines_in_citations(self) -> None:
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n\n[chunk-1]\n\n[chunk-2]\n\n"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_citation_ids_with_colon_and_hex(self) -> None:
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[source-name:abcdef1234567890123456]\n[other:fedcba65432109876543210]"
        )
        allowed = frozenset(["source-name:abcdef1234567890123456", "other:fedcba65432109876543210"])
        answer = parse_generated_answer(raw, allowed)
        self.assertEqual(len(answer.citation_ids), 2)

    def test_citation_ids_with_trailing_period(self) -> None:
        """Citation IDs with trailing period after closing bracket are extracted correctly."""
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[chunk-1].\n[chunk-2]."
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_citation_ids_with_trailing_comma(self) -> None:
        """Citation IDs with trailing comma after closing bracket are extracted correctly."""
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[chunk-1],\n[chunk-2],"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_citation_ids_with_trailing_semicolon(self) -> None:
        """Citation IDs with trailing semicolon after closing bracket are extracted correctly."""
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[chunk-1];\n[chunk-2];"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_citation_ids_with_trailing_colon(self) -> None:
        """Citation IDs with trailing colon after closing bracket are extracted correctly."""
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[chunk-1]:\n[chunk-2]:"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_citation_ids_with_trailing_parenthesis(self) -> None:
        """Citation IDs with trailing parenthesis after closing bracket are extracted correctly."""
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[chunk-1])\n[chunk-2])"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_bare_citation_id_fallback(self) -> None:
        """Bare citation IDs (no brackets) are still accepted as fallback."""
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\nchunk-1\nchunk-2"
        )
        answer = parse_generated_answer(raw, self.allowed_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_invalid_citation_id_still_rejected(self) -> None:
        """Invalid citation IDs (not in allowed list) are still rejected."""
        raw = (
            "ANSWER:\nAnswer.\n\n"
            "CITATIONS:\n[unknown-chunk]"
        )
        with self.assertRaises(GroundingError) as cm:
            parse_generated_answer(raw, self.allowed_ids)
        self.assertIn("unknown", str(cm.exception).lower())