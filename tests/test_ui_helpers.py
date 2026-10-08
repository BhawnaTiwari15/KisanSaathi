"""Deterministic tests for the pure Streamlit presentation helpers.

These cover ``kisansathi.ui.helpers`` only: no Streamlit import, no network,
no model loading, and no I/O. The helper module itself must stay importable
without the ``ui`` extra installed, which is what keeps the presentation
layer out of the application's dependency surface.
"""

import unittest

from kisansathi.domain.schemas import Citation, Language, ResponseStatus
from kisansathi.eligibility.models import FACT_NAMES, EligibilityRequest
from kisansathi.ui.helpers import (
    FACT_LABELS,
    FACT_SELECTIONS,
    LANGUAGE_LABELS,
    SAFE_ERROR_MESSAGE,
    build_eligibility_request,
    build_request_state,
    fact_selection_to_bool,
    image_content_type_for,
    invoke_graph_safely,
    parse_location,
    render_citation_text,
    status_footer,
    status_label,
    validate_text,
)

QUESTION = "Which scheme am I eligible for?"


def make_citation(**overrides: object) -> Citation:
    """Return a valid Citation, overriding any field under test."""
    values: dict[str, object] = {
        "source_id": "pm-kisan-revised-faq",
        "title": "PM-KISAN Revised FAQ",
        "url": "https://example.gov.in/pm-kisan",
        "page_number": 4,
        "page_end": 6,
        "issuing_authority": "Ministry of Agriculture",
    }
    values.update(overrides)
    return Citation(**values)  # type: ignore[arg-type]


class StubGraph:
    """Minimal graph stand-in: returns, raises, or returns a non-mapping."""

    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.states: list[object] = []

    def invoke(self, state: object) -> object:
        self.states.append(state)
        if self._error is not None:
            raise self._error
        return self._result


class ValidateTextTests(unittest.TestCase):
    def test_non_blank_text_is_valid(self) -> None:
        self.assertTrue(validate_text("rain today?"))

    def test_surrounding_whitespace_is_ignored(self) -> None:
        self.assertTrue(validate_text("  rain today?  "))

    def test_blank_text_is_invalid(self) -> None:
        self.assertFalse(validate_text("   "))
        self.assertFalse(validate_text(""))


class ParseLocationTests(unittest.TestCase):
    def test_valid_pair_is_returned(self) -> None:
        self.assertEqual(parse_location("28.6139", "77.2090"), (28.6139, 77.209))

    def test_surrounding_whitespace_is_ignored(self) -> None:
        self.assertEqual(parse_location(" 28.6 ", " 77.2 "), (28.6, 77.2))

    def test_partial_input_is_rejected(self) -> None:
        self.assertIsNone(parse_location("28.6", ""))
        self.assertIsNone(parse_location("", "77.2"))
        self.assertIsNone(parse_location("   ", "   "))

    def test_non_numeric_input_is_rejected(self) -> None:
        self.assertIsNone(parse_location("north", "east"))
        self.assertIsNone(parse_location("28.6abc", "77.2"))

    def test_out_of_range_input_is_rejected(self) -> None:
        self.assertIsNone(parse_location("91", "77.2"))
        self.assertIsNone(parse_location("-90.1", "77.2"))
        self.assertIsNone(parse_location("28.6", "180.1"))

    def test_non_finite_input_is_rejected(self) -> None:
        self.assertIsNone(parse_location("nan", "77.2"))
        self.assertIsNone(parse_location("inf", "0"))


class RenderCitationTextTests(unittest.TestCase):
    def test_full_citation_renders_every_field(self) -> None:
        self.assertEqual(
            render_citation_text(make_citation()),
            "PM-KISAN Revised FAQ  pp. 4-6  *Ministry of Agriculture*  "
            "[https://example.gov.in/pm-kisan](https://example.gov.in/pm-kisan)",
        )

    def test_single_page_renders_one_page(self) -> None:
        citation = make_citation(page_number=4, page_end=None)
        self.assertEqual(
            render_citation_text(citation),
            "PM-KISAN Revised FAQ  p. 4  *Ministry of Agriculture*  "
            "[https://example.gov.in/pm-kisan](https://example.gov.in/pm-kisan)",
        )

    def test_equal_page_bounds_render_one_page(self) -> None:
        citation = make_citation(page_number=4, page_end=4)
        self.assertIn("  p. 4  ", render_citation_text(citation))

    def test_absent_optional_fields_are_omitted(self) -> None:
        citation = make_citation(
            page_number=None, page_end=None, issuing_authority=None
        )
        self.assertEqual(
            render_citation_text(citation),
            "PM-KISAN Revised FAQ  "
            "[https://example.gov.in/pm-kisan](https://example.gov.in/pm-kisan)",
        )


class StatusPresentationTests(unittest.TestCase):
    def test_labels_cover_every_status(self) -> None:
        self.assertEqual(status_label(ResponseStatus.ANSWERED), "Answered")
        self.assertEqual(
            status_label(ResponseStatus.NEEDS_CLARIFICATION), "Needs clarification"
        )
        self.assertEqual(status_label(ResponseStatus.ABSTAINED), "Abstained")

    def test_footers_exist_only_where_guidance_is_needed(self) -> None:
        self.assertIsNone(status_footer(ResponseStatus.ANSWERED))
        self.assertIsNotNone(status_footer(ResponseStatus.ABSTAINED))
        self.assertIsNotNone(status_footer(ResponseStatus.NEEDS_CLARIFICATION))


class FactSelectionTests(unittest.TestCase):
    def test_known_selections_map_to_tri_state_values(self) -> None:
        self.assertIs(fact_selection_to_bool("Yes"), True)
        self.assertIs(fact_selection_to_bool("No"), False)
        self.assertIsNone(fact_selection_to_bool("Not sure"))

    def test_unknown_or_non_string_selection_is_not_supplied(self) -> None:
        self.assertIsNone(fact_selection_to_bool("Maybe"))
        self.assertIsNone(fact_selection_to_bool(1))  # type: ignore[arg-type]

    def test_selection_vocabulary_is_stable(self) -> None:
        self.assertEqual(FACT_SELECTIONS, ("Not sure", "Yes", "No"))


class BuildEligibilityRequestTests(unittest.TestCase):
    def test_all_unknown_yields_no_request(self) -> None:
        selections = {fact: "Not sure" for fact in FACT_NAMES}
        self.assertIsNone(build_eligibility_request(selections))

    def test_empty_mapping_yields_no_request(self) -> None:
        self.assertIsNone(build_eligibility_request({}))

    def test_explicit_selections_become_facts(self) -> None:
        selections = {
            "landholding_in_own_name": "Yes",
            "land_is_cultivable": "No",
            "land_used_for_non_agricultural_purpose": "Not sure",
            "family_member_paid_income_tax_last_assessment_year": "Not sure",
        }
        request = build_eligibility_request(selections)
        self.assertIsInstance(request, EligibilityRequest)
        assert request is not None
        self.assertIs(request.landholding_in_own_name, True)
        self.assertIs(request.land_is_cultivable, False)
        self.assertIsNone(request.land_used_for_non_agricultural_purpose)
        self.assertIsNone(request.family_member_paid_income_tax_last_assessment_year)
        self.assertEqual(request.scheme, "pm-kisan")

    def test_unknown_keys_are_ignored(self) -> None:
        request = build_eligibility_request({"unrelated": "Yes"})
        self.assertIsNone(request)


class LabelCoverageTests(unittest.TestCase):
    def test_fact_labels_match_the_eligibility_model(self) -> None:
        self.assertEqual(set(FACT_LABELS), set(FACT_NAMES))

    def test_language_labels_cover_every_supported_language(self) -> None:
        self.assertEqual(set(LANGUAGE_LABELS), set(Language))


class ImageContentTypeTests(unittest.TestCase):
    def test_supported_types_pass_through(self) -> None:
        self.assertEqual(image_content_type_for("image/png"), "image/png")
        self.assertEqual(image_content_type_for("image/jpeg"), "image/jpeg")

    def test_jpg_alias_is_normalized(self) -> None:
        self.assertEqual(image_content_type_for("image/jpg"), "image/jpeg")

    def test_parameters_are_stripped(self) -> None:
        self.assertEqual(image_content_type_for("image/png; charset=binary"), "image/png")

    def test_unusable_values_fall_back_to_jpeg(self) -> None:
        self.assertEqual(image_content_type_for(None), "image/jpeg")
        self.assertEqual(image_content_type_for(""), "image/jpeg")
        self.assertEqual(image_content_type_for("text/html"), "image/jpeg")


class BuildRequestStateTests(unittest.TestCase):
    def test_required_state_keys_are_present(self) -> None:
        state = build_request_state(QUESTION, Language.ENGLISH)
        for key in ("message", "route", "retrieved_chunks", "response"):
            self.assertIn(key, state)
        self.assertEqual(state["route"], "")
        self.assertIsNone(state["response"])
        self.assertEqual(state["retrieved_chunks"], ())

    def test_message_carries_text_language_and_location(self) -> None:
        state = build_request_state(
            QUESTION, Language.HINDI, location=(28.6, 77.2)
        )
        message = state["message"]
        self.assertEqual(message.text, QUESTION)
        self.assertIs(message.language, Language.HINDI)
        self.assertEqual(message.latitude, 28.6)
        self.assertEqual(message.longitude, 77.2)

    def test_absent_location_stays_absent(self) -> None:
        state = build_request_state(QUESTION, Language.ENGLISH)
        self.assertIsNone(state["message"].latitude)
        self.assertIsNone(state["message"].longitude)

    def test_modality_content_types_are_dropped_without_bytes(self) -> None:
        state = build_request_state(
            QUESTION,
            Language.ENGLISH,
            image_content_type="image/png",
            image_filename="leaf.png",
            audio_content_type="audio/wav",
        )
        self.assertIsNone(state["image_data"])
        self.assertIsNone(state["image_content_type"])
        self.assertIsNone(state["image_filename"])
        self.assertIsNone(state["audio_data"])
        self.assertIsNone(state["audio_content_type"])

    def test_modality_payload_is_passed_through_with_bytes(self) -> None:
        state = build_request_state(
            QUESTION,
            Language.ENGLISH,
            image_bytes=b"png",
            image_content_type="image/png",
            image_filename="leaf.png",
            audio_bytes=b"wav",
            audio_content_type="audio/wav",
        )
        self.assertEqual(state["image_data"], b"png")
        self.assertEqual(state["image_content_type"], "image/png")
        self.assertEqual(state["image_filename"], "leaf.png")
        self.assertEqual(state["audio_data"], b"wav")
        self.assertEqual(state["audio_content_type"], "audio/wav")

    def test_eligibility_request_is_passed_through(self) -> None:
        request = build_eligibility_request({"landholding_in_own_name": "Yes"})
        state = build_request_state(
            QUESTION, Language.ENGLISH, eligibility=request
        )
        self.assertIs(state["eligibility_request"], request)

    def test_blank_question_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_request_state("   ", Language.ENGLISH)


class InvokeGraphSafelyTests(unittest.TestCase):
    def test_successful_invocation_returns_the_response(self) -> None:
        response = object()
        graph = StubGraph(result={"response": response})
        result, error = invoke_graph_safely(graph, {"route": ""})
        self.assertIs(result, response)
        self.assertIsNone(error)
        self.assertEqual(graph.states, [{"route": ""}])

    def test_raising_graph_returns_the_safe_message(self) -> None:
        graph = StubGraph(error=RuntimeError("boom"))
        result, error = invoke_graph_safely(graph, {})
        self.assertIsNone(result)
        self.assertEqual(error, SAFE_ERROR_MESSAGE)

    def test_non_mapping_result_returns_the_safe_message(self) -> None:
        graph = StubGraph(result="unexpected")
        result, error = invoke_graph_safely(graph, {})
        self.assertIsNone(result)
        self.assertEqual(error, SAFE_ERROR_MESSAGE)

    def test_missing_response_returns_the_safe_message(self) -> None:
        graph = StubGraph(result={"route": "finalize"})
        result, error = invoke_graph_safely(graph, {})
        self.assertIsNone(result)
        self.assertEqual(error, SAFE_ERROR_MESSAGE)


if __name__ == "__main__":
    unittest.main()
