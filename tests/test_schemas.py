import unittest

from kisansathi.domain.schemas import (
    AssistantResponse,
    Citation,
    Language,
    ResponseStatus,
    UserMessage,
)


class DomainSchemaTests(unittest.TestCase):
    def test_user_message_accepts_hindi_text(self) -> None:
        message = UserMessage(text="नमस्ते", language=Language.HINDI)

        self.assertEqual(message.language, Language.HINDI)

    def test_user_message_rejects_blank_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "Message text must not be empty"):
            UserMessage(text="   ")

    def test_user_message_defaults_to_no_coordinates(self) -> None:
        message = UserMessage(text="What is the weather today?")

        self.assertIsNone(message.latitude)
        self.assertIsNone(message.longitude)

    def test_user_message_accepts_optional_coordinates(self) -> None:
        message = UserMessage(
            text="What is the weather today?",
            latitude=28.6139,
            longitude=77.2090,
        )

        self.assertEqual(message.latitude, 28.6139)
        self.assertEqual(message.longitude, 77.2090)

    def test_user_message_rejects_out_of_range_coordinates(self) -> None:
        with self.subTest(field="latitude", value=90.5):
            with self.assertRaisesRegex(ValueError, r"latitude must be within \[-90, 90\]"):
                UserMessage(text="What is the weather today?", latitude=90.5, longitude=77.2090)
        with self.subTest(field="longitude", value=-180.5):
            with self.assertRaisesRegex(ValueError, r"longitude must be within \[-180, 180\]"):
                UserMessage(text="What is the weather today?", latitude=28.6139, longitude=-180.5)

    def test_user_message_rejects_non_numeric_coordinates(self) -> None:
        for latitude, longitude in ((True, 77.2090), (28.6139, False), ("28.6", 77.2090)):
            with self.subTest(latitude=latitude, longitude=longitude):
                with self.assertRaises(TypeError):
                    UserMessage(
                        text="What is the weather today?",
                        latitude=latitude,
                        longitude=longitude,
                    )

    def test_citation_requires_a_positive_page_number(self) -> None:
        with self.assertRaisesRegex(ValueError, "page_number must be at least 1"):
            Citation(source_id="scheme-1", title="Scheme", url="https://example.gov.in", page_number=0)

    def test_response_defaults_to_answered_without_citations(self) -> None:
        response = AssistantResponse(text="Hello", language=Language.ENGLISH)

        self.assertEqual(response.status, ResponseStatus.ANSWERED)
        self.assertEqual(response.citations, ())

    def test_response_can_represent_abstention_with_citations(self) -> None:
        citation = Citation(
            source_id="scheme-1",
            title="Official scheme document",
            url="https://example.gov.in/scheme.pdf",
            page_number=2,
        )
        response = AssistantResponse(
            text="The available information is not sufficient.",
            language=Language.ENGLISH,
            status=ResponseStatus.ABSTAINED,
            citations=(citation,),
        )

        self.assertEqual(response.status, ResponseStatus.ABSTAINED)
        self.assertEqual(response.citations, (citation,))


if __name__ == "__main__":
    unittest.main()