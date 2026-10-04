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