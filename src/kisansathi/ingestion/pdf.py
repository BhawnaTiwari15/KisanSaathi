"""Page-aware extraction from local, text-based PDF files."""

from pathlib import Path

from kisansathi.ingestion.models import ExtractedPage, ExtractionResult, IngestionError, TextBlock


class PdfExtractionError(IngestionError):
    """Raised when a PDF cannot be parsed or contains no pages."""


def extract_pdf(path: str | Path, source_id: str, sha256: str) -> ExtractionResult:
    """Extract every PDF page and expose one-based page numbers."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PdfExtractionError(f"PDF input does not exist: {pdf_path}")

    try:
        import pymupdf
    except ImportError as error:
        raise PdfExtractionError(
            "PyMuPDF is required for PDF extraction; install the project runtime dependencies"
        ) from error

    try:
        document = pymupdf.open(pdf_path)
    except Exception as error:
        raise PdfExtractionError(f"Cannot open PDF {pdf_path}: {error}") from error

    try:
        if document.needs_pass:
            raise PdfExtractionError(f"Encrypted PDF requires a password: {pdf_path}")
        if len(document) == 0:
            raise PdfExtractionError(f"PDF contains no pages: {pdf_path}")

        pages: list[ExtractedPage] = []
        for page_index in range(len(document)):
            page_number = page_index + 1
            page = document.load_page(page_index)
            page_data = page.get_text("dict", sort=True)
            blocks: list[TextBlock] = []
            for block_number, raw_block in enumerate(page_data.get("blocks", ())):
                if raw_block.get("type") != 0:
                    continue
                line_texts: list[str] = []
                font_sizes: list[float] = []
                is_bold = False
                for line in raw_block.get("lines", ()):
                    spans = line.get("spans", ())
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if text:
                        line_texts.append(text)
                    for span in spans:
                        if span.get("text", "").strip():
                            font_sizes.append(float(span.get("size", 0.0)))
                        font_name = str(span.get("font", "")).lower()
                        is_bold = is_bold or "bold" in font_name or bool(span.get("flags", 0) & 16)
                text = "\n".join(line_texts).strip()
                if text:
                    blocks.append(
                        TextBlock(
                            text=text,
                            block_number=block_number,
                            font_size=max(font_sizes, default=0.0),
                            is_bold=is_bold,
                        )
                    )
            pages.append(
                ExtractedPage(
                    source_id=source_id,
                    sha256=sha256,
                    page_number=page_number,
                    text="\n\n".join(block.text for block in blocks),
                    blocks=tuple(blocks),
                )
            )
    except PdfExtractionError:
        raise
    except Exception as error:
        raise PdfExtractionError(f"Failed while extracting PDF {pdf_path}: {error}") from error
    finally:
        document.close()

    warnings: list[str] = []
    empty_pages = [page.page_number for page in pages if not page.text.strip()]
    if empty_pages:
        warnings.append("Pages with no extractable text: " + ", ".join(map(str, empty_pages)))
    if all(not page.text.strip() for page in pages):
        warnings.append("Document has no extractable text; it may be scanned or image-only")

    return ExtractionResult(
        pages=tuple(pages),
        parser_version=str(getattr(pymupdf, "VersionBind", "unknown")),
        warnings=tuple(warnings),
    )