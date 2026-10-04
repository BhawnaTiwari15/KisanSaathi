"""Deterministic heading-aware chunking with paragraph-based fallback."""

import hashlib
from statistics import median

from kisansathi.ingestion.models import ExtractedPage, TextBlock, TextChunk


DEFAULT_MAX_CHARS = 1200
HEADING_MAX_CHARS = 180


def _is_heading(block: TextBlock, typical_font_size: float) -> bool:
    if len(block.text) > HEADING_MAX_CHARS:
        return False
    if block.is_bold:
        return True
    return typical_font_size > 0 and block.font_size >= typical_font_size * 1.2


def _split_oversized(text: str, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for word in paragraph.split():
            if len(word) > max_chars:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(word[offset : offset + max_chars] for offset in range(0, len(word), max_chars))
                continue
            candidate = f"{current} {word}" if current else word
            if len(candidate) > max_chars:
                pieces.append(current)
                current = word
            else:
                current = candidate
    if current:
        pieces.append(current)
    return pieces


def _chunk_id(
    source_id: str,
    sha256: str,
    ordinal: int,
    heading: str | None,
    text: str,
    page_start: int,
    page_end: int,
) -> str:
    identity = "\0".join(
        (source_id, sha256, str(ordinal), heading or "", text, str(page_start), str(page_end))
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"{source_id}:{digest}"


def chunk_pages(
    pages: tuple[ExtractedPage, ...], max_chars: int = DEFAULT_MAX_CHARS
) -> tuple[TextChunk, ...]:
    """Group blocks by likely headings, falling back to bounded paragraph chunks."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not pages:
        return ()
    source_id = pages[0].source_id
    sha256 = pages[0].sha256
    if any(page.source_id != source_id or page.sha256 != sha256 for page in pages):
        raise ValueError("All pages in a chunking operation must belong to one document version")

    all_blocks = [block for page in pages for block in page.blocks]
    font_sizes = [block.font_size for block in all_blocks if block.font_size > 0]
    typical_font_size = median(font_sizes) if font_sizes else 0.0

    chunks: list[TextChunk] = []
    ordinal = 0
    active_heading: str | None = None
    active_parts: list[str] = []
    active_pages: list[int] = []

    def flush() -> None:
        nonlocal ordinal, active_parts, active_pages
        if not active_parts:
            return
        text = "\n\n".join(active_parts)
        page_start = min(active_pages)
        page_end = max(active_pages)
        chunks.append(
            TextChunk(
                chunk_id=_chunk_id(
                    source_id, sha256, ordinal, active_heading, text, page_start, page_end
                ),
                source_id=source_id,
                sha256=sha256,
                ordinal=ordinal,
                heading=active_heading,
                text=text,
                page_start=page_start,
                page_end=page_end,
            )
        )
        ordinal += 1
        active_parts = []
        active_pages = []

    for page in pages:
        for block in page.blocks:
            if _is_heading(block, typical_font_size):
                flush()
                active_heading = block.text.strip()
                continue
            for piece in _split_oversized(block.text, max_chars):
                current_length = len("\n\n".join(active_parts))
                if active_parts and current_length + 2 + len(piece) > max_chars:
                    flush()
                active_parts.append(piece)
                active_pages.append(page.page_number)
    flush()
    return tuple(chunks)