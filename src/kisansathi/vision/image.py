"""Image input validation and normalization.

Validates MIME type, file signature, size limits, and basic integrity.
"""

from dataclasses import dataclass

from kisansathi.vision.models import (
    ImageValidationError,
    UnsupportedFormatError,
    ImageTooLargeError,
)


SUPPORTED_IMAGE_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
})

# Maximum image size (10 MB)
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_SIZE_BYTES = 100

# Maximum dimensions to prevent decompression bombs
MAX_DIMENSION = 8192
MAX_PIXELS = 50_000_000  # 50 megapixels

# File signatures (magic bytes) for supported formats
FILE_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "image/jpeg": (b"\xFF\xD8\xFF",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/webp": (b"RIFF", b"WEBP"),  # RIFF container with WEBP
    "image/heic": (b"ftypheic", b"ftypmif1"),  # HEIC/HEIF in MP4 container
    "image/heif": (b"ftypheic", b"ftypmif1"),
}


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    """Validated image ready for analysis."""

    data: bytes
    content_type: str
    size_bytes: int
    detected_format: str
    filename: str | None
    width: int | None = None
    height: int | None = None


def _detect_format_from_signature(data: bytes) -> str | None:
    """Detect image format from file signature (magic bytes)."""
    if len(data) < 12:
        return None

    # JPEG: FF D8 FF
    if data[:3] == b"\xFF\xD8\xFF":
        return "jpeg"

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"

    # WebP: RIFF....WEBP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp"

    # HEIC/HEIF: ftypheic or ftypmif1 at offset 4 (MP4 container)
    if len(data) >= 12:
        if data[4:12] in (b"ftypheic", b"ftypmif1"):
            return "heic"

    return None


def _validate_dimensions(data: bytes, detected_format: str) -> tuple[int | None, int | None]:
    """Attempt to extract and validate image dimensions.
    
    Returns (width, height) if successfully parsed, (None, None) otherwise.
    Does not fully decode the image - only reads headers.
    """
    try:
        if detected_format == "jpeg":
            # JPEG: scan for SOF marker (0xFF 0xC0-0xCF except 0xC4, 0xC8, 0xCC)
            i = 2  # Skip SOI
            while i < len(data) - 8:
                if data[i] == 0xFF and 0xC0 <= data[i + 1] <= 0xCF and data[i + 1] not in (0xC4, 0xC8, 0xCC):
                    height = int.from_bytes(data[i + 5:i + 7], "big")
                    width = int.from_bytes(data[i + 7:i + 9], "big")
                    return width, height
                i += 1
        elif detected_format == "png":
            # PNG: IHDR chunk at offset 8, width at 16-19, height at 20-23
            if len(data) >= 24 and data[12:16] == b"IHDR":
                width = int.from_bytes(data[16:20], "big")
                height = int.from_bytes(data[20:24], "big")
                return width, height
        elif detected_format == "webp":
            # WebP: VP8/VP8L chunk after RIFF header
            if len(data) >= 30 and data[12:16] == b"VP8 ":
                # VP8: width at 26-27, height at 28-29 (little endian, 14 bits each)
                w = int.from_bytes(data[26:28], "little") & 0x3FFF
                h = int.from_bytes(data[28:30], "little") & 0x3FFF
                return w, h
            elif len(data) >= 30 and data[12:16] == b"VP8L":
                # VP8L: width-1 at 21-24, height-1 at 24-27 (little endian, 14 bits each)
                w = (int.from_bytes(data[21:25], "little") & 0x3FFF) + 1
                h = (int.from_bytes(data[25:29], "little") & 0x3FFF) + 1
                return w, h
        elif detected_format in ("heic", "heif"):
            # HEIC/HEIF: dimensions in 'pasp' or 'ihdr' box - skip for now
            pass
    except Exception:
        pass
    return None, None


def validate_image_input(
    image_data: bytes,
    content_type: str,
    *,
    max_size: int = MAX_IMAGE_SIZE_BYTES,
    min_size: int = MIN_IMAGE_SIZE_BYTES,
    filename: str | None = None,
) -> ValidatedImage:
    """Validate image input and return normalized metadata.

    Args:
        image_data: Raw image bytes.
        content_type: MIME type from client (may be spoofed).
        max_size: Maximum allowed size in bytes.
        min_size: Minimum allowed size in bytes.
        filename: Original filename if available.

    Returns:
        ValidatedImage with normalized metadata.

    Raises:
        ImageValidationError: Any validation failure.
        ImageTooLargeError: Image exceeds size limit.
        UnsupportedFormatError: Unsupported format.
    """
    if not image_data:
        raise ImageValidationError("Image data is empty")

    size = len(image_data)
    if size < min_size:
        raise ImageValidationError(
            f"Image data too small: {size} bytes (minimum {min_size})"
        )
    if size > max_size:
        raise ImageTooLargeError(
            f"Image too large: {size} bytes (maximum {max_size})"
        )

    if not content_type:
        raise ImageValidationError("Content type is required")

    normalized_content_type = content_type.lower().split(";")[0].strip()
    if normalized_content_type not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedFormatError(
            f"Unsupported content type: {normalized_content_type}. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}"
        )

    # Validate file signature matches declared content type
    detected_format = _detect_format_from_signature(image_data)
    format_to_mime = {
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "heic": "image/heic",
        "heif": "image/heif",
    }
    detected_mime = format_to_mime.get(detected_format)

    if not detected_mime:
        raise ImageValidationError(
            "Unable to detect image format or unsupported format. "
            "File may be corrupted or not a valid image."
        )

    # Strict MIME match - reject if content-type doesn't match actual format
    if detected_mime != normalized_content_type:
        raise ImageValidationError(
            f"Content type mismatch: declared {normalized_content_type}, "
            f"detected {detected_mime}. File may be mislabeled or corrupted."
        )

    # Validate dimensions to prevent decompression bombs
    width, height = _validate_dimensions(image_data, detected_format)
    if width is not None and height is not None:
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            raise ImageValidationError(
                f"Image dimensions too large: {width}x{height} "
                f"(maximum {MAX_DIMENSION}x{MAX_DIMENSION})"
            )
        if width * height > MAX_PIXELS:
            raise ImageValidationError(
                f"Image has too many pixels: {width * height:,} "
                f"(maximum {MAX_PIXELS:,})"
            )

    return ValidatedImage(
        data=image_data,
        content_type=normalized_content_type,
        size_bytes=size,
        detected_format=detected_format,
        filename=filename,
        width=width,
        height=height,
    )