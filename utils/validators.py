"""
FalconAI-Kids input validation utilities.

This module validates and normalizes:
- User names
- Birth dates and calculated ages
- Reference photos
- Image formats, headers, dimensions and file sizes
- Optional interests
- Complete pipeline input payloads

The implementation intentionally uses Python standard-library
functionality only. No GPU, ML or external image-processing
dependency is required.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import INPUT_VALIDATION
from utils.exceptions import (
    AgeOutOfRangeError,
    InvalidBirthdayError,
    InvalidNameError,
    InvalidPhotoError,
    ValidationError,
)


# ============================================================================
# Constants
# ============================================================================

NAME_REGEX = re.compile(
    r"^[^\W\d_](?:[^\W\d_]|[\s\-'\u2019\u0060])*$",
    re.UNICODE,
)

DATE_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}$"
)

IMAGE_SIGNATURES = {
    ".jpg": b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
    ".png": b"\x89PNG\r\n\x1a\n",
}


# ============================================================================
# Configuration helpers
# ============================================================================

def _get_setting(
    key: str,
    default: Any = None,
) -> Any:
    """Safely read a validation setting."""
    return INPUT_VALIDATION.get(key, default)


def _get_allowed_image_formats() -> set[str]:
    """Return normalized allowed image extensions."""
    formats = _get_setting(
        "allowed_image_formats",
        [".jpg", ".jpeg", ".png", ".webp"],
    )

    normalized: set[str] = set()

    for value in formats:
        extension = str(value).strip().lower()

        if not extension:
            continue

        if not extension.startswith("."):
            extension = f".{extension}"

        normalized.add(extension)

    return normalized


def _get_max_image_size_bytes() -> int:
    """Convert configured image size from MB to bytes."""
    max_mb = float(
        _get_setting(
            "max_image_size_mb",
            10,
        )
    )

    if max_mb <= 0:
        raise ValueError(
            "INPUT_VALIDATION['max_image_size_mb'] must be greater than zero."
        )

    return int(max_mb * 1024 * 1024)


def _get_min_image_resolution() -> Tuple[int, int]:
    """Return configured minimum image dimensions."""
    resolution = _get_setting(
        "min_image_resolution",
        (256, 256),
    )

    if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
        raise ValueError(
            "INPUT_VALIDATION['min_image_resolution'] must contain "
            "exactly two values."
        )

    width = int(resolution[0])
    height = int(resolution[1])

    if width <= 0 or height <= 0:
        raise ValueError(
            "Minimum image resolution values must be greater than zero."
        )

    return width, height


# ============================================================================
# Photo validation
# ============================================================================

def validate_photo(
    photo_path: Union[str, Path],
) -> Path:
    """
    Validate a reference image and return its resolved Path.

    Validation includes:
    - existence
    - regular-file check
    - extension
    - file size
    - binary signature
    - image dimensions
    """
    if photo_path is None:
        raise InvalidPhotoError(
            "A photo path is required.",
            photo_path=None,
        )

    try:
        path = Path(photo_path).expanduser()
    except (TypeError, ValueError) as exc:
        raise InvalidPhotoError(
            f"Invalid photo path: {exc}",
            photo_path=str(photo_path),
        ) from exc

    if not path.exists():
        raise InvalidPhotoError(
            f"Image file does not exist: '{path}'.",
            photo_path=str(path),
        )

    if not path.is_file():
        raise InvalidPhotoError(
            f"Path does not point to a regular file: '{path}'.",
            photo_path=str(path),
        )

    suffix = path.suffix.lower()
    allowed_formats = _get_allowed_image_formats()

    if suffix not in allowed_formats:
        allowed = ", ".join(sorted(allowed_formats))

        raise InvalidPhotoError(
            (
                f"Unsupported image format '{suffix or '<none>'}'. "
                f"Allowed formats: {allowed}."
            ),
            photo_path=str(path),
        )

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise InvalidPhotoError(
            f"Unable to inspect image file: {exc}",
            photo_path=str(path),
        ) from exc

    if file_size <= 0:
        raise InvalidPhotoError(
            "Image file is empty.",
            photo_path=str(path),
        )

    max_size_bytes = _get_max_image_size_bytes()

    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)

        raise InvalidPhotoError(
            (
                f"Image size is {size_mb:.2f} MB, "
                f"but the maximum allowed size is {max_mb:.2f} MB."
            ),
            photo_path=str(path),
        )

    _validate_image_header(path)
    _validate_image_resolution(path)

    return path.resolve()


def _validate_image_header(path: Path) -> None:
    """
    Validate the image's binary signature.

    Supported:
    - JPEG
    - PNG
    - WEBP
    """
    try:
        with path.open("rb") as file:
            header = file.read(32)

    except (OSError, IOError) as exc:
        raise InvalidPhotoError(
            f"Unable to read image header: {exc}",
            photo_path=str(path),
        ) from exc

    if not header:
        raise InvalidPhotoError(
            "Image file contains no readable data.",
            photo_path=str(path),
        )

    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        if not header.startswith(b"\xff\xd8\xff"):
            raise InvalidPhotoError(
                "File extension indicates JPEG, but the binary signature "
                "does not match JPEG format.",
                photo_path=str(path),
            )

        return

    if suffix == ".png":
        if not header.startswith(b"\x89PNG\r\n\x1a\n"):
            raise InvalidPhotoError(
                "File extension indicates PNG, but the binary signature "
                "does not match PNG format.",
                photo_path=str(path),
            )

        return

    if suffix == ".webp":
        if (
            len(header) < 12
            or header[:4] != b"RIFF"
            or header[8:12] != b"WEBP"
        ):
            raise InvalidPhotoError(
                "File extension indicates WEBP, but the binary signature "
                "does not match WEBP format.",
                photo_path=str(path),
            )

        return

    raise InvalidPhotoError(
        f"Unsupported image format '{suffix}'.",
        photo_path=str(path),
    )


# ============================================================================
# Image dimensions
# ============================================================================

def _validate_image_resolution(path: Path) -> None:
    """Validate image width and height without loading the entire image."""
    min_width, min_height = _get_min_image_resolution()

    try:
        width, height = _read_image_dimensions(path)

    except InvalidPhotoError:
        raise

    except (OSError, IOError) as exc:
        raise InvalidPhotoError(
            f"Unable to read image dimensions: {exc}",
            photo_path=str(path),
        ) from exc

    except Exception as exc:
        raise InvalidPhotoError(
            f"Unable to determine image dimensions: {exc}",
            photo_path=str(path),
        ) from exc

    if width is None or height is None:
        raise InvalidPhotoError(
            (
                "Unable to determine image dimensions. "
                "The image may be malformed or use an unsupported encoding."
            ),
            photo_path=str(path),
        )

    if width <= 0 or height <= 0:
        raise InvalidPhotoError(
            f"Invalid image dimensions: {width}x{height}.",
            photo_path=str(path),
        )

    if width < min_width or height < min_height:
        raise InvalidPhotoError(
            (
                f"Image resolution {width}x{height}px is below the "
                f"minimum required resolution of "
                f"{min_width}x{min_height}px."
            ),
            photo_path=str(path),
        )


def _read_image_dimensions(
    path: Path,
) -> Tuple[Optional[int], Optional[int]]:
    """Read dimensions from supported image headers."""
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return _read_jpeg_dimensions(path)

    if suffix == ".png":
        return _read_png_dimensions(path)

    if suffix == ".webp":
        return _read_webp_dimensions(path)

    return None, None


def _read_png_dimensions(
    path: Path,
) -> Tuple[Optional[int], Optional[int]]:
    """Read PNG dimensions from the IHDR chunk."""
    with path.open("rb") as file:
        signature = file.read(8)

        if signature != b"\x89PNG\r\n\x1a\n":
            raise InvalidPhotoError(
                "Invalid PNG signature.",
                photo_path=str(path),
            )

        ihdr_length_bytes = file.read(4)

        if len(ihdr_length_bytes) != 4:
            raise InvalidPhotoError(
                "PNG file is truncated before the IHDR chunk.",
                photo_path=str(path),
            )

        ihdr_length = int.from_bytes(
            ihdr_length_bytes,
            "big",
        )

        if ihdr_length < 8:
            raise InvalidPhotoError(
                "PNG IHDR chunk is malformed.",
                photo_path=str(path),
            )

        chunk_type = file.read(4)

        if chunk_type != b"IHDR":
            raise InvalidPhotoError(
                "PNG file does not contain a valid IHDR chunk.",
                photo_path=str(path),
            )

        dimensions = file.read(8)

        if len(dimensions) != 8:
            raise InvalidPhotoError(
                "PNG file is truncated inside the IHDR chunk.",
                photo_path=str(path),
            )

        width = int.from_bytes(
            dimensions[:4],
            "big",
        )

        height = int.from_bytes(
            dimensions[4:8],
            "big",
        )

        return width, height


def _read_jpeg_dimensions(
    path: Path,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Read JPEG dimensions by walking JPEG markers.

    Supports common SOF markers rather than relying on a single
    hard-coded JPEG encoding.
    """
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }

    with path.open("rb") as file:
        if file.read(2) != b"\xff\xd8":
            raise InvalidPhotoError(
                "Invalid JPEG start-of-image marker.",
                photo_path=str(path),
            )

        while True:
            byte = file.read(1)

            if not byte:
                break

            if byte != b"\xff":
                continue

            # JPEG markers can contain repeated FF bytes.
            while byte == b"\xff":
                byte = file.read(1)

            if not byte:
                break

            marker = byte[0]

            # Standalone markers.
            if marker in {
                0xD8,
                0xD9,
                0x01,
            } or 0xD0 <= marker <= 0xD7:
                continue

            length_bytes = file.read(2)

            if len(length_bytes) != 2:
                break

            segment_length = int.from_bytes(
                length_bytes,
                "big",
            )

            if segment_length < 2:
                raise InvalidPhotoError(
                    "JPEG segment contains an invalid length.",
                    photo_path=str(path),
                )

            if marker in sof_markers:
                precision = file.read(1)

                if not precision:
                    break

                dimensions = file.read(4)

                if len(dimensions) != 4:
                    break

                height = int.from_bytes(
                    dimensions[:2],
                    "big",
                )

                width = int.from_bytes(
                    dimensions[2:4],
                    "big",
                )

                return width, height

            file.seek(
                segment_length - 2,
                1,
            )

    return None, None


def _read_webp_dimensions(
    path: Path,
) -> Tuple[Optional[int], Optional[int]]:
    """Read dimensions from common WEBP VP8, VP8L and VP8X headers."""
    with path.open("rb") as file:
        header = file.read(32)

    if len(header) < 16:
        raise InvalidPhotoError(
            "WEBP file is too small to contain a valid header.",
            photo_path=str(path),
        )

    if header[:4] != b"RIFF" or header[8:12] != b"WEBP":
        raise InvalidPhotoError(
            "Invalid WEBP container header.",
            photo_path=str(path),
        )

    chunk_type = header[12:16]

    # ------------------------------------------------------------------
    # Lossy WEBP: VP8
    # ------------------------------------------------------------------
    if chunk_type == b"VP8 ":
        if len(header) < 30:
            raise InvalidPhotoError(
                "WEBP VP8 header is truncated.",
                photo_path=str(path),
            )

        # Search for the VP8 frame start code.
        with path.open("rb") as file:
            data = file.read(64)

        frame_start = data.find(
            b"\x9d\x01\x2a",
            16,
        )

        if frame_start == -1 or len(data) < frame_start + 7:
            return None, None

        width = int.from_bytes(
            data[frame_start + 3:frame_start + 5],
            "little",
        ) & 0x3FFF

        height = int.from_bytes(
            data[frame_start + 5:frame_start + 7],
            "little",
        ) & 0x3FFF

        return width, height

    # ------------------------------------------------------------------
    # Lossless WEBP: VP8L
    # ------------------------------------------------------------------
    if chunk_type == b"VP8L":
        if len(header) < 25:
            raise InvalidPhotoError(
                "WEBP VP8L header is truncated.",
                photo_path=str(path),
            )

        if header[20] != 0x2F:
            raise InvalidPhotoError(
                "Invalid WEBP VP8L signature.",
                photo_path=str(path),
            )

        b1 = header[21]
        b2 = header[22]
        b3 = header[23]
        b4 = header[24]

        width = 1 + (
            b1
            | ((b2 & 0x3F) << 8)
        )

        height = 1 + (
            ((b2 >> 6) & 0x03)
            | (b3 << 2)
            | ((b4 & 0x0F) << 10)
        )

        return width, height

    # ------------------------------------------------------------------
    # Extended WEBP: VP8X
    # ------------------------------------------------------------------
    if chunk_type == b"VP8X":
        if len(header) < 30:
            raise InvalidPhotoError(
                "WEBP VP8X header is truncated.",
                photo_path=str(path),
            )

        width = 1 + int.from_bytes(
            header[24:27],
            "little",
        )

        height = 1 + int.from_bytes(
            header[27:30],
            "little",
        )

        return width, height

    return None, None


# ============================================================================
# Name validation
# ============================================================================

def validate_name(name: str) -> str:
    """
    Validate and normalize a character/person name.

    Allowed:
    - Unicode letters
    - spaces
    - hyphens
    - apostrophes
    - backticks
    """
    if name is None:
        raise InvalidNameError(
            "Name cannot be null.",
            name=None,
        )

    if not isinstance(name, str):
        raise InvalidNameError(
            "Name must be a string.",
            name=str(name),
        )

    name = name.strip()

    if not name:
        raise InvalidNameError(
            "Name cannot be empty.",
            name=name,
        )

    max_length = int(
        _get_setting(
            "max_name_length",
            100,
        )
    )

    if len(name) > max_length:
        raise InvalidNameError(
            (
                f"Name is too long ({len(name)} characters). "
                f"Maximum length is {max_length} characters."
            ),
            name=name,
        )

    if len(name) < 2:
        raise InvalidNameError(
            "Name must contain at least 2 characters.",
            name=name,
        )

    if any(char.isdigit() for char in name):
        raise InvalidNameError(
            "Name cannot contain numeric digits.",
            name=name,
        )

    if not NAME_REGEX.fullmatch(name):
        raise InvalidNameError(
            (
                "Name contains unsupported characters. "
                "Only letters, spaces, hyphens and apostrophes are allowed."
            ),
            name=name,
        )

    # Collapse repeated whitespace without destroying hyphenated names.
    normalized = " ".join(name.split())

    # Preserve the original capitalization where possible.
    normalized = " ".join(
        part[:1].upper() + part[1:]
        for part in normalized.split(" ")
    )

    return normalized


# ============================================================================
# Birthday validation
# ============================================================================

def validate_birthday(
    birthday: str,
) -> Tuple[datetime, int]:
    """
    Validate YYYY-MM-DD birthday and calculate current age.
    """
    if birthday is None:
        raise InvalidBirthdayError(
            "Birthday cannot be null.",
            birthday=None,
        )

    if not isinstance(birthday, str):
        raise InvalidBirthdayError(
            "Birthday must be a string in YYYY-MM-DD format.",
            birthday=str(birthday),
        )

    birthday = birthday.strip()

    if not birthday:
        raise InvalidBirthdayError(
            "Birthday cannot be empty.",
            birthday=birthday,
        )

    if not DATE_REGEX.fullmatch(birthday):
        raise InvalidBirthdayError(
            (
                f"Invalid birthday format: '{birthday}'. "
                "Expected YYYY-MM-DD."
            ),
            birthday=birthday,
        )

    try:
        birthday_date = datetime.strptime(
            birthday,
            "%Y-%m-%d",
        )

    except ValueError as exc:
        raise InvalidBirthdayError(
            f"Invalid calendar date: {exc}",
            birthday=birthday,
        ) from exc

    today = date.today()

    if birthday_date.date() > today:
        raise InvalidBirthdayError(
            "Birthday cannot be in the future.",
            birthday=birthday,
        )

    age = _calculate_age(
        birthday_date.date(),
        today,
    )

    min_age = int(
        _get_setting(
            "min_age_years",
            0,
        )
    )

    max_age = int(
        _get_setting(
            "max_age_years",
            120,
        )
    )

    if age < min_age or age > max_age:
        raise AgeOutOfRangeError(
            age,
            min_age,
            max_age,
        )

    return birthday_date, age


def _calculate_age(
    birthday: date,
    today: date,
) -> int:
    """Calculate age accurately, including leap-year birthdays."""
    age = today.year - birthday.year

    if (today.month, today.day) < (
        birthday.month,
        birthday.day,
    ):
        age -= 1

    return age


# ============================================================================
# Interests
# ============================================================================

def validate_interests(
    interests: str,
) -> str:
    """
    Validate optional free-text interests.

    Example:
        "space, dinosaurs, robots"
    """
    if interests is None:
        raise ValidationError(
            "Interests cannot be null.",
            field="interests",
            value=None,
        )

    if not isinstance(interests, str):
        raise ValidationError(
            "Interests must be a string.",
            field="interests",
            value=interests,
        )

    interests = interests.strip()

    if not interests:
        raise ValidationError(
            "Interests cannot be empty.",
            field="interests",
            value=interests,
        )

    max_length = int(
        _get_setting(
            "max_interests_length",
            500,
        )
    )

    if len(interests) > max_length:
        raise ValidationError(
            (
                f"Interests are too long ({len(interests)} characters). "
                f"Maximum length is {max_length} characters."
            ),
            field="interests",
            value=interests,
        )

    return interests


# ============================================================================
# Complete input validation
# ============================================================================

def validate_inputs(
    name: str,
    birthday: str,
    photo_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Validate the core FalconAI-Kids pipeline inputs.

    Photo is intentionally optional because the pipeline can generate
    generic characters without a reference photograph.

    Returns:
        {
            "name": str,
            "birthday": datetime,
            "age": int,
            "photo_path": Optional[Path]
        }

    Raises:
        ValidationError:
            If one or more inputs are invalid.
    """
    errors: List[str] = []
    result: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Name
    # ------------------------------------------------------------------
    try:
        result["name"] = validate_name(name)

    except InvalidNameError as exc:
        errors.append(str(exc))

    # ------------------------------------------------------------------
    # Birthday
    # ------------------------------------------------------------------
    try:
        birthday_value, age = validate_birthday(
            birthday
        )

        result["birthday"] = birthday_value
        result["age"] = age

    except (
        InvalidBirthdayError,
        AgeOutOfRangeError,
    ) as exc:
        errors.append(str(exc))

    # ------------------------------------------------------------------
    # Photo
    # ------------------------------------------------------------------
    if photo_path is not None:
        try:
            result["photo_path"] = validate_photo(
                photo_path
            )

        except InvalidPhotoError as exc:
            errors.append(str(exc))

    else:
        result["photo_path"] = None

    # ------------------------------------------------------------------
    # Combined result
    # ------------------------------------------------------------------
    if errors:
        if len(errors) == 1:
            raise ValidationError(
                errors[0]
            )

        combined = (
            "Multiple input validation errors:\n"
            + "\n".join(
                f"  {index}. {error}"
                for index, error in enumerate(
                    errors,
                    start=1,
                )
            )
        )

        raise ValidationError(
            combined
        )

    return result


# ============================================================================
# Convenience helpers
# ============================================================================

def is_valid_name(name: str) -> bool:
    """Return True if the name passes validation."""
    try:
        validate_name(name)
        return True
    except InvalidNameError:
        return False


def is_valid_birthday(birthday: str) -> bool:
    """Return True if the birthday passes validation."""
    try:
        validate_birthday(birthday)
        return True
    except (
        InvalidBirthdayError,
        AgeOutOfRangeError,
    ):
        return False


def is_valid_photo(
    photo_path: Union[str, Path],
) -> bool:
    """Return True if the photo passes validation."""
    try:
        validate_photo(photo_path)
        return True
    except InvalidPhotoError:
        return False
