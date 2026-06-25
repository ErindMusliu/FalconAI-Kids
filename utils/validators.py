import re
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Tuple, List, Any

from config.settings import INPUT_VALIDATION
from utils.exceptions import (
    InvalidPhotoError,
    InvalidNameError,
    InvalidBirthdayError,
    AgeOutOfRangeError,
    ValidationError,
)

NAME_REGEX = re.compile(r"^[^\W\d_]([^\W\d_]|\s|\-|\'|\`)*$", re.UNICODE)

def validate_photo(photo_path: Union[str, Path]) -> Path:
    path = Path(photo_path)

    if not path.exists():
        raise InvalidPhotoError(
            f"Target image resource file does not exist: '{path}'",
            photo_path=str(path)
        )

    if not path.is_file():
        raise InvalidPhotoError(
            f"Provided target path route does not point to a valid file: '{path}'",
            photo_path=str(path)
        )

    allowed_formats = INPUT_VALIDATION["allowed_image_formats"]
    suffix = path.suffix.lower()
    if suffix not in allowed_formats:
        raise InvalidPhotoError(
            f"Unsupported file format extension tracking token: '{suffix}'. "
            f"Allowed extensions: {', '.join(allowed_formats)}",
            photo_path=str(path)
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise InvalidPhotoError(
            "The submitted image file is corrupted or empty (0 bytes).",
            photo_path=str(path)
        )

    max_size_bytes = INPUT_VALIDATION["max_image_size_mb"] * 1024 * 1024
    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        raise InvalidPhotoError(
            f"The image asset scale footprint ({size_mb:.1f} MB) exceeds safety limits. "
            f"Maximum allowed file threshold size: {INPUT_VALIDATION['max_image_size_mb']} MB",
            photo_path=str(path)
        )

    _validate_image_header(path)
    _validate_image_resolution(path)

    return path.resolve()

def _validate_image_header(path: Path) -> None:
    magic_bytes = {
        b'\xff\xd8\xff': "JPEG",
        b'\x89PNG\r\n\x1a\n': "PNG",
        b'RIFF': "WEBP",
    }

    try:
        with open(path, 'rb') as f:
            header = f.read(12)

        valid = False
        for magic, fmt in magic_bytes.items():
            if header.startswith(magic):
                if magic == b'RIFF' and header[8:12] != b'WEBP':
                    continue
                valid = True
                break

        if not valid:
            raise InvalidPhotoError(
                "Image signature validation failed. File header contains mismatched binary identifiers. "
                "Verify the source file is an uncorrupted JPEG, PNG, or WEBP asset.",
                photo_path=str(path)
            )

    except (IOError, OSError) as e:
        raise InvalidPhotoError(
            f"Operating system blocked access to target file stream indicators: {e}",
            photo_path=str(path)
        )


def _validate_image_resolution(path: Path) -> None:
    try:
        suffix = path.suffix.lower()
        min_w, min_h = INPUT_VALIDATION["min_image_resolution"]
        width, height = None, None

        if suffix in ['.jpg', '.jpeg']:
            with open(path, 'rb') as f:
                f.read(2)
                while True:
                    marker_bytes = f.read(2)
                    if not marker_bytes or marker_bytes[0] != 0xFF:
                        break
                    marker = marker_bytes[1]
                    if marker in (0xC0, 0xC2):
                        f.read(3)
                        height = int.from_bytes(f.read(2), 'big')
                        width = int.from_bytes(f.read(2), 'big')
                        break
                    else:
                        block_len = int.from_bytes(f.read(2), 'big') - 2
                        f.seek(block_len, 1)

        elif suffix == '.png':
            with open(path, 'rb') as f:
                f.seek(16)
                width = int.from_bytes(f.read(4), 'big')
                height = int.from_bytes(f.read(4), 'big')

        elif suffix == '.webp':
            with open(path, 'rb') as f:
                f.seek(12)
                chunk_header = f.read(4)
                if chunk_header == b'VP8 ':
                    f.seek(26)
                    width = int.from_bytes(f.read(2), 'little') & 0x3FFF
                    height = int.from_bytes(f.read(2), 'little') & 0x3FFF
                elif chunk_header == b'VP8L':
                    f.seek(21)
                    b1, b2, b3, b4 = f.read(4)
                    width = 1 + (((b2 & 0x3F) << 8) | b1)
                    height = 1 + (((b4 & 0xF) << 10) | (b3 << 2) | ((b2 & 0xC0) >> 6))
                elif chunk_header == b'VP8X':
                    f.seek(24)
                    w_bytes = f.read(3)
                    h_bytes = f.read(3)
                    width = 1 + int.from_bytes(w_bytes, 'little')
                    height = 1 + int.from_bytes(h_bytes, 'little')

        if width and height:
            if width < min_w or height < min_h:
                raise InvalidPhotoError(
                    f"The dimensions of the submitted image file are too small ({width}x{height}px). "
                    f"Platform minimum bounds require at least: {min_w}x{min_h}px. "
                    "Please upload a higher-resolution portrait image file asset.",
                    photo_path=str(path)
                )

    except InvalidPhotoError:
        raise
    except Exception as parse_err:
        pass

def validate_name(name: str) -> str:
    if not name:
        raise InvalidNameError("The subject name parameter field cannot be left blank.", name=name)

    name = name.strip()
    if not name:
        raise InvalidNameError("The subject name parameter field cannot consist entirely of whitespace.", name=name)

    max_length = INPUT_VALIDATION["max_name_length"]
    if len(name) > max_length:
        raise InvalidNameError(
            f"The provided name exceeds structural limits ({len(name)} characters). "
            f"Maximum allowable constraint length limit: {max_length} characters.",
            name=name
        )

    if len(name) < 2:
        raise InvalidNameError(
            f"The provided name is too short ({len(name)} characters). Names must contain at least 2 characters.",
            name=name
        )

    if any(char.isdigit() for char in name):
        raise InvalidNameError("Naming validation error; input values must not contain numeric digits.", name=name)

    if not NAME_REGEX.match(name):
        raise InvalidNameError(
            "The target string input block contains illegal character payloads. "
            "Only alphabetic letters, spaces, hyphens, and apostrophes are allowed.",
            name=name
        )

    return " ".join(word.capitalize() for word in name.split())

def validate_birthday(birthday: str) -> Tuple[datetime, int]:
    if not birthday:
        raise InvalidBirthdayError("Chronological parameters input missing; date token field is blank.", birthday=birthday)

    birthday = birthday.strip()
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', birthday):
        raise InvalidBirthdayError(
            f"Invalid target formatting identifier captured: '{birthday}'. "
            "Please use the required ISO Standard format sequence structure: YYYY-MM-DD (e.g., 2018-05-10).",
            birthday=birthday
        )

    try:
        bday = datetime.strptime(birthday, "%Y-%m-%d")
    except ValueError as parse_err:
        raise InvalidBirthdayError(
            f"The entry target calendar values do not correspond to an actual calendar timeline: {parse_err}",
            birthday=birthday
        )

    today = datetime.today()
    if bday.date() > today.date():
        raise InvalidBirthdayError(
            f"Chronological registration failure; date parameter value context ('{birthday}') lies in the future. "
            f"Current platform reference timestamp: {today.strftime('%Y-%m-%d')}.",
            birthday=birthday
        )

    if (today - bday).days > 100 * 365:
        raise InvalidBirthdayError(
            f"The provided chronological timestamp ('{birthday}') exceeds maximum historical thresholds.",
            birthday=birthday
        )

    age = today.year - bday.year - ((today.month, today.day) < (bday.month, bday.day))

    min_age = INPUT_VALIDATION["min_age_years"]
    max_age = INPUT_VALIDATION["max_age_years"]

    if age < min_age or age > max_age:
        raise AgeOutOfRangeError(age, min_age, max_age)

    return bday, age

def validate_inputs(photo_path: Union[str, Path], name: str, birthday: str) -> Dict[str, Any]:
    errors: List[str] = []
    result: Dict[str, Any] = {}

    try:
        result["photo_path"] = validate_photo(photo_path)
    except InvalidPhotoError as e:
        errors.append(str(e))

    try:
        result["name"] = validate_name(name)
    except InvalidNameError as e:
        errors.append(str(e))

    try:
        bday, age = validate_birthday(birthday)
        result["birthday"] = bday
        result["age"] = age
    except (InvalidBirthdayError, AgeOutOfRangeError) as e:
        errors.append(str(e))

    if errors:
        if len(errors) == 1:
            raise ValidationError(errors[0])
        else:
            combined = "Multiple data payload schema exceptions flagged:\n" + "\n".join(
                f"  {i+1}. {err}" for i, err in enumerate(errors)
            )
            raise ValidationError(combined)

    return result
