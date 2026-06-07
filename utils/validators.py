import re
from datetime import datetime
from pathlib import Path
from typing import Union

from config.settings import INPUT_VALIDATION
from utils.exceptions import (
    InvalidPhotoError,
    InvalidNameError,
    InvalidBirthdayError,
    AgeOutOfRangeError,
    ValidationError,
)

def validate_photo(photo_path: Union[str, Path]) -> Path:
    path = Path(photo_path)

    if not path.exists():
        raise InvalidPhotoError(
            f"File nuk ekziston: '{path}'",
            photo_path=str(path)
        )

    if not path.is_file():
        raise InvalidPhotoError(
            f"Rruga nuk është file: '{path}'",
            photo_path=str(path)
        )

    allowed_formats = INPUT_VALIDATION["allowed_image_formats"]
    suffix = path.suffix.lower()
    if suffix not in allowed_formats:
        raise InvalidPhotoError(
            f"Format i palejuar '{suffix}'. "
            f"Formatet e lejuara: {', '.join(allowed_formats)}",
            photo_path=str(path)
        )

    max_size_bytes = INPUT_VALIDATION["max_image_size_mb"] * 1024 * 1024
    file_size = path.stat().st_size

    if file_size == 0:
        raise InvalidPhotoError(
            "File është bosh (0 bytes)",
            photo_path=str(path)
        )

    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        raise InvalidPhotoError(
            f"File është shumë i madh ({size_mb:.1f}MB). "
            f"Maksimumi i lejuar: {INPUT_VALIDATION['max_image_size_mb']}MB",
            photo_path=str(path)
        )

    _validate_image_header(path)

    _validate_image_resolution(path)

    return path.resolve()


def _validate_image_header(path: Path) -> None:
    magic_bytes = {
        b'\xff\xd8\xff'        : "JPEG",
        b'\x89PNG\r\n\x1a\n'  : "PNG",
        b'RIFF'                : "WEBP",
    }

    try:
        with open(path, 'rb') as f:
            header = f.read(12)

        valid = False
        for magic, fmt in magic_bytes.items():
            if header[:len(magic)] == magic:
                valid = True
                break
            if magic == b'RIFF' and header[8:12] == b'WEBP':
                valid = True
                break

        if not valid:
            raise InvalidPhotoError(
                "File nuk është imazh i vlefshëm. "
                "Sigurohu që file-i është vërtet JPG, PNG ose WEBP.",
                photo_path=str(path)
            )

    except (IOError, OSError) as e:
        raise InvalidPhotoError(
            f"Nuk mund të lexohet file-i: {e}",
            photo_path=str(path)
        )

def _validate_image_resolution(path: Path) -> None:
    """Kontrollo rezolucionin minimal të imazhit."""
    try:
        suffix = path.suffix.lower()
        min_w, min_h = INPUT_VALIDATION["min_image_resolution"]

        with open(path, 'rb') as f:
            data = f.read(24)

        width, height = None, None

        if suffix in ['.jpg', '.jpeg']:
            with open(path, 'rb') as f:
                content = f.read()
            i = 0
            while i < len(content) - 1:
                if content[i] == 0xFF:
                    marker = content[i+1]
                    if marker in [0xC0, 0xC2]:
                        height = int.from_bytes(content[i+5:i+7], 'big')
                        width  = int.from_bytes(content[i+7:i+9], 'big')
                        break
                    elif marker in [0xD8, 0xD9, 0xDA]:
                        i += 2
                        continue
                    else:
                        length = int.from_bytes(content[i+2:i+4], 'big')
                        i += 2 + length
                        continue
                i += 1

        elif suffix == '.png':
            if data[:8] == b'\x89PNG\r\n\x1a\n':
                width  = int.from_bytes(data[16:20], 'big')
                height = int.from_bytes(data[20:24], 'big')

        if width and height:
            if width < min_w or height < min_h:
                raise InvalidPhotoError(
                    f"Imazhi është shumë i vogël ({width}x{height}px). "
                    f"Minimumi i kërkuar: {min_w}x{min_h}px. "
                    f"Ju lutem dërgoni foto me cilësi më të lartë.",
                    photo_path=str(path)
                )

    except InvalidPhotoError:
        raise
    except Exception:
        pass

def validate_name(name: str) -> str:
    if not name:
        raise InvalidNameError("Emri nuk mund të jetë bosh.", name=name)

    name = name.strip()

    if not name:
        raise InvalidNameError(
            "Emri nuk mund të përbëhet vetëm nga hapësira.",
            name=name
        )

    max_length = INPUT_VALIDATION["max_name_length"]
    if len(name) > max_length:
        raise InvalidNameError(
            f"Emri është shumë i gjatë ({len(name)} karaktere). "
            f"Maksimumi: {max_length} karaktere.",
            name=name
        )

    if len(name) < 2:
        raise InvalidNameError(
            "Emri duhet të ketë të paktën 2 karaktere.",
            name=name
        )

    valid_pattern = re.compile(
        r"^[a-zA-Zàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
        r"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ"
        r"çëÇËšžŠŽ"
        r"\s\-\'\`]+"
        r"$",
        re.UNICODE
    )

    if not valid_pattern.match(name):
        raise InvalidNameError(
            "Emri përmban karaktere të palejuara. "
            "Lejohen vetëm shkronja, hapësira dhe vizë.",
            name=name
        )

    if any(c.isdigit() for c in name):
        raise InvalidNameError(
            "Emri nuk mund të përmbajë numra.",
            name=name
        )

    name_formatted = " ".join(
        word.capitalize() for word in name.split()
    )

    return name_formatted

def validate_birthday(birthday: str) -> tuple[datetime, int]:
    if not birthday:
        raise InvalidBirthdayError(
            "Datëlindja nuk mund të jetë bosh.",
            birthday=birthday
        )

    birthday = birthday.strip()

    if not re.match(r'^\d{4}-\d{2}-\d{2}$', birthday):
        raise InvalidBirthdayError(
            f"Format i gabuar: '{birthday}'. "
            f"Përdor formatin YYYY-MM-DD, p.sh: 2018-05-10",
            birthday=birthday
        )

    try:
        bday = datetime.strptime(birthday, "%Y-%m-%d")
    except ValueError as e:
        raise InvalidBirthdayError(
            f"Data '{birthday}' nuk është e vlefshme: {e}",
            birthday=birthday
        )

    today = datetime.today()

    if bday.date() > today.date():
        raise InvalidBirthdayError(
            f"Datëlindja '{birthday}' është në të ardhmen. "
            f"Sot është {today.strftime('%Y-%m-%d')}.",
            birthday=birthday
        )

    if (today - bday).days > 100 * 365:
        raise InvalidBirthdayError(
            f"Datëlindja '{birthday}' është shumë e vjetër.",
            birthday=birthday
        )

    age = (
        today.year - bday.year
        - ((today.month, today.day) < (bday.month, bday.day))
    )

    min_age = INPUT_VALIDATION["min_age_years"]
    max_age = INPUT_VALIDATION["max_age_years"]

    if age < min_age or age > max_age:
        raise AgeOutOfRangeError(age, min_age, max_age)

    return bday, age

def validate_inputs(
    photo_path: Union[str, Path],
    name: str,
    birthday: str,
) -> dict:
    errors = []
    result = {}

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
            combined = "Gabime të shumta validimi:\n" + "\n".join(
                f"  {i+1}. {err}" for i, err in enumerate(errors)
            )
            raise ValidationError(combined)

    return result