import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class TestValidateFace:
    def test_invalid_photo_path(self):
        from utils.exceptions import InvalidPhotoError
        from utils.validators import validate_photo
        with pytest.raises(InvalidPhotoError):
            validate_photo("/path/qe/nuk/ekziston.jpg")

    def test_invalid_format(self, tmp_path):
        from utils.exceptions import InvalidPhotoError
        from utils.validators import validate_photo
        f = tmp_path / "foto.bmp"
        f.write_bytes(b"fake")
        with pytest.raises(InvalidPhotoError):
            validate_photo(f)

    def test_empty_file(self, tmp_path):
        from utils.exceptions import InvalidPhotoError
        from utils.validators import validate_photo
        f = tmp_path / "foto.jpg"
        f.write_bytes(b"")
        with pytest.raises(InvalidPhotoError):
            validate_photo(f)

    def test_valid_jpeg(self, tmp_path):
        from utils.validators import validate_photo
        f = tmp_path / "foto.jpg"
        f.write_bytes(b'\xff\xd8\xff' + b'\x00' * 1000)
        result = validate_photo(f)
        assert result == f.resolve()

class TestFaceProcessorAugmentation:
    def test_aug_flip(self):
        import cv2
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]

        flipped = cv2.flip(img, 1)

        assert flipped[0, 99, 0] == 255
        assert flipped[0, 0, 0] == 0

    def test_aug_rotation(self):
        import cv2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        assert rotated.shape == (100, 100, 3)

    def test_face_size_calculation(self):
        class MockFace:
            bbox = np.array([10, 20, 110, 120])

        from pipeline.face_processor import FaceProcessor
        fp = object.__new__(FaceProcessor)
        size = fp._get_face_size(MockFace())
        assert size == 100.0

    def test_face_area_calculation(self):
        class MockFace:
            bbox = np.array([0, 0, 50, 80])

        from pipeline.face_processor import FaceProcessor
        fp = object.__new__(FaceProcessor)
        area = fp._get_face_area(MockFace())
        assert area == 4000.0