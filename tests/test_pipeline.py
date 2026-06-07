import sys
from pathlib import Path
from datetime import datetime
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class TestExceptions:
    def test_validation_error_is_falconai(self):
        from utils.exceptions import ValidationError, FalconAIException
        err = ValidationError("test")
        assert isinstance(err, FalconAIException)

    def test_face_error_is_pipeline_error(self):
        from utils.exceptions import FaceNotDetectedError, PipelineError
        err = FaceNotDetectedError()
        assert isinstance(err, PipelineError)
        assert err.code == "FACE_NOT_DETECTED"

    def test_multiple_faces_error(self):
        from utils.exceptions import MultipleFacesError
        err = MultipleFacesError(5)
        assert "5" in str(err)
        assert err.details["face_count"] == 5

    def test_age_out_of_range(self):
        from utils.exceptions import AgeOutOfRangeError
        err = AgeOutOfRangeError(20, 1, 16)
        assert "20" in str(err)

    def test_handle_exception_cuda_oom(self):
        from utils.exceptions import handle_exception, OutOfMemoryError
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2GB")
        wrapped = handle_exception(exc)
        assert isinstance(wrapped, OutOfMemoryError)

    def test_handle_exception_cuda_unavailable(self):
        from utils.exceptions import handle_exception, CUDANotAvailableError
        exc = RuntimeError("CUDA is not available")
        wrapped = handle_exception(exc)
        assert isinstance(wrapped, CUDANotAvailableError)

    def test_to_dict(self):
        from utils.exceptions import FalconAIException
        err = FalconAIException("test", code="TEST_CODE", details={"k": "v"})
        d = err.to_dict()
        assert d["error_code"] == "TEST_CODE"
        assert d["details"]["k"] == "v"


class TestLogger:
    def test_get_logger_returns_same_instance(self):
        from utils.logger import get_logger
        l1 = get_logger("test.module")
        l2 = get_logger("test.module")
        assert l1 is l2

    def test_logger_has_custom_methods(self):
        from utils.logger import get_logger
        logger = get_logger("test.custom")
        assert hasattr(logger, "success")
        assert hasattr(logger, "step")

    def test_pipeline_formatter(self):
        from utils.logger import get_pipeline_formatter
        pf = get_pipeline_formatter(total_steps=5)
        assert pf.total_steps == 5
        assert pf.current_step == 0

class TestTextUtils:
    def test_clean_text_removes_emoji(self):
        from ml.preprocessing.text_utils import clean_text
        result = clean_text("Përshëndetje 🎉 botë!")
        assert "🎉" not in result
        assert "Përshëndetje" in result

    def test_sanitize_filename(self):
        from ml.preprocessing.text_utils import sanitize_filename
        result = sanitize_filename("Erind Musliu")
        assert " " not in result
        assert result == "erind_musliu"

    def test_split_sentences(self):
        from ml.preprocessing.text_utils import split_into_sentences
        text = "Fjali e parë. Fjali e dytë! Fjali e tretë?"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_format_duration(self):
        from ml.preprocessing.text_utils import format_duration
        assert format_duration(65)   == "01:05"
        assert format_duration(3661) == "01:01:01"
        assert format_duration(30)   == "00:30"

    def test_truncate_text(self):
        from ml.preprocessing.text_utils import truncate_text
        long_text = "A" * 100
        result    = truncate_text(long_text, max_chars=20)
        assert len(result) <= 20
        assert result.endswith("...")

    def test_extract_json_valid(self):
        from ml.preprocessing.text_utils import extract_json_from_text
        import json
        data   = {"key": "value", "num": 42}
        text   = f"Disa tekst: {json.dumps(data)} dhe më shumë tekst"
        result = extract_json_from_text(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["key"] == "value"

class TestImageUtils:
    def test_blend_images(self):
        import numpy as np
        from ml.preprocessing.image_utils import blend_images

        img_a = np.zeros((10, 10, 3), dtype=np.uint8)
        img_b = np.full((10, 10, 3), 200, dtype=np.uint8)

        blended = blend_images(img_a, img_b, alpha=0.5)
        assert blended.shape == (10, 10, 3)
        assert 90 < blended[5, 5, 0] < 110

    def test_blend_alpha_zero(self):
        import numpy as np
        from ml.preprocessing.image_utils import blend_images

        img_a = np.zeros((5, 5, 3), dtype=np.uint8)
        img_b = np.full((5, 5, 3), 255, dtype=np.uint8)

        result = blend_images(img_a, img_b, alpha=0.0)
        assert np.all(result == 0)

    def test_blend_alpha_one(self):
        import numpy as np
        from ml.preprocessing.image_utils import blend_images

        img_a = np.zeros((5, 5, 3), dtype=np.uint8)
        img_b = np.full((5, 5, 3), 255, dtype=np.uint8)

        result = blend_images(img_a, img_b, alpha=1.0)
        assert np.all(result == 255)

    def test_create_blank_image(self):
        import numpy as np
        from ml.preprocessing.image_utils import create_blank_image

        img = create_blank_image(100, 80, color=(255, 0, 0))
        assert img.shape == (80, 100, 3)
        assert img[0, 0, 0] == 255
        assert img[0, 0, 1] == 0

    def test_numpy_pil_roundtrip(self):
        import numpy as np
        from ml.preprocessing.image_utils import numpy_to_pil, pil_to_numpy

        original = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        pil_img  = numpy_to_pil(original)
        restored = pil_to_numpy(pil_img)

        assert restored.shape == original.shape
        assert np.allclose(original, restored, atol=1)