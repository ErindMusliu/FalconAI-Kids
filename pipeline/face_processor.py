from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import FACE_CONFIG
from utils.logger import get_logger
from utils.exceptions import (
    FaceProcessingError,
    FaceNotDetectedError,
    MultipleFacesError,
    InvalidPhotoError,
)


logger = get_logger(__name__)


class FaceProcessor:
    """
    Lightweight CPU-only face processor.

    This implementation intentionally avoids:
        - CUDA
        - InsightFace
        - ONNX GPU providers
        - large face-recognition models

    OpenCV Haar Cascade is used for face detection.

    The processor is responsible for:
        1. Loading the user's image.
        2. Detecting faces.
        3. Selecting the most relevant face.
        4. Cropping the face.
        5. Creating an aligned reference image.
        6. Creating lightweight augmentations.
        7. Returning metadata expected by downstream pipeline stages.

    IMPORTANT:
        The `embedding` field is retained for compatibility with the
        existing pipeline, but it is NOT a semantic face embedding.
        It is a lightweight normalized visual descriptor generated from
        the processed face crop.
    """

    def __init__(self):
        self.model = None

        self.det_thresh = float(
            FACE_CONFIG.get(
                "det_thresh",
                0.5,
            )
        )

        self.min_size = int(
            FACE_CONFIG.get(
                "min_face_size",
                60,
            )
        )

        self.max_faces_allowed = int(
            FACE_CONFIG.get(
                "max_faces_allowed",
                3,
            )
        )

        self.n_augmentations = int(
            FACE_CONFIG.get(
                "n_augmentations",
                4,
            )
        )

        self.scale_factor = float(
            FACE_CONFIG.get(
                "scale_factor",
                1.1,
            )
        )

        self.min_neighbors = int(
            FACE_CONFIG.get(
                "min_neighbors",
                5,
            )
        )

        self._load_model()

    # ------------------------------------------------------------------
    # MODEL INITIALIZATION
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Loads OpenCV's built-in Haar Cascade.

        No external AI model is required.
        """

        try:
            cascade_path = cv2.data.haarcascades + (
                "haarcascade_frontalface_default.xml"
            )

            cascade = cv2.CascadeClassifier(
                cascade_path
            )

            if cascade.empty():
                raise FaceProcessingError(
                    "OpenCV Haar Cascade could not be loaded."
                )

            self.model = cascade

            logger.success(
                "CPU face detector initialized successfully "
                "(OpenCV Haar Cascade)."
            )

        except FaceProcessingError:
            raise

        except Exception as e:
            raise FaceProcessingError(
                f"Failed to initialize CPU face detector: {e}"
            )

    # ------------------------------------------------------------------
    # MAIN PROCESSING
    # ------------------------------------------------------------------

    def process(
        self,
        photo_path: Path,
        temp_dir: Path,
    ) -> dict:
        """
        Process one portrait image.

        Returns a dictionary compatible with the previous
        InsightFace-based implementation.
        """

        photo_path = Path(photo_path)
        temp_dir = Path(temp_dir)

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.debug(
            f"Processing input portrait photo: {photo_path}"
        )

        image = self._load_image(
            photo_path
        )

        height, width = image.shape[:2]

        logger.debug(
            f"Image loaded successfully | "
            f"Dimensions: {width}x{height}px"
        )

        faces = self._detect_faces(
            image
        )

        self._validate_faces(
            faces
        )

        face = self._select_best_face(
            faces
        )

        x, y, w, h = face

        logger.debug(
            f"Target face selected | "
            f"bbox: {[x, y, x + w, y + h]}"
        )

        crop_path = (
            temp_dir /
            "face_cropped.png"
        )

        crop_image = self._crop_face(
            image,
            face,
            padding=0.35,
        )

        self._save_image(
            crop_image,
            crop_path,
        )

        aligned_path = self._get_aligned_face(
            image,
            face,
            temp_dir,
        )

        reference_image_path = (
            aligned_path
            if aligned_path is not None
            else crop_path
        )

        if aligned_path is not None:
            logger.debug(
                "Aligned face generated successfully."
            )
        else:
            logger.debug(
                "Alignment unavailable; "
                "using padded face crop."
            )

        embedding = self._extract_embedding(
            crop_image
        )

        augmented_paths = self._augment_face(
            face_image=crop_image,
            temp_dir=temp_dir,
        )

        logger.debug(
            f"Generated {len(augmented_paths)} "
            f"lightweight face variations."
        )

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        result = {
            "embedding": embedding,

            "face_image_path": reference_image_path,

            "crop_path": crop_path,

            "aligned_path": aligned_path,

            "augmented_paths": augmented_paths,

            "bbox": [
                x1,
                y1,
                x2,
                y2,
            ],

            # Haar does not produce a true confidence score.
            "det_score": 1.0,

            # Haar does not produce facial landmarks.
            "landmarks": None,

            "original_size": (
                width,
                height,
            ),

            "face_size": (
                crop_image.shape[1],
                crop_image.shape[0],
            ),
        }

        logger.success(
            "CPU face processing completed | "
            f"Face size: {result['face_size']} | "
            f"Reference: "
            f"{'aligned' if aligned_path else 'cropped'}"
        )

        return result

    # ------------------------------------------------------------------
    # IMAGE LOADING
    # ------------------------------------------------------------------

    def _load_image(
        self,
        photo_path: Path,
    ) -> np.ndarray:
        """
        Loads image safely using OpenCV.
        """

        if not photo_path.exists():
            raise InvalidPhotoError(
                "Image file does not exist.",
                photo_path=str(photo_path),
            )

        try:
            img = cv2.imread(
                str(photo_path),
                cv2.IMREAD_UNCHANGED,
            )

            if img is None:
                raw = np.fromfile(
                    str(photo_path),
                    dtype=np.uint8,
                )

                img = cv2.imdecode(
                    raw,
                    cv2.IMREAD_UNCHANGED,
                )

            if img is None:
                raise InvalidPhotoError(
                    "Image file could not be read. "
                    "The file may be corrupted or empty.",
                    photo_path=str(photo_path),
                )

            # BGRA -> BGR
            if (
                img.ndim == 3
                and img.shape[2] == 4
            ):
                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGRA2BGR,
                )

            # Grayscale -> BGR
            elif img.ndim == 2:
                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_GRAY2BGR,
                )

            if img.shape[0] < 32 or img.shape[1] < 32:
                raise InvalidPhotoError(
                    "Image resolution is too small for reliable "
                    "face processing.",
                    photo_path=str(photo_path),
                )

            return img

        except (
            InvalidPhotoError,
            FaceProcessingError,
        ):
            raise

        except Exception as e:
            raise FaceProcessingError(
                f"Failed during image loading: {e}"
            )

    # ------------------------------------------------------------------
    # FACE DETECTION
    # ------------------------------------------------------------------

    def _detect_faces(
        self,
        image: np.ndarray,
    ) -> list:
        """
        Detect faces using OpenCV Haar Cascade.

        Returns:
            list of tuples:
                (x, y, width, height)
        """

        if self.model is None:
            raise FaceProcessingError(
                "Face detector has not been initialized."
            )

        try:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            # Improve detection consistency.
            gray = cv2.equalizeHist(
                gray
            )

            faces = self.model.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(
                    self.min_size,
                    self.min_size,
                ),
            )

            faces = [
                tuple(map(int, face))
                for face in faces
                if face[2] >= self.min_size
                and face[3] >= self.min_size
            ]

            if faces:
                logger.debug(
                    f"Detected {len(faces)} "
                    f"face candidate(s)."
                )

                return faces

            logger.debug(
                "No face found at original resolution. "
                "Trying CPU upscale fallback."
            )

            return self._detect_with_upscale(
                image
            )

        except Exception as e:
            raise FaceProcessingError(
                f"Face detector failed: {e}"
            )

    def _detect_with_upscale(
        self,
        image: np.ndarray,
    ) -> list:
        """
        Second-pass detection on a 2x CPU-upscaled image.

        Coordinates are converted back to original resolution.
        """

        try:
            height, width = image.shape[:2]

            upscaled = cv2.resize(
                image,
                (
                    width * 2,
                    height * 2,
                ),
                interpolation=cv2.INTER_CUBIC,
            )

            gray = cv2.cvtColor(
                upscaled,
                cv2.COLOR_BGR2GRAY,
            )

            gray = cv2.equalizeHist(
                gray
            )

            faces = self.model.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=max(
                    4,
                    self.min_neighbors - 1,
                ),
                minSize=(
                    self.min_size * 2,
                    self.min_size * 2,
                ),
            )

            result = []

            for x, y, w, h in faces:
                x = int(x / 2)
                y = int(y / 2)
                w = int(w / 2)
                h = int(h / 2)

                if (
                    w >= self.min_size
                    and h >= self.min_size
                ):
                    result.append(
                        (
                            x,
                            y,
                            w,
                            h,
                        )
                    )

            return result

        except Exception as e:
            logger.debug(
                f"Upscale face detection failed: {e}"
            )

            return []

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_faces(
        self,
        faces: list,
    ) -> None:
        """
        Validate detected faces.
        """

        if not faces:
            raise FaceNotDetectedError()

        if (
            self.max_faces_allowed > 0
            and len(faces) > self.max_faces_allowed
        ):
            raise MultipleFacesError(
                len(faces)
            )

    def _select_best_face(
        self,
        faces: list,
    ):
        """
        Select the largest detected face.

        For a portrait image, the largest face is normally the
        intended subject.
        """

        if len(faces) == 1:
            return faces[0]

        best = max(
            faces,
            key=self._get_face_area,
        )

        logger.debug(
            f"Multiple faces detected ({len(faces)}). "
            f"Selected largest face."
        )

        return best

    # ------------------------------------------------------------------
    # CROPPING
    # ------------------------------------------------------------------

    def _crop_face(
        self,
        image: np.ndarray,
        face,
        padding: float = 0.35,
    ) -> np.ndarray:
        """
        Crop face with surrounding context.

        Output is normalized to 512x512.
        """

        height, width = image.shape[:2]

        x, y, w, h = map(
            int,
            face,
        )

        pad_x = int(
            w * padding
        )

        pad_y = int(
            h * padding
        )

        x1 = max(
            0,
            x - pad_x,
        )

        y1 = max(
            0,
            y - pad_y,
        )

        x2 = min(
            width,
            x + w + pad_x,
        )

        y2 = min(
            height,
            y + h + pad_y,
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            raise FaceProcessingError(
                "Computed face crop is invalid."
            )

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            raise FaceProcessingError(
                "Face crop produced an empty image."
            )

        crop = cv2.resize(
            crop,
            (512, 512),
            interpolation=cv2.INTER_LANCZOS4,
        )

        return crop

    # ------------------------------------------------------------------
    # ALIGNMENT
    # ------------------------------------------------------------------

    def _get_aligned_face(
        self,
        image: np.ndarray,
        face,
        temp_dir: Path,
    ) -> Optional[Path]:
        """
        Creates a simple geometrically centered face reference.

        Unlike InsightFace alignment, this does not require facial
        landmarks or an external model.
        """

        try:
            crop = self._crop_face(
                image,
                face,
                padding=0.45,
            )

            aligned_path = (
                temp_dir /
                "face_aligned.png"
            )

            self._save_image(
                crop,
                aligned_path,
            )

            return aligned_path

        except Exception as e:
            logger.debug(
                f"CPU face alignment skipped: {e}"
            )

            return None

    # ------------------------------------------------------------------
    # LIGHTWEIGHT EMBEDDING
    # ------------------------------------------------------------------

    def _extract_embedding(
        self,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        Creates a lightweight visual descriptor.

        IMPORTANT:
            This is NOT a face-recognition embedding like ArcFace.

        It exists for compatibility with the existing pipeline and
        provides a deterministic compact representation of the face
        crop without requiring a neural network.
        """

        try:
            image = cv2.resize(
                face_image,
                (32, 32),
                interpolation=cv2.INTER_AREA,
            )

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            descriptor = gray.astype(
                np.float32
            ).flatten()

            descriptor -= np.mean(
                descriptor
            )

            norm = np.linalg.norm(
                descriptor
            )

            if norm > 1e-6:
                descriptor /= norm

            return descriptor

        except Exception as e:
            raise FaceProcessingError(
                f"Failed to create lightweight face descriptor: {e}"
            )

    # ------------------------------------------------------------------
    # AUGMENTATION
    # ------------------------------------------------------------------

    def _augment_face(
        self,
        face_image: np.ndarray,
        temp_dir: Path,
    ) -> list[Path]:
        """
        Creates lightweight CPU augmentations.

        These are useful for downstream image processing without
        loading a training model.
        """

        augmented_paths: list[Path] = []

        if self.n_augmentations <= 0:
            return augmented_paths

        aug_dir = (
            temp_dir /
            "augmented"
        )

        aug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        original_path = (
            aug_dir /
            "aug_0_original.png"
        )

        self._save_image(
            face_image,
            original_path,
        )

        augmented_paths.append(
            original_path
        )

        augmentations = [
            (
                "brightness",
                self._aug_brightness,
            ),
            (
                "flip",
                self._aug_flip,
            ),
            (
                "rotation",
                self._aug_rotation,
            ),
            (
                "contrast",
                self._aug_contrast,
            ),
        ]

        for index, (
            name,
            function,
        ) in enumerate(
            augmentations[
                :self.n_augmentations
            ],
            start=1,
        ):
            try:
                augmented = function(
                    face_image.copy()
                )

                path = (
                    aug_dir /
                    f"aug_{index}_{name}.png"
                )

                self._save_image(
                    augmented,
                    path,
                )

                augmented_paths.append(
                    path
                )

            except Exception as e:
                logger.debug(
                    f"Augmentation '{name}' failed: {e}"
                )

        return augmented_paths

    # ------------------------------------------------------------------
    # AUGMENTATION OPERATIONS
    # ------------------------------------------------------------------

    def _aug_brightness(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        ).astype(
            np.float32
        )

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] * 1.15,
            0,
            255,
        )

        return cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2BGR,
        )

    def _aug_flip(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        return cv2.flip(
            image,
            1,
        )

    def _aug_rotation(
        self,
        image: np.ndarray,
        angle: float = 8.0,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        center = (
            width // 2,
            height // 2,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )

    def _aug_contrast(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.float32
        )

        lab[:, :, 0] = np.clip(
            lab[:, :, 0] * 1.1,
            0,
            255,
        )

        return cv2.cvtColor(
            lab.astype(np.uint8),
            cv2.COLOR_LAB2BGR,
        )

    # ------------------------------------------------------------------
    # FILE OPERATIONS
    # ------------------------------------------------------------------

    def _save_image(
        self,
        image: np.ndarray,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(path),
            image,
        )

        if not success:
            raise FaceProcessingError(
                f"Failed to write image: {path}"
            )

    # ------------------------------------------------------------------
    # FACE GEOMETRY
    # ------------------------------------------------------------------

    def _get_face_size(
        self,
        face,
    ) -> float:
        """
        Returns average face dimension.
        """

        _, _, width, height = face

        return (
            float(width)
            + float(height)
        ) / 2.0

    def _get_face_area(
        self,
        face,
    ) -> float:
        """
        Returns bounding-box area.
        """

        _, _, width, height = face

        return float(
            width * height
        )
