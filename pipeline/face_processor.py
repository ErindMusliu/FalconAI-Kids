import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from config.settings import FACE_CONFIG, DEVICE
from utils.logger import get_logger
from utils.exceptions import (
    FaceProcessingError,
    FaceNotDetectedError,
    MultipleFacesError,
    InvalidPhotoError,
)

logger = get_logger(__name__)

class FaceProcessor:
    def __init__(self):
        self.model = None
        self.det_thresh = FACE_CONFIG["det_thresh"]
        self.min_size = FACE_CONFIG["min_face_size"]
        self.max_faces_allowed = FACE_CONFIG.get("max_faces_allowed", 3)
        self.n_augmentations = FACE_CONFIG.get("n_augmentations", 4)
        self._load_model()

    def _load_model(self) -> None:
        try:
            import insightface
            from insightface.app import FaceAnalysis

            root_dir = FACE_CONFIG.get("root_dir", str(Path.home() / ".insightface"))

            logger.debug(
                f"Loading InsightFace | "
                f"model: {FACE_CONFIG['model_name']} | "
                f"device: {DEVICE} | root: {root_dir}"
            )

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if DEVICE == "cuda"
                else ["CPUExecutionProvider"]
            )

            self.model = FaceAnalysis(
                name=FACE_CONFIG["model_name"],
                root=root_dir,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )

            self.model.prepare(
                ctx_id=FACE_CONFIG["ctx_id"],
                det_size=FACE_CONFIG["detection_size"],
                det_thresh=self.det_thresh,
            )

            logger.success("InsightFace engine successfully initialized")

        except ImportError:
            raise FaceProcessingError(
                "insightface library is not installed. "
                "Please run: pip install insightface onnxruntime"
                + ("-gpu" if DEVICE == "cuda" else "")
            )
        except Exception as e:
            raise FaceProcessingError(f"Error while loading InsightFace model: {e}")

    def process(
        self,
        photo_path: Path,
        temp_dir: Path,
    ) -> dict:
        logger.debug(f"Processing input portrait photo: {photo_path}")

        image = self._load_image(photo_path)
        h, w = image.shape[:2]
        logger.debug(f"Image successfully read | Dimensions: {w}x{h}px")

        faces = self._detect_faces(image)
        self._validate_faces(faces)

        face = self._select_best_face(faces)
        logger.debug(
            f"Target face selected | "
            f"bbox: {face.bbox.astype(int).tolist()} | "
            f"confidence: {face.det_score:.3f}"
        )

        crop_path = temp_dir / "face_cropped.png"
        crop_image = self._crop_face(image, face, padding=0.3)
        self._save_image(crop_image, crop_path)

        embedding = self._extract_embedding(face)
        logger.debug(f"Face embedding extracted | Dimensions: {embedding.shape}")

        aligned_path = self._get_aligned_face(image, face, temp_dir)

        reference_image_path = aligned_path if aligned_path is not None else crop_path
        if aligned_path is not None:
            logger.debug("Using aligned face as the primary reference image for downstream generation.")
        else:
            logger.debug("Alignment unavailable; falling back to padded bounding-box crop as reference image.")

        augmented_paths = self._augment_face(
            face_image=crop_image,
            temp_dir=temp_dir,
        )
        logger.debug(f"Data augmentation completed | Generated {len(augmented_paths)} variations")

        result = {
            "embedding": embedding,
            "face_image_path": reference_image_path,
            "crop_path": crop_path,
            "aligned_path": aligned_path,
            "augmented_paths": augmented_paths,
            "bbox": face.bbox.astype(int).tolist(),
            "det_score": float(face.det_score),
            "landmarks": face.kps.tolist() if face.kps is not None else None,
            "original_size": (w, h),
            "face_size": (crop_image.shape[1], crop_image.shape[0]),
        }

        logger.success(
            f"Face pipeline processing finished | "
            f"Confidence Score: {face.det_score:.3f} | "
            f"Extracted Crop Size: {result['face_size']} | "
            f"Reference: {'aligned' if aligned_path else 'cropped'}"
        )

        return result

    def _load_image(self, photo_path: Path) -> np.ndarray:
        try:
            img = cv2.imread(str(photo_path))

            if img is None:
                img = cv2.imdecode(
                    np.fromfile(str(photo_path), dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

            if img is None:
                raise InvalidPhotoError(
                    "Image file could not be read. Ensure the file is not corrupted or empty.",
                    photo_path=str(photo_path)
                )

            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            return img

        except (InvalidPhotoError, FaceProcessingError):
            raise
        except Exception as e:
            raise FaceProcessingError(f"Failed during raw image acquisition load: {e}")

    def _detect_faces(self, image: np.ndarray) -> list:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.model.get(image_rgb)
        except Exception as e:
            raise FaceProcessingError(f"Face detector exception occurred during inference: {e}")

        faces = [
            f for f in faces
            if self._get_face_size(f) >= self.min_size
        ]

        if not faces:
            logger.debug("No faces found under default resolution layout. Attempting upscaled fallback scan...")
            faces = self._detect_with_upscale(image)

        if not faces:
            raise FaceNotDetectedError()

        logger.debug(f"Detected {len(faces)} qualified structural face candidate(s)")
        return faces

    def _detect_with_upscale(self, image: np.ndarray) -> list:
        try:
            h, w = image.shape[:2]
            upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            image_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
            faces = self.model.get(image_rgb)

            for face in faces:
                face.bbox /= 2
                if face.kps is not None:
                    face.kps /= 2

            return [
                f for f in faces
                if self._get_face_size(f) >= self.min_size
            ]
        except Exception:
            return []

    def _validate_faces(self, faces: list) -> None:
        if not faces:
            raise FaceNotDetectedError()

        high_conf_faces = [f for f in faces if f.det_score > 0.7]
        if len(high_conf_faces) > self.max_faces_allowed:
            raise MultipleFacesError(len(high_conf_faces))

    def _select_best_face(self, faces: list):
        if len(faces) == 1:
            return faces[0]

        best = max(faces, key=lambda f: self._get_face_area(f))
        logger.debug(
            f"Multiple candidates present ({len(faces)} total). "
            f"Selected largest bounding matrix area (Score: {best.det_score:.3f})"
        )
        return best

    def _crop_face(
        self,
        image: np.ndarray,
        face,
        padding: float = 0.3,
    ) -> np.ndarray:
        h, w = image.shape[:2]

        x1, y1, x2, y2 = map(int, map(round, face.bbox))

        face_w = x2 - x1
        face_h = y2 - y1
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            raise FaceProcessingError("Computed crop bounding slices yielded an empty or invalid matrix frame.")

        face_crop = image[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        return face_crop

    def _get_aligned_face(
        self,
        image: np.ndarray,
        face,
        temp_dir: Path,
    ) -> Optional[Path]:
        try:
            from insightface.utils import face_align

            if face.kps is None:
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            aligned_rgb = face_align.norm_crop(
                image_rgb,
                landmark=face.kps,
                image_size=512,
            )

            aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
            aligned_path = temp_dir / "face_aligned.png"
            cv2.imwrite(str(aligned_path), aligned_bgr)

            return aligned_path

        except Exception as e:
            logger.debug(f"Optional face affine alignment transformation skipped: {e}")
            return None

    def _extract_embedding(self, face) -> np.ndarray:
        if face.embedding is None:
            raise FaceProcessingError(
                "Face verification embedding vector is empty. "
                "Verify if model assets (buffalo_l bundle) are completely downloaded."
            )

        embedding = face.embedding.copy()

        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding

    def _augment_face(
        self,
        face_image: np.ndarray,
        temp_dir: Path,
    ) -> list[Path]:
        augmented_paths: list[Path] = []
        if self.n_augmentations <= 0:
            return augmented_paths

        aug_dir = temp_dir / "augmented"
        aug_dir.mkdir(exist_ok=True, parents=True)

        orig_path = aug_dir / "aug_0_original.png"
        self._save_image(face_image, orig_path)
        augmented_paths.append(orig_path)

        try:
            augmentations = [
                ("brightness", self._aug_brightness),
                ("flip", self._aug_flip),
                ("rotation", self._aug_rotation),
                ("contrast", self._aug_contrast),
            ]

            for i, (aug_name, aug_fn) in enumerate(augmentations[: self.n_augmentations], 1):
                aug_image = aug_fn(face_image.copy())
                aug_path = aug_dir / f"aug_{i}_{aug_name}.png"
                self._save_image(aug_image, aug_path)
                augmented_paths.append(aug_path)

        except Exception as e:
            logger.debug(f"Minor non-blocking failure encountered during augmentation iterations: {e}")

        return augmented_paths

    def _aug_brightness(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _aug_flip(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    def _aug_rotation(self, image: np.ndarray, angle: float = 8.0) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _aug_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.1, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _save_image(self, image: np.ndarray, path: Path) -> None:
        success = cv2.imwrite(str(path), image)
        if not success:
            raise FaceProcessingError(f"Disk write transaction failed targeting file location: {path}")

    def _get_face_size(self, face) -> float:
        x1, y1, x2, y2 = face.bbox
        return ((x2 - x1) + (y2 - y1)) / 2

    def _get_face_area(self, face) -> float:
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) * (y2 - y1)
