"""
FalconAI-Kids exception hierarchy.

Centralized application exceptions used across the project.

Design goals:
- Clear and predictable exception hierarchy
- Stable machine-readable error codes
- Structured error details
- Safe serialization for APIs/logging
- Backward compatibility with existing imports
- CPU-first architecture with optional GPU-related errors
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class FalconAIException(Exception):
    """Base exception for all FalconAI-Kids application errors."""

    default_code = "FALCONAI_ERROR"

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.message = str(message)
        self.code = code or self.default_code
        self.details: dict[str, Any] = dict(details or {})

        super().__init__(self.message)

    def __str__(self) -> str:
        base = f"[{self.code}] {self.message}"

        if not self.details:
            return base

        details = ", ".join(
            f"{key}={value}"
            for key, value in self.details.items()
        )

        return f"{base} | {details}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the exception."""
        return {
            "error_code": self.code,
            "message": self.message,
            "details": self.details,
            "exception": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(FalconAIException):
    """Base exception for invalid user or pipeline input."""

    default_code = "VALIDATION_ERROR"

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
    ) -> None:
        details: dict[str, Any] = {}

        if field:
            details["field"] = field

        if value is not None:
            details["value"] = str(value)

        super().__init__(
            message,
            code=self.default_code,
            details=details,
        )

        self.field = field
        self.value = value


class InvalidPhotoError(ValidationError):
    """Raised when a supplied photo cannot be processed."""

    default_code = "INVALID_PHOTO"

    def __init__(
        self,
        reason: str,
        photo_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            f"Invalid photo asset: {reason}",
            field="photo",
            value=photo_path,
        )
        self.code = self.default_code


class InvalidNameError(ValidationError):
    """Raised when a character/name value is invalid."""

    default_code = "INVALID_NAME"

    def __init__(
        self,
        reason: str,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            f"Invalid name: {reason}",
            field="name",
            value=name,
        )
        self.code = self.default_code


class InvalidBirthdayError(ValidationError):
    """Raised when a birthday value is invalid."""

    default_code = "INVALID_BIRTHDAY"

    def __init__(
        self,
        reason: str,
        birthday: Optional[str] = None,
    ) -> None:
        super().__init__(
            f"Invalid birthday: {reason}",
            field="birthday",
            value=birthday,
        )
        self.code = self.default_code


class AgeOutOfRangeError(ValidationError):
    """Raised when an age falls outside the supported range."""

    default_code = "AGE_OUT_OF_RANGE"

    def __init__(
        self,
        age: int,
        min_age: int,
        max_age: int,
    ) -> None:
        message = (
            f"Age {age} is outside the supported range "
            f"({min_age}-{max_age})."
        )

        super().__init__(
            message,
            field="birthday",
            value=age,
        )

        self.code = self.default_code
        self.age = age
        self.min_age = min_age
        self.max_age = max_age

        self.details.update(
            {
                "age": age,
                "min_age": min_age,
                "max_age": max_age,
            }
        )


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------

class ModelLoadError(FalconAIException):
    """Raised when a model cannot be loaded."""

    default_code = "MODEL_LOAD_ERROR"

    def __init__(
        self,
        model_name: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            code=self.default_code,
            details={
                "model": model_name,
                "reason": reason,
            },
        )

        self.model_name = model_name
        self.reason = reason


class ModelNotFoundError(ModelLoadError):
    """Raised when a requested model is unavailable."""

    default_code = "MODEL_NOT_FOUND"

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        reason = (
            "The requested model was not found in the available "
            "local model directories."
        )

        super().__init__(model_name, reason)

        self.code = self.default_code

        if cache_dir:
            self.details["cache_dir"] = cache_dir


class ModelInferenceError(FalconAIException):
    """Raised when model inference fails."""

    default_code = "MODEL_INFERENCE_ERROR"

    def __init__(
        self,
        model_name: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"Model inference failed for '{model_name}': {reason}",
            code=self.default_code,
            details={
                "model": model_name,
                "reason": reason,
            },
        )

        self.model_name = model_name
        self.reason = reason


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineError(FalconAIException):
    """Base exception for pipeline-stage failures."""

    default_code = "PIPELINE_ERROR"

    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        details: dict[str, Any] = {}

        if step:
            details["step"] = step

        if reason:
            details["reason"] = reason

        super().__init__(
            message,
            code=self.default_code,
            details=details,
        )

        self.step = step
        self.reason = reason


class FaceProcessingError(PipelineError):
    """Base exception for face-processing failures."""

    default_code = "FACE_PROCESSING_ERROR"

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Face processing failed: {reason}",
            step="face_processor",
            reason=reason,
        )
        self.code = self.default_code


class FaceNotDetectedError(FaceProcessingError):
    """Raised when no face can be detected."""

    default_code = "FACE_NOT_DETECTED"

    def __init__(self) -> None:
        super().__init__(
            "No human face was detected in the provided image."
        )
        self.code = self.default_code


class MultipleFacesError(FaceProcessingError):
    """Raised when more than one face is detected."""

    default_code = "MULTIPLE_FACES_DETECTED"

    def __init__(self, count: int) -> None:
        super().__init__(
            f"Detected {count} faces. Exactly one face is required."
        )

        self.code = self.default_code
        self.face_count = count
        self.details["face_count"] = count


class StoryGenerationError(PipelineError):
    """Raised when story generation fails."""

    default_code = "STORY_GENERATION_ERROR"

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Story generation failed: {reason}",
            step="story_generator",
            reason=reason,
        )
        self.code = self.default_code


class FrameGenerationError(PipelineError):
    """Raised when scene/frame generation fails."""

    default_code = "FRAME_GENERATION_ERROR"

    def __init__(
        self,
        reason: str,
        scene_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"Frame generation failed: {reason}",
            step="frame_generator",
            reason=reason,
        )

        self.code = self.default_code
        self.scene_index = scene_index

        if scene_index is not None:
            self.details["scene_index"] = scene_index


# ---------------------------------------------------------------------------
# Character animation
# ---------------------------------------------------------------------------

class CharacterAnimationError(PipelineError):
    """Base exception for character animation failures."""

    default_code = "CHARACTER_ANIMATION_ERROR"

    def __init__(
        self,
        reason: str,
        scene_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"Character animation failed: {reason}",
            step="character_animator",
            reason=reason,
        )

        self.code = self.default_code
        self.scene_index = scene_index

        if scene_index is not None:
            self.details["scene_index"] = scene_index


class TalkingHeadGenerationError(CharacterAnimationError):
    """Raised when talking-head generation fails."""

    default_code = "TALKING_HEAD_GENERATION_ERROR"

    def __init__(
        self,
        reason: str,
        scene_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"Talking-head generation failed: {reason}",
            scene_index=scene_index,
        )
        self.code = self.default_code


class MouthAnimationError(CharacterAnimationError):
    """Raised when procedural mouth animation fails."""

    default_code = "MOUTH_ANIMATION_ERROR"

    def __init__(
        self,
        reason: str,
        scene_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"Procedural mouth animation failed: {reason}",
            scene_index=scene_index,
        )
        self.code = self.default_code


class AnimationCompositingError(CharacterAnimationError):
    """Raised when animated output cannot be composited."""

    default_code = "ANIMATION_COMPOSITING_ERROR"

    def __init__(
        self,
        reason: str,
        scene_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"Animation compositing failed: {reason}",
            scene_index=scene_index,
        )
        self.code = self.default_code


# ---------------------------------------------------------------------------
# Audio / Video
# ---------------------------------------------------------------------------

class AudioGenerationError(PipelineError):
    """Raised when narration/audio generation fails."""

    default_code = "AUDIO_GENERATION_ERROR"

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Audio generation failed: {reason}",
            step="audio_generator",
            reason=reason,
        )
        self.code = self.default_code


class AudioAnalysisError(PipelineError):
    """Raised when audio analysis fails."""

    default_code = "AUDIO_ANALYSIS_ERROR"

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Audio analysis failed: {reason}",
            step="audio_analyzer",
            reason=reason,
        )
        self.code = self.default_code


class VideoAssemblyError(PipelineError):
    """Raised when video assembly fails."""

    default_code = "VIDEO_ASSEMBLY_ERROR"

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Video assembly failed: {reason}",
            step="video_assembler",
            reason=reason,
        )
        self.code = self.default_code


class UpscalingError(PipelineError):
    """Raised when image/video upscaling fails."""

    default_code = "UPSCALING_ERROR"

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Upscaling failed: {reason}",
            step="upscaler",
            reason=reason,
        )
        self.code = self.default_code


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class StorageError(FalconAIException):
    """Base exception for storage/file-system failures."""

    default_code = "STORAGE_ERROR"

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
    ) -> None:
        details: dict[str, Any] = {}

        if path:
            details["path"] = path

        super().__init__(
            message,
            code=self.default_code,
            details=details,
        )

        self.path = path


class LocalStorageFileNotFoundError(StorageError):
    """Raised when a requested local file does not exist."""

    default_code = "FILE_NOT_FOUND"

    def __init__(self, path: str) -> None:
        super().__init__(
            f"File not found: {path}",
            path=path,
        )
        self.code = self.default_code


class DiskSpaceError(StorageError):
    """Raised when insufficient disk space is available."""

    default_code = "DISK_SPACE_ERROR"

    def __init__(
        self,
        required_gb: float,
        available_gb: float,
    ) -> None:
        super().__init__(
            (
                "Insufficient disk space. "
                f"Required: {required_gb:.2f} GB, "
                f"available: {available_gb:.2f} GB."
            )
        )

        self.code = self.default_code
        self.required_gb = required_gb
        self.available_gb = available_gb

        self.details.update(
            {
                "required_gb": required_gb,
                "available_gb": available_gb,
            }
        )


class S3UploadError(StorageError):
    """Raised when an S3 upload fails."""

    default_code = "S3_UPLOAD_ERROR"

    def __init__(
        self,
        bucket: str,
        key: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"S3 upload failed: {reason}",
        )

        self.code = self.default_code
        self.bucket = bucket
        self.key = key
        self.reason = reason

        self.details.update(
            {
                "bucket": bucket,
                "key": key,
                "reason": reason,
            }
        )


# ---------------------------------------------------------------------------
# GPU / hardware
# ---------------------------------------------------------------------------

class GPUError(FalconAIException):
    """Base exception for optional GPU acceleration failures."""

    default_code = "GPU_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code=self.default_code,
        )


class CUDANotAvailableError(GPUError):
    """Raised when CUDA acceleration is requested but unavailable."""

    default_code = "CUDA_NOT_AVAILABLE"

    def __init__(self) -> None:
        super().__init__(
            "CUDA acceleration is not available on this system."
        )
        self.code = self.default_code


class OutOfMemoryError(GPUError):
    """
    Raised when a hardware processing backend runs out of memory.

    This can represent VRAM exhaustion or, depending on the caller,
    another hardware-memory limitation.
    """

    default_code = "OUT_OF_MEMORY"

    def __init__(
        self,
        required_gb: Optional[float] = None,
    ) -> None:
        message = "The processing operation ran out of available memory."

        if required_gb is not None:
            message += (
                f" At least {required_gb:.2f} GB of additional "
                "memory may be required."
            )

        super().__init__(message)

        self.code = self.default_code
        self.required_gb = required_gb

        if required_gb is not None:
            self.details["required_gb"] = required_gb


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------

def handle_exception(
    exc: Exception,
    logger: Optional[Any] = None,
) -> FalconAIException:
    """
    Normalize an arbitrary exception into FalconAIException.

    Existing FalconAI exceptions are returned unchanged.
    """

    if isinstance(exc, FalconAIException):
        return exc

    message = str(exc).strip()
    exc_type = type(exc).__name__
    normalized = message.lower()

    # GPU / CUDA errors.
    if (
        "cuda out of memory" in normalized
        or "out of memory" in normalized and "cuda" in normalized
        or "vram" in normalized
    ):
        return OutOfMemoryError()

    if (
        "cuda" in normalized
        and (
            "not available" in normalized
            or "unavailable" in normalized
            or "no cuda" in normalized
        )
    ):
        return CUDANotAvailableError()

    # File-system errors.
    if isinstance(exc, FileNotFoundError):
        return LocalStorageFileNotFoundError(
            getattr(exc, "filename", None) or message
        )

    if isinstance(exc, (IOError, OSError)):
        return StorageError(
            f"File-system operation failed: {message}"
        )

    # Generic fallback.
    wrapped = FalconAIException(
        message=(
            f"Unexpected {exc_type}: "
            f"{message or 'No error message was provided.'}"
        ),
        code="UNEXPECTED_ERROR",
        details={
            "original_type": exc_type,
        },
    )

    if logger is not None:
        try:
            logger.exception(
                "Unexpected exception captured: %s",
                exc,
            )
        except Exception:
            # Logging must never hide the original application error.
            pass

    return wrapped


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

class FalconAIError(FalconAIException):
    """Backward-compatible alias for FalconAIException."""

    pass
