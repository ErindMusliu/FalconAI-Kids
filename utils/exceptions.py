from typing import Optional, Any

class FalconAIException(Exception):
    def __init__(self, message: str, code: str = "FalconAI_Error", details: Optional[dict] = None) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        base = f"[{self.code}] {self.message}"
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base} | {detail_str}"
        return base
    
    def to_dict(self) -> dict:
        return {
            "error_code": self.code,
            "message": self.message,
            "details": self.details
        }

class ValidationError(FalconAIException):
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None) -> None:
        details = {}
        if field:   
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        super().__init__(message, code="VALIDATION_ERROR", details=details)
        self.field = field
        self.value = value

class InvalidPhotoError(ValidationError):
    def __init__(self, reason: str, photo_path: Optional[str] = None) -> None:
        message = f"Invalid photo processing file asset: {reason}"
        super().__init__(message, field="photo", value=photo_path)
        self.code = "INVALID_PHOTO"


class InvalidNameError(ValidationError):
    def __init__(self, reason: str, name: Optional[str] = None) -> None:
        message = f"Invalid character composition or naming structure: {reason}"
        super().__init__(message, field="name", value=name)
        self.code = "INVALID_NAME"

class InvalidBirthdayError(ValidationError):
    def __init__(self, reason: str, birthday: Optional[str] = None) -> None:
        message = f"Invalid birthday value specification: {reason}"
        super().__init__(message, field="birthday", value=birthday)
        self.code = "INVALID_BIRTHDAY"


class AgeOutOfRangeError(ValidationError):
    def __init__(self, age: int, min_age: int, max_age: int) -> None:
        message = (
            f"The computed runtime age ({age}) falls outside allowable platform boundaries. "
            f"Expected constraints require a target range between {min_age} and {max_age} years old."
        )
        super().__init__(message, field="birthday", value=age)
        self.code = "AGE_OUT_OF_RANGE"
        self.age = age

class ModelLoadError(FalconAIException):
    def __init__(self, model_name: str, reason: str) -> None:
        message = f"Failed to instantiate machine learning model weights '{model_name}': {reason}"
        super().__init__(message, code="MODEL_LOAD_ERROR", details={"model": model_name, "reason": reason})
        self.model_name = model_name

class ModelNotFoundError(ModelLoadError):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None) -> None:
        reason = "The specified model identity could not be discovered inside local caches and remote downloads are disabled."
        super().__init__(model_name, reason)
        self.code = "MODEL_NOT_FOUND"
        if cache_dir:
            self.details["cache_dir"] = cache_dir


class ModelInferenceError(FalconAIException):
    def __init__(self, model_name: str, reason: str) -> None:
        message = f"Runtime model inference processing failure flagged inside token layers '{model_name}': {reason}"
        super().__init__(
            message,
            code="MODEL_INFERENCE_ERROR",
            details={"model": model_name, "reason": reason}
        )
        self.model_name = model_name


class PipelineError(FalconAIException):
    def __init__(self, message: str, step: Optional[str] = None, reason: Optional[str] = None) -> None:
        details = {}
        if step:
            details["step"] = step
        if reason:
            details["reason"] = reason
        super().__init__(message, code="PIPELINE_ERROR", details=details)
        self.step = step


class FaceProcessingError(PipelineError):
    def __init__(self, reason: str) -> None:
        message = f"Biometric tracking and face extraction routine experienced a breakdown: {reason}"
        super().__init__(message, step="face_processor", reason=reason)
        self.code = "FACE_PROCESSING_ERROR"

class FaceNotDetectedError(FaceProcessingError):
    def __init__(self) -> None:
        super().__init__(
            "No human facial coordinates discovered within the provided image array. "
            "Ensure the subject's posture is centered, clear, and illuminated properly."
        )
        self.code = "FACE_NOT_DETECTED"


class MultipleFacesError(FaceProcessingError):
    def __init__(self, count: int) -> None:
        super().__init__(
            f"Detected {count} human faces inside the processed photo matrix. "
            f"Please submit a portrait focusing exclusively on a single subject layout."
        )
        self.code = "MULTIPLE_FACES_DETECTED"
        self.details["face_count"] = count


class StoryGenerationError(PipelineError):
    def __init__(self, reason: str) -> None:
        message = f"Semantic text compilation error raised during prompt processing routine: {reason}"
        super().__init__(message, step="story_generator", reason=reason)
        self.code = "STORY_GENERATION_ERROR"


class FrameGenerationError(PipelineError):
    def __init__(self, reason: str, scene_index: Optional[int] = None) -> None:
        message = f"Diffusion graphics generation thread terminated with failure parameters: {reason}"
        super().__init__(message, step="frame_generator", reason=reason)
        self.code = "FRAME_GENERATION_ERROR"
        if scene_index is not None:
            self.details["scene_index"] = scene_index


class AudioGenerationError(PipelineError):
    def __init__(self, reason: str) -> None:
        message = f"Acoustic synthesis execution module failed to build clean audio blocks: {reason}"
        super().__init__(message, step="audio_generator", reason=reason)
        self.code = "AUDIO_GENERATION_ERROR"

class VideoAssemblyError(PipelineError):
    def __init__(self, reason: str) -> None:
        message = f"Media composition engine failed to bundle active components securely: {reason}"
        super().__init__(message, step="video_assembler", reason=reason)
        self.code = "VIDEO_ASSEMBLY_ERROR"


class UpscalingError(PipelineError):
    def __init__(self, reason: str) -> None:
        message = f"Visual resolution upscaling pass was aborted prematurely: {reason}"
        super().__init__(message, step="upscaler", reason=reason)
        self.code = "UPSCALING_ERROR"

class StorageError(FalconAIException):
    def __init__(self, message: str, path: Optional[str] = None) -> None:
        details = {"path": path} if path else {}
        super().__init__(message, code="STORAGE_ERROR", details=details)


class LocalStorageFileNotFoundError(StorageError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Requested storage target object resource could not be found at path location: {path}", path=path)
        self.code = "FILE_NOT_FOUND"

class DiskSpaceError(StorageError):
    def __init__(self, required_gb: float, available_gb: float) -> None:
        message = (
            f"Insufficient persistent storage disk space available to handle write loops. "
            f"Required baseline allocation space: {required_gb:.1f} GB, "
            f"Discovered system workspace space: {available_gb:.1f} GB"
        )
        super().__init__(message)
        self.code = "DISK_SPACE_ERROR"
        self.details["required_gb"] = required_gb
        self.details["available_gb"] = available_gb

class S3UploadError(StorageError):
    def __init__(self, bucket: str, key: str, reason: str) -> None:
        message = f"Failed to pipe local data assets to remote cloud cluster s3://{bucket}/{key}: {reason}"
        super().__init__(message)
        self.code = "S3_UPLOAD_ERROR"
        self.details.update({"bucket": bucket, "key": key, "reason": reason})

class GPUError(FalconAIException):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="GPU_ERROR")

class CUDANotAvailableError(GPUError):
    def __init__(self) -> None:
        super().__init__(
            "NVIDIA CUDA hardware acceleration layer is inaccessible inside this host context. "
            "Ensure functional drivers are installed or modify configuration parameters to DEVICE=cpu inside your .env configuration file."
        )
        self.code = "CUDA_NOT_AVAILABLE"

class OutOfMemoryError(GPUError):
    def __init__(self, required_gb: Optional[float] = None) -> None:
        msg = "Hardware engine processing halted; graphic processing unit memory space (VRAM) is completely exhausted."
        if required_gb:
            msg += f" Processing step execution limits mandate a fallback minimum of {required_gb} GB free VRAM blocks."
        msg += " Lower render resolution scaling dimensions or execute inference cycles using system CPU mapping routes instead."
        super().__init__(msg)
        self.code = "OUT_OF_MEMORY"

def handle_exception(exc: Exception, logger: Optional[Any] = None) -> FalconAIException:
    if isinstance(exc, FalconAIException):
        return exc

    message = str(exc)
    exc_type = type(exc).__name__

    if "CUDA out of memory" in message or "VRAM" in message:
        return OutOfMemoryError()

    if "CUDA" in message and "not available" in message:
        return CUDANotAvailableError()

    if isinstance(exc, (IOError, OSError)) or "No such file" in message:
        return StorageError(f"Operating system file system hardware boundary conflict flagged: {message}")

    wrapped = FalconAIException(
        message=f"An unmapped, unexpected system exception occurred ({exc_type}): {message}",
        code="UNEXPECTED_ERROR",
        details={"original_type": exc_type}
    )

    if logger:
        logger.exception(f"Unexpected underlying system breakdown captured cleanly: {exc}")

    return wrapped

class FalconAIError(FalconAIException):
    pass
