class FalconAIException(Exception):
    def __init__(self, message: str, code: str = "FalconAI_Error",details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        base = f"[{self.code}] {self.message}"

        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k,v in self.details.items())
            return f"{base} | {detail_str}"
        
        return base
    
    def to_dict(self) -> dict:
        return{
            "error_code": self.code,
            "message": self.message,
            "details": self.details
        }
    
class ValidationError(FalconAIException):
    def __init__(self, message: str, field: str = None, value = None):
        details = {}

        if field:   
            details["field"] = field

        if value is not None:
            details["value"] = str(value)

        super().__init__(message,code="VALIDATION_ERROR",details=details)
        self.field = field
        self.value = value

class InvalidPhotoError(ValidationError):
    def __init__(self, reason: str, photo_path: str = None):
        message = f"Invalid photo: {reason}"
        super().__init__(message,field="photo",value=photo_path)
        self.code = "INVALID_PHOTO"

class InvalidNameError(ValidationError):
    def __init__(self, reason: str, name: str = None):
        message = f"Invalid name: {reason}"
        super().__init__(message,field="name",value=name)
        self.code = "INVALID_NAME"

class InvalidBirthdayError(ValidationError):
    def __init__(self, reason: str, birthday: str = None):
        message = f"Invalid birthday: {reason}"
        super().__init__(message,field="birthday",value=birthday)
        self.code = "INVALID_BIRTHDAY"

class AgeOutOfRangeError(ValidationError):
    def __init__(self, age: int, min_age: int, max_age: int):
        message = (f"Mosha {age} vjeq eshte jashte kufijve.",f"Lejohet vetem {min_age}-{max_age} vjeq.")
        super().__init__(message, field="Birthday", value=age)
        self.code = "AGE_OUT_OF_RANGE"
        self.age = age

class ModelLoadError(FalconAIException):
    def __init__(self, model_name: str, reason: str):
        message = f"Nuk mund te ngarkohet modeli '{model_name}' : {reason}"
        super().__init__(message,code="MODEL_LOAD_ERROR",details={"model": model_name,"reason": reason})
        self.model_name = model_name


class ModelNotFoundError(ModelLoadError):
    def __init__(self, model_name: str, cache_dir: str = None):
        reason = "modeli nuk ekziston dhe nuk mund te shkarkohet"
        super().__init__(model_name, reason)
        self.code = "MODEL_NOT_FOUND"
        if cache_dir:
            self.details["cache_dir"] = cache_dir


class ModelInferenceError(FalconAIException):
    def __init__(self, model_name: str, reason: str):
        message = f"Gabim gjate inferimit te modelit '{model_name}': {reason}"
        super().__init__(
            message,
            code="MODEL_INFERENCE_ERROR",
            details={"model": model_name, "reason": reason}
        )
        self.model_name = model_name

class PipelineError(FalconAIException):
    def __init__(self, message: str, step: str = None, reason: str = None):
        details = {}
        if step:
            details["step"] = step
        if reason:
            details["reason"] = reason
        super().__init__(message, code="PIPELINE_ERROR", details=details)
        self.step = step


class FaceProcessingError(PipelineError):
    def __init__(self, reason: str):
        message = f"Gabim ne procesimin e fytyrës: {reason}"
        super().__init__(message, step="face_processor", reason=reason)
        self.code = "FACE_PROCESSING_ERROR"


class FaceNotDetectedError(FaceProcessingError):
    def __init__(self):
        super().__init__(
            "Asnje fytyre nuk u gjet ne foto. "
            "Sigurohu qe fytyra eshte e qarte dhe e ndriçuar mire."
        )
        self.code = "FACE_NOT_DETECTED"


class MultipleFacesError(FaceProcessingError):
    def __init__(self, count: int):
        super().__init__(
            f"U gjetën {count} fytyra ne foto. "
            f"Ju lutem dërgoni foto ku shihet vetem fytyra e femijes."
        )
        self.code = "MULTIPLE_FACES_DETECTED"
        self.details["face_count"] = count


class StoryGenerationError(PipelineError):
    def __init__(self, reason: str):
        message = f"Gabim ne gjenerimin e historisë: {reason}"
        super().__init__(message, step="story_generator", reason=reason)
        self.code = "STORY_GENERATION_ERROR"


class FrameGenerationError(PipelineError):
    def __init__(self, reason: str, scene_index: int = None):
        message = f"Gabim ne gjenerimin e frame-ve: {reason}"
        super().__init__(message, step="frame_generator", reason=reason)
        self.code = "FRAME_GENERATION_ERROR"
        if scene_index is not None:
            self.details["scene_index"] = scene_index


class AudioGenerationError(PipelineError):
    def __init__(self, reason: str):
        message = f"Gabim ne gjenerimin e audios: {reason}"
        super().__init__(message, step="audio_generator", reason=reason)
        self.code = "AUDIO_GENERATION_ERROR"


class VideoAssemblyError(PipelineError):
    def __init__(self, reason: str):
        message = f"Gabim ne bashkimin e videos: {reason}"
        super().__init__(message, step="video_assembler", reason=reason)
        self.code = "VIDEO_ASSEMBLY_ERROR"


class UpscalingError(PipelineError):
    def __init__(self, reason: str):
        message = f"Gabim ne upscaling: {reason}"
        super().__init__(message, step="upscaler", reason=reason)
        self.code = "UPSCALING_ERROR"

class StorageError(FalconAIException):
    def __init__(self, message: str, path: str = None):
        details = {"path": path} if path else {}
        super().__init__(message, code="STORAGE_ERROR", details=details)


class FileNotFoundError(StorageError):
    def __init__(self, path: str):
        super().__init__(f"File nuk u gjet: {path}", path=path)
        self.code = "FILE_NOT_FOUND"


class DiskSpaceError(StorageError):
    def __init__(self, required_gb: float, available_gb: float):
        message = (
            f"Hapesire e pamjaftueshme ne disk. "
            f"Kërkohet: {required_gb:.1f}GB, "
            f"Disponueshme: {available_gb:.1f}GB"
        )
        super().__init__(message)
        self.code = "DISK_SPACE_ERROR"
        self.details["required_gb"] = required_gb
        self.details["available_gb"] = available_gb


class S3UploadError(StorageError):
    def __init__(self, bucket: str, key: str, reason: str):
        message = f"Gabim duke ngarkuar ne S3 s3://{bucket}/{key}: {reason}"
        super().__init__(message)
        self.code = "S3_UPLOAD_ERROR"
        self.details.update({"bucket": bucket, "key": key, "reason": reason})

class GPUError(FalconAIException):
    def __init__(self, message: str):
        super().__init__(message, code="GPU_ERROR")


class CUDANotAvailableError(GPUError):
    def __init__(self):
        super().__init__(
            "CUDA nuk eshte e disponueshme. "
            "Sigurohu qe ke GPU NVIDIA dhe CUDA te instaluar, "
            "ose vendos DEVICE=cpu ne .env per te perdorur CPU."
        )
        self.code = "CUDA_NOT_AVAILABLE"


class OutOfMemoryError(GPUError):
    def __init__(self, required_gb: float = None):
        msg = "GPU memory (VRAM) e pamjaftueshme per te ekzekutuar modelin."
        if required_gb:
            msg += f" Kerkohet te pakten {required_gb}GB VRAM."
        msg += " Provo te ulesh rezolucionin ose te perdorsh CPU."
        super().__init__(msg)
        self.code = "OUT_OF_MEMORY"

def handle_exception(exc: Exception, logger=None) -> FalconAIException:
    if isinstance(exc, FalconAIException):
        return exc

    message = str(exc)
    exc_type = type(exc).__name__

    if "CUDA out of memory" in message or "VRAM" in message:
        return OutOfMemoryError()

    if "CUDA" in message and "not available" in message:
        return CUDANotAvailableError()

    if isinstance(exc, (IOError, OSError)) or "No such file" in message:
        return StorageError(f"Gabim file-system: {message}")

    wrapped = FalconAIException(
        message=f"Gabim i papritur ({exc_type}): {message}",
        code="UNEXPECTED_ERROR",
        details={"original_type": exc_type}
    )

    if logger:
        logger.exception(f"Gabim i papritur u kapur: {exc}")

    return wrapped