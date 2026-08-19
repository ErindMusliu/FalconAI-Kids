"""
FalconAI-Kids logging utilities.

Features:
- Console and file logging
- Custom SUCCESS and STEP levels
- Optional ANSI colors
- Thread-safe logger creation
- Duplicate-handler protection
- Runtime global log-level changes
- Pipeline progress display
- Exception-safe logging
- CPU-only / dependency-free implementation
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config.settings import LOG_CONFIG


# ============================================================================
# Custom logging levels
# ============================================================================

SUCCESS_LEVEL = 25
STEP_LEVEL = 15

logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
logging.addLevelName(STEP_LEVEL, "STEP")


_logger_lock = threading.RLock()
_loggers: Dict[str, "FalconAILogger"] = {}


# ============================================================================
# ANSI colors
# ============================================================================

class Colors:
    RESET = "\033[0m"

    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BG_RED = "\033[41m"


LEVEL_STYLES: Dict[str, Tuple[str, str]] = {
    "DEBUG": (
        Colors.DIM + Colors.WHITE,
        " DBG ",
    ),
    "STEP": (
        Colors.BRIGHT_CYAN,
        " STEP ",
    ),
    "INFO": (
        Colors.BRIGHT_BLUE,
        " INFO ",
    ),
    "SUCCESS": (
        Colors.BRIGHT_GREEN,
        "  OK  ",
    ),
    "WARNING": (
        Colors.BRIGHT_YELLOW,
        " WARN ",
    ),
    "ERROR": (
        Colors.BRIGHT_RED,
        " ERR ",
    ),
    "CRITICAL": (
        Colors.BOLD + Colors.BG_RED + Colors.BRIGHT_WHITE,
        " CRIT ",
    ),
}


# ============================================================================
# Helpers
# ============================================================================

def _normalize_level(level: Any) -> int:
    """
    Convert a logging level name/value into a valid numeric level.

    Falls back to INFO for invalid values.
    """
    if isinstance(level, int):
        return level

    if isinstance(level, str):
        value = level.strip().upper()

        if value == "SUCCESS":
            return SUCCESS_LEVEL

        if value == "STEP":
            return STEP_LEVEL

        resolved = getattr(logging, value, None)

        if isinstance(resolved, int):
            return resolved

    return logging.INFO


def _shorten_logger_name(name: str, max_length: int = 25) -> str:
    """Shorten long logger names while preserving module structure."""
    if len(name) <= max_length:
        return name.ljust(max_length)

    parts = name.split(".")

    if len(parts) > 1:
        shortened = ".".join(
            part[0]
            for part in parts[:-1]
        ) + "." + parts[-1]

        if len(shortened) <= max_length:
            return shortened.ljust(max_length)

    return name[-max_length:].ljust(max_length)


def _supports_color() -> bool:
    """Return whether the current stdout environment supports ANSI colors."""
    if os.environ.get("NO_COLOR") is not None:
        return False

    if os.environ.get("TERM", "").lower() == "dumb":
        return False

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32

            stdout_handle = kernel32.GetStdHandle(-11)

            if stdout_handle in (-1, 0):
                return False

            mode = wintypes.DWORD()

            if not kernel32.GetConsoleMode(
                stdout_handle,
                ctypes.byref(mode),
            ):
                return False

            # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            mode.value |= 0x0004

            if kernel32.SetConsoleMode(
                stdout_handle,
                mode.value,
            ):
                return True

        except Exception:
            pass

        return bool(
            os.environ.get("ANSICON")
            or os.environ.get("WT_SESSION")
            or os.environ.get("ConEmuANSI") == "ON"
            or os.environ.get("TERM_PROGRAM") == "vscode"
        )

    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


# ============================================================================
# Formatters
# ============================================================================

class ColorFormatter(logging.Formatter):
    """Human-friendly console formatter with optional ANSI colors."""

    def __init__(
        self,
        use_color: bool = True,
    ) -> None:
        super().__init__()
        self.use_color = bool(use_color and _supports_color())

    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname.upper()

        color, label = LEVEL_STYLES.get(
            level_name,
            (
                Colors.WHITE,
                f" {level_name[:6]:^6} ",
            ),
        )

        timestamp = time.strftime(
            "%H:%M:%S",
            time.localtime(record.created),
        )

        logger_name = _shorten_logger_name(record.name)

        message = record.getMessage()

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        if self.use_color:
            return (
                f"{Colors.DIM}{timestamp}{Colors.RESET} "
                f"{color}{Colors.BOLD}[{label}]{Colors.RESET} "
                f"{Colors.DIM}{logger_name}{Colors.RESET}  "
                f"{color}{message}{Colors.RESET}"
            )

        return (
            f"{timestamp} "
            f"[{label.strip():<6}] "
            f"{logger_name}  "
            f"{message}"
        )


class FileFormatter(logging.Formatter):
    """Stable plain-text formatter for persistent log files."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(record.created),
        )

        level_name = record.levelname.ljust(8)
        logger_name = record.name[-30:].ljust(30)

        message = record.getMessage()

        result = (
            f"{timestamp} | "
            f"{level_name} | "
            f"{logger_name} | "
            f"{message}"
        )

        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

        return result


# ============================================================================
# Pipeline formatter
# ============================================================================

class PipelineFormatter:
    """
    Console progress formatter for the FalconAI-Kids pipeline.

    This class only handles presentation. It does not perform any
    processing and therefore does not require GPU/ML dependencies.
    """

    STEPS = (
        "face_processor",
        "story_generator",
        "frame_generator",
        "audio_generator",
        "video_assembler",
        "upscaler",
    )

    STEP_LABELS = {
        "face_processor": "Face Processing",
        "story_generator": "Story Generation",
        "frame_generator": "Frame Generation",
        "audio_generator": "Audio Generation",
        "video_assembler": "Video Assembly",
        "upscaler": "Upscaling",
    }

    STEP_ICONS = {
        "face_processor": "👤",
        "story_generator": "📖",
        "frame_generator": "🎨",
        "audio_generator": "🎵",
        "video_assembler": "🎬",
        "upscaler": "✨",
    }

    def __init__(
        self,
        total_steps: int = 6,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be greater than zero.")

        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.monotonic()
        self.step_start_time = self.start_time
        self._progress_active = False

    def start_step(self, step_name: str) -> None:
        """Start a new pipeline step."""
        self.current_step = min(
            self.current_step + 1,
            self.total_steps,
        )

        self.step_start_time = time.monotonic()
        self._progress_active = False

        label = self.STEP_LABELS.get(
            step_name,
            step_name.replace("_", " ").title(),
        )

        icon = self.STEP_ICONS.get(
            step_name,
            "▶",
        )

        bar = self._make_bar(
            self.current_step - 1,
        )

        self._print(
            f"\n{Colors.BRIGHT_CYAN}{bar}{Colors.RESET}"
        )

        self._print(
            f"  {icon}  "
            f"{Colors.BOLD}{Colors.BRIGHT_WHITE}"
            f"Step {self.current_step}/{self.total_steps}: "
            f"{label}"
            f"{Colors.RESET}"
        )

    def end_step(
        self,
        step_name: str,
        success: bool = True,
    ) -> None:
        """Finish the current pipeline step."""
        elapsed = time.monotonic() - self.step_start_time

        label = self.STEP_LABELS.get(
            step_name,
            step_name.replace("_", " ").title(),
        )

        if success:
            self._print(
                f"  {Colors.BRIGHT_GREEN}✓{Colors.RESET}  "
                f"{label} completed "
                f"{Colors.DIM}({elapsed:.1f}s){Colors.RESET}"
            )
        else:
            self._print(
                f"  {Colors.BRIGHT_RED}✗{Colors.RESET}  "
                f"{label} failed "
                f"{Colors.DIM}({elapsed:.1f}s){Colors.RESET}"
            )

        self._progress_active = False

    def update_progress(
        self,
        current: int,
        total: int,
        label: str = "",
    ) -> None:
        """Render progress for a sub-task."""
        if total <= 0:
            total = 1

        current = max(0, min(current, total))

        percentage = current / total

        width = 30
        filled = int(width * percentage)

        bar = (
            "█" * filled
            + "░" * (width - filled)
        )

        text = (
            f"\r  {Colors.CYAN}[{bar}]{Colors.RESET} "
            f"{Colors.BOLD}{current}/{total}{Colors.RESET}"
        )

        if label:
            text += (
                f" {Colors.DIM}{label}{Colors.RESET}"
            )

        print(
            text,
            end="",
            flush=True,
        )

        self._progress_active = current < total

        if current >= total:
            print()

    def finish(self) -> None:
        """Mark the complete pipeline as finished."""
        if self._progress_active:
            print()

        elapsed = time.monotonic() - self.start_time

        bar = self._make_bar(self.total_steps)

        self._print(
            f"\n{Colors.BRIGHT_GREEN}{bar}{Colors.RESET}"
        )

        self._print(
            f"\n  "
            f"{Colors.BOLD}{Colors.BRIGHT_GREEN}"
            f"Pipeline completed successfully "
            f"in {elapsed:.1f} seconds."
            f"{Colors.RESET}\n"
        )

    def fail(self, reason: str = "") -> None:
        """Display a pipeline failure message."""
        if self._progress_active:
            print()

        elapsed = time.monotonic() - self.start_time

        message = (
            f"\n  {Colors.BRIGHT_RED}"
            f"✗ Pipeline failed "
            f"({elapsed:.1f}s)"
            f"{Colors.RESET}"
        )

        if reason:
            message += f": {reason}"

        self._print(message)

    def reset(self) -> None:
        """Reset pipeline timing and progress."""
        self.current_step = 0
        self.start_time = time.monotonic()
        self.step_start_time = self.start_time
        self._progress_active = False

    def _make_bar(self, completed: int) -> str:
        width = 50

        completed = max(
            0,
            min(completed, self.total_steps),
        )

        percentage = completed / self.total_steps

        filled = int(width * percentage)

        return (
            f"  ["
            f"{'█' * filled}"
            f"{'░' * (width - filled)}"
            f"] {completed}/{self.total_steps}"
        )

    @staticmethod
    def _print(message: str) -> None:
        print(message, flush=True)


# ============================================================================
# Logger
# ============================================================================

class FalconAILogger(logging.Logger):
    """Custom logger with SUCCESS and STEP helpers."""

    def success(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log a successful operation."""
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(
                SUCCESS_LEVEL,
                message,
                args,
                **kwargs,
            )

    def step(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log a pipeline step."""
        if self.isEnabledFor(STEP_LEVEL):
            self._log(
                STEP_LEVEL,
                message,
                args,
                **kwargs,
            )


# Make custom logger class available to logging.getLogger().
logging.setLoggerClass(FalconAILogger)


# ============================================================================
# Logger creation
# ============================================================================

def get_logger(
    name: str,
    level: Optional[str] = None,
) -> FalconAILogger:
    """
    Return a configured FalconAI logger.

    Logger instances are cached and handlers are created only once.
    """
    if not name:
        name = "falconai"

    with _logger_lock:
        cached = _loggers.get(name)

        if cached is not None:
            if level is not None:
                _apply_logger_level(
                    cached,
                    _normalize_level(level),
                )

            return cached

        logger = logging.getLogger(name)

        if not isinstance(logger, FalconAILogger):
            # This normally only happens when another part of the
            # application created the logger before setLoggerClass().
            logger.__class__ = FalconAILogger

        log_level = _normalize_level(
            level
            or LOG_CONFIG.get("level", "INFO")
        )

        logger.setLevel(log_level)
        logger.propagate = False

        # Remove stale handlers that may have been attached by a
        # previous initialization path.
        _remove_duplicate_handlers(logger)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        use_color = _config_bool(
            "color",
            default=True,
        )

        console_handler.setFormatter(
            ColorFormatter(
                use_color=use_color,
            )
        )

        logger.addHandler(console_handler)

        if _config_bool(
            "log_to_file",
            default=True,
        ):
            log_file = Path(
                LOG_CONFIG.get(
                    "log_file",
                    "logs/falconai.log",
                )
            )

            try:
                log_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                file_handler = logging.FileHandler(
                    log_file,
                    encoding="utf-8",
                )

                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(
                    FileFormatter()
                )

                logger.addHandler(file_handler)

            except (OSError, PermissionError) as exc:
                # Logging must not crash the application simply because
                # the persistent log file cannot be created.
                logger.warning(
                    "Unable to initialize file logging: %s",
                    exc,
                )

        _loggers[name] = logger

        return logger


def _remove_duplicate_handlers(
    logger: logging.Logger,
) -> None:
    """Remove duplicate handlers while preserving useful handlers."""
    seen: set[tuple[type, str]] = set()

    for handler in list(logger.handlers):
        stream_name = ""

        if isinstance(handler, logging.StreamHandler):
            stream_name = str(
                getattr(handler, "stream", "")
            )

        key = (
            type(handler),
            stream_name,
        )

        if key in seen:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        else:
            seen.add(key)


def _config_bool(
    key: str,
    default: bool = False,
) -> bool:
    """Read a boolean configuration value safely."""
    value = LOG_CONFIG.get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "enabled",
        }

    return bool(value)


def _apply_logger_level(
    logger: FalconAILogger,
    level: int,
) -> None:
    """Apply a logging level to logger and its handlers."""
    logger.setLevel(level)

    for handler in logger.handlers:
        # File handler keeps DEBUG logs so detailed diagnostics
        # are still available in persistent logs.
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(level)


# ============================================================================
# Global configuration
# ============================================================================

def set_global_level(level: str) -> None:
    """
    Change the logging level of all existing FalconAI loggers.
    """
    numeric_level = _normalize_level(level)

    with _logger_lock:
        for logger in _loggers.values():
            _apply_logger_level(
                logger,
                numeric_level,
            )


def get_pipeline_formatter(
    total_steps: int = 6,
) -> PipelineFormatter:
    """Create a pipeline progress formatter."""
    return PipelineFormatter(
        total_steps=total_steps,
    )


# ============================================================================
# Shutdown / cleanup
# ============================================================================

def close_loggers() -> None:
    """
    Close all FalconAI logger handlers.

    Useful for tests, worker shutdown, or application restart.
    """
    with _logger_lock:
        for logger in _loggers.values():
            for handler in list(logger.handlers):
                try:
                    handler.flush()
                    handler.close()
                except Exception:
                    pass

                logger.removeHandler(handler)

        _loggers.clear()
