import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from config.settings import LOG_CONFIG

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

STEP_LEVEL = 15
logging.addLevelName(STEP_LEVEL, "STEP")

class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"

LEVEL_STYLES = {
    "DEBUG"   : (Colors.DIM + Colors.WHITE,        "  DBG "),
    "STEP"    : (Colors.BRIGHT_CYAN,               " STEP "),
    "INFO"    : (Colors.BRIGHT_BLUE,               " INFO "),
    "SUCCESS" : (Colors.BRIGHT_GREEN,              "  OK  "),
    "WARNING" : (Colors.BRIGHT_YELLOW,             " WARN "),
    "ERROR"   : (Colors.BRIGHT_RED,                "  ERR "),
    "CRITICAL": (Colors.BOLD + Colors.BG_RED,      " CRIT "),
}

class ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname
        color, label = LEVEL_STYLES.get(
            level_name,
            (Colors.WHITE, level_name[:6].center(6))
        )

        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        name = record.name
        if len(name) > 25:
            parts = name.split(".")
            name = ".".join(p[0] for p in parts[:-1]) + "." + parts[-1]
        name = name[-25:].ljust(25)

        message = record.getMessage()

        exc_text = ""
        if record.exc_info:
            exc_text = "\n" + self.formatException(record.exc_info)

        if self.use_color:
            return (
                f"{Colors.DIM}{ts}{Colors.RESET} "
                f"{color}{Colors.BOLD}[{label}]{Colors.RESET} "
                f"{Colors.DIM}{name}{Colors.RESET}  "
                f"{color}{message}{Colors.RESET}"
                f"{exc_text}"
            )
        else:
            return (
                f"{ts} [{label}] {name}  {message}{exc_text}"
            )

class FileFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname.ljust(8)
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        name = record.name[-30:].ljust(30)
        message = record.getMessage()

        base = f"{ts} | {level_name} | {name} | {message}"

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base

class PipelineFormatter:
    STEPS = [
        "face_processor",
        "story_generator",
        "frame_generator",
        "audio_generator",
        "video_assembler",
        "upscaler",
    ]

    STEP_LABELS = {
        "face_processor"  : "Procesim fytyre",
        "story_generator" : "Gjenerim historie",
        "frame_generator" : "Gjenerim frames",
        "audio_generator" : "Gjenerim audio",
        "video_assembler" : "Bashkim video",
        "upscaler"        : "Upscaling",
    }

    STEP_ICONS = {
        "face_processor"  : "👤",
        "story_generator" : "📖",
        "frame_generator" : "🎨",
        "audio_generator" : "🎵",
        "video_assembler" : "🎬",
        "upscaler"        : "✨",
    }

    def __init__(self, total_steps: int = 6):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_start_time = time.time()

    def start_step(self, step_name: str) -> None:
        self.current_step += 1
        self.step_start_time = time.time()

        label = self.STEP_LABELS.get(step_name, step_name)
        icon  = self.STEP_ICONS.get(step_name, "▶")

        bar = self._make_bar(self.current_step - 1)

        print(f"\n{Colors.BRIGHT_CYAN}{bar}{Colors.RESET}")
        print(
            f"  {icon}  {Colors.BOLD}{Colors.BRIGHT_WHITE}"
            f"Hapi {self.current_step}/{self.total_steps}: {label}"
            f"{Colors.RESET}"
        )

    def end_step(self, step_name: str, success: bool = True) -> None:
        elapsed = time.time() - self.step_start_time
        label   = self.STEP_LABELS.get(step_name, step_name)

        if success:
            print(
                f"  {Colors.BRIGHT_GREEN}✓{Colors.RESET}  "
                f"{label} perfundoi "
                f"{Colors.DIM}({elapsed:.1f}s){Colors.RESET}"
            )
        else:
            print(
                f"  {Colors.BRIGHT_RED}✗{Colors.RESET}  "
                f"{label} deshtoi "
                f"{Colors.DIM}({elapsed:.1f}s){Colors.RESET}"
            )

    def update_progress(self, current: int, total: int, label: str = "") -> None:
        pct   = current / total if total > 0 else 0
        width = 30
        filled = int(width * pct)
        bar   = "█" * filled + "░" * (width - filled)

        print(
            f"\r  {Colors.CYAN}[{bar}]{Colors.RESET} "
            f"{Colors.BOLD}{current}/{total}{Colors.RESET} "
            f"{Colors.DIM}{label}{Colors.RESET}",
            end="", flush=True
        )
        if current >= total:
            print()

    def finish(self) -> None:
        total_elapsed = time.time() - self.start_time
        bar = self._make_bar(self.total_steps)
        print(f"\n{Colors.BRIGHT_GREEN}{bar}{Colors.RESET}")
        print(
            f"\n  {Colors.BOLD}{Colors.BRIGHT_GREEN}"
            f"Pipeline perfundoi ne {total_elapsed:.1f} sekonda!"
            f"{Colors.RESET}\n"
        )

    def _make_bar(self, completed: int) -> str:
        width = 50
        pct   = completed / self.total_steps if self.total_steps > 0 else 0
        filled = int(width * pct)
        return f"  [{'█' * filled}{'░' * (width - filled)}] {completed}/{self.total_steps}"

class FalconAILogger(logging.Logger):
    def success(self, message: str, *args, **kwargs) -> None:
        """Log mesazh suksesi (nivel 25, mbi INFO)."""
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, message, args, **kwargs)

    def step(self, message: str, *args, **kwargs) -> None:
        """Log hap të pipeline-it (nivel 15, mbi DEBUG)."""
        if self.isEnabledFor(STEP_LEVEL):
            self._log(STEP_LEVEL, message, args, **kwargs)

logging.setLoggerClass(FalconAILogger)

_loggers: dict[str, FalconAILogger] = {}

def get_logger(name: str, level: Optional[str] = None) -> FalconAILogger:
    if name in _loggers:
        return _loggers[name]

    logger: FalconAILogger = logging.getLogger(name)

    log_level_str = level or LOG_CONFIG.get("level", "INFO")
    log_level     = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    if logger.handlers:
        _loggers[name] = logger
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColorFormatter(use_color=True))
    logger.addHandler(console_handler)

    if LOG_CONFIG.get("log_to_file", True):
        log_file = Path(LOG_CONFIG.get("log_file", "logs/falconai.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)

    logger.propagate = False

    _loggers[name] = logger
    return logger

def _supports_color() -> bool:
    if sys.platform == "win32":
        return (
            "ANSICON" in __import__("os").environ
            or "WT_SESSION" in __import__("os").environ
            or __import__("os").environ.get("TERM_PROGRAM") == "vscode"
        )
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def set_global_level(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    for logger in _loggers.values():
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)


def get_pipeline_formatter(total_steps: int = 6) -> PipelineFormatter:
    return PipelineFormatter(total_steps=total_steps)