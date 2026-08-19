"""
FalconAI-Kids application entry point.

This module:
- Parses CLI arguments
- Validates user inputs
- Builds the pipeline execution context
- Configures pipeline options
- Starts PipelineOrchestrator
- Reports progress and results
- Handles expected application failures cleanly

The entry point is intentionally lightweight. Pipeline logic belongs in
pipeline/orchestrator.py and validation logic belongs in utils/validators.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import OUTPUT_DIR, PIPELINE_CONFIG
from pipeline.orchestrator import PipelineOrchestrator
from utils.exceptions import (
    FalconAIException,
    ModelLoadError,
    PipelineError,
    ValidationError,
)
from utils.logger import get_logger
from utils.validators import validate_inputs


logger = get_logger(__name__)


# ============================================================================
# CLI configuration
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the FalconAI-Kids command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="FalconAI Kids",
        description=(
            "Generate a personalized animated storybook video "
            "for children."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Required inputs
    # ------------------------------------------------------------------
    required = parser.add_argument_group("Required Inputs")

    required.add_argument(
        "--name",
        type=str,
        required=True,
        help="Child's name.",
    )

    required.add_argument(
        "--birthday",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="Child's birthday in YYYY-MM-DD format.",
    )

    # ------------------------------------------------------------------
    # Character / story inputs
    # ------------------------------------------------------------------
    story = parser.add_argument_group("Story Configuration")

    story.add_argument(
        "--photo",
        type=str,
        default=None,
        help=(
            "Optional portrait photo. If omitted, the pipeline "
            "uses generic characters."
        ),
    )

    story.add_argument(
        "--gender",
        type=str,
        default=None,
        help="Optional gender information used for story phrasing.",
    )

    story.add_argument(
        "--theme",
        type=str,
        default="magical adventure",
        help="Primary story theme.",
    )

    story.add_argument(
        "--favorite_animal",
        type=str,
        default="friendly creature",
        help="Animal or creature featured in the story.",
    )

    story.add_argument(
        "--trait",
        type=str,
        default="brave",
        help="Character trait highlighted by the story.",
    )

    story.add_argument(
        "--language",
        type=str,
        default="Albanian",
        help="Story and narration language.",
    )

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------
    runtime = parser.add_argument_group("Runtime Configuration")

    runtime.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory where generated output files are stored.",
    )

    runtime.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )

    runtime.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    runtime.add_argument(
        "--no_audio",
        action="store_true",
        help="Disable narration and music generation.",
    )

    runtime.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Keep temporary pipeline files after generation.",
    )

    return parser


# ============================================================================
# Console UI
# ============================================================================

def print_banner() -> None:
    """Print the FalconAI-Kids CLI banner."""

    print(
        """
╔══════════════════════════════════════════════════════════╗
║                    FALCONAI KIDS                        ║
║          Personalized Animated Storybook                ║
╚══════════════════════════════════════════════════════════╝
"""
    )


def print_summary(
    args: argparse.Namespace,
    validated: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a human-readable summary of the requested generation."""

    age = None

    if validated:
        age = validated.get("age")

    age_text = (
        f"{age} years old"
        if age is not None
        else "unknown"
    )

    photo_text = (
        str(validated["photo_path"])
        if validated and validated.get("photo_path")
        else "(none — generic characters)"
    )

    print("─" * 60)
    print(f"  Child Name : {validated.get('name', args.name) if validated else args.name}")
    print(f"  Birthday   : {args.birthday} ({age_text})")
    print(f"  Photo      : {photo_text}")
    print(f"  Theme      : {args.theme}")
    print(f"  Animal     : {args.favorite_animal}")
    print(f"  Trait      : {args.trait}")
    print(f"  Gender     : {args.gender or '(not specified)'}")
    print(f"  Language   : {args.language}")
    print(f"  Output Dir : {args.output_dir}")
    print(f"  Audio      : {'Disabled' if args.no_audio else 'Enabled'}")
    print(
        f"  Cleanup    : "
        f"{'Disabled' if args.no_cleanup else 'Enabled'}"
    )
    print("─" * 60)
    print()


def print_result(
    output_path: Path,
    elapsed: float,
) -> None:
    """Print successful pipeline completion information."""

    print()
    print("═" * 60)
    print("  ✓ STORYBOOK VIDEO GENERATED SUCCESSFULLY")
    print("═" * 60)
    print(f"  Output File    : {output_path.name}")
    print(f"  Full Path      : {output_path}")
    print(f"  Execution Time : {elapsed:.1f} seconds")
    print("═" * 60)
    print()


def print_error(
    title: str,
    message: str,
) -> None:
    """Print a consistent CLI error block."""

    print()
    print("─" * 60)
    print(f"  ✗ {title}")
    print("─" * 60)
    print(f"  {message}")
    print("─" * 60)
    print()


# ============================================================================
# Progress reporting
# ============================================================================

def make_progress_callback():
    """
    Create the callback consumed by PipelineOrchestrator.

    The callback intentionally does not contain pipeline logic.
    """

    last_length = 0

    def _callback(
        current: int,
        total: int,
        label: str = "",
    ) -> None:
        nonlocal last_length

        if total <= 0:
            return

        current = max(0, min(current, total))
        percentage = current / total

        width = 30
        filled = int(width * percentage)

        bar = (
            "█" * filled
            + "░" * (width - filled)
        )

        message = (
            f"  [{bar}] "
            f"{current}/{total} "
            f"{label}"
        )

        # Prevent stale characters from previous, longer messages.
        padding = max(
            0,
            last_length - len(message),
        )

        print(
            "\r"
            + message
            + (" " * padding),
            end="",
            flush=True,
        )

        last_length = len(message)

        if current >= total:
            print()

    return _callback


# ============================================================================
# Input / context preparation
# ============================================================================

def validate_cli_inputs(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Validate all user-controlled core inputs.

    The actual validation implementation lives in utils.validators.
    """

    return validate_inputs(
        name=args.name,
        birthday=args.birthday,
        photo_path=args.photo,
    )


def validate_output_directory(
    output_dir: str,
) -> Path:
    """
    Validate and prepare the output directory.

    The directory may not exist yet; it will be created.
    """

    if not output_dir or not output_dir.strip():
        raise ValidationError(
            "Output directory cannot be empty.",
            field="output_dir",
            value=output_dir,
        )

    try:
        path = Path(output_dir).expanduser()
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Invalid output directory: {exc}",
            field="output_dir",
            value=output_dir,
        ) from exc

    try:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise ValidationError(
            f"Unable to create output directory: {exc}",
            field="output_dir",
            value=str(path),
        ) from exc

    if not path.is_dir():
        raise ValidationError(
            "Output path does not point to a directory.",
            field="output_dir",
            value=str(path),
        )

    return path.resolve()


def build_pipeline_context(
    args: argparse.Namespace,
    validated: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Build the context consumed by PipelineOrchestrator.
    """

    configured_steps = PIPELINE_CONFIG.get(
        "steps",
        [],
    )

    if not isinstance(configured_steps, (list, tuple)):
        raise PipelineError(
            "Pipeline configuration contains an invalid steps definition.",
            step="orchestrator",
        )

    steps = list(configured_steps)

    # Audio can be disabled from the CLI.
    if args.no_audio:
        steps = [
            step
            for step in steps
            if step != "audio_generator"
        ]

    context: Dict[str, Any] = {
        "name": validated["name"],
        "birthday": args.birthday,
        "age": validated["age"],
        "photo": (
            str(validated["photo_path"])
            if validated.get("photo_path")
            else None
        ),
        "gender": args.gender,
        "language": args.language,
        "preferences": {
            "theme": args.theme,
            "favorite_animal": args.favorite_animal,
            "trait": args.trait,
        },
        "seed": args.seed,
        "output_dir": str(output_dir),
        "steps": steps,
        "audio_enabled": not args.no_audio,
        "cleanup_temp": (
            False
            if args.no_cleanup
            else PIPELINE_CONFIG.get(
                "cleanup_temp",
                True,
            )
        ),
    }

    # Explicitly tell downstream stages that no audio is expected.
    if args.no_audio:
        context["audio_paths"] = None

    return context


# ============================================================================
# Logging
# ============================================================================

def configure_verbose_logging() -> None:
    """Enable DEBUG logging for FalconAI-Kids loggers."""

    logging.getLogger().setLevel(
        logging.DEBUG
    )

    logger.setLevel(
        logging.DEBUG
    )

    logger.debug(
        "Verbose logging enabled."
    )


# ============================================================================
# Pipeline execution
# ============================================================================

def create_orchestrator(
    context: Dict[str, Any],
) -> PipelineOrchestrator:
    """
    Instantiate the pipeline orchestrator.

    Model-loading failures are handled by the caller so that CLI output
    remains clean and user-friendly.
    """

    logger.info(
        "Initializing FalconAI-Kids pipeline..."
    )

    return PipelineOrchestrator(
        context
    )


def run_pipeline(
    orchestrator: PipelineOrchestrator,
) -> Path:
    """Execute the pipeline and return the generated output path."""

    progress_callback = make_progress_callback()

    return orchestrator.run(
        progress_callback=progress_callback
    )


# ============================================================================
# Main application
# ============================================================================

def main(
    argv: Optional[list[str]] = None,
) -> int:
    """
    Application entry point.

    Returns:
        0 on success
        1 on expected application failure
        130 when interrupted by the user
    """

    print_banner()

    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse uses SystemExit for --help and invalid arguments.
        return int(exc.code)

    # ------------------------------------------------------------------
    # Verbose mode
    # ------------------------------------------------------------------
    if args.verbose:
        configure_verbose_logging()

    # ------------------------------------------------------------------
    # Validate core inputs
    # ------------------------------------------------------------------
    logger.info(
        "Validating input parameters..."
    )

    try:
        validated = validate_cli_inputs(
            args
        )

        output_dir = validate_output_directory(
            args.output_dir
        )

    except ValidationError as exc:
        logger.error(
            "Input validation failed: %s",
            exc,
        )

        print_error(
            "INPUT VALIDATION ERROR",
            str(exc),
        )

        return 1

    print_summary(
        args,
        validated,
    )

    logger.success(
        "All input parameters validated successfully."
    )

    # ------------------------------------------------------------------
    # Build pipeline context
    # ------------------------------------------------------------------
    try:
        context = build_pipeline_context(
            args=args,
            validated=validated,
            output_dir=output_dir,
        )

    except PipelineError as exc:
        logger.error(
            "Pipeline configuration error: %s",
            exc,
        )

        print_error(
            "PIPELINE CONFIGURATION ERROR",
            str(exc),
        )

        return 1

    steps = context["steps"]

    logger.info(
        "Pipeline configured with %d step(s): %s",
        len(steps),
        ", ".join(steps) if steps else "(none)",
    )

    # ------------------------------------------------------------------
    # Initialize orchestrator
    # ------------------------------------------------------------------
    logger.info(
        "Initializing pipeline runtime..."
    )

    print(
        "  Initializing generation pipeline..."
    )
    print()

    try:
        orchestrator = create_orchestrator(
            context
        )

        # Keep this explicit for compatibility with existing
        # PipelineOrchestrator implementations.
        if hasattr(
            orchestrator,
            "cleanup_temp",
        ):
            orchestrator.cleanup_temp = context[
                "cleanup_temp"
            ]

    except ModelLoadError as exc:
        logger.error(
            "Model initialization failed: %s",
            exc,
        )

        print_error(
            "MODEL INITIALIZATION ERROR",
            str(exc),
        )

        print(
            "  Check that required local model files are available "
            "and that the configured runtime environment is valid."
        )
        print()

        return 1

    except ValueError as exc:
        logger.error(
            "Invalid pipeline configuration: %s",
            exc,
        )

        print_error(
            "RUNTIME CONFIGURATION ERROR",
            str(exc),
        )

        return 1

    except FalconAIException as exc:
        logger.error(
            "Application initialization failed: %s",
            exc,
        )

        print_error(
            "APPLICATION ERROR",
            str(exc),
        )

        return 1

    except Exception as exc:
        logger.exception(
            "Unexpected orchestrator initialization failure."
        )

        print_error(
            "UNEXPECTED INITIALIZATION ERROR",
            str(exc),
        )

        return 1

    # ------------------------------------------------------------------
    # Execute pipeline
    # ------------------------------------------------------------------
    logger.info(
        "Starting story generation for '%s'.",
        validated["name"],
    )

    print(
        f"  Generating personalized storybook "
        f"for {validated['name']}..."
    )
    print()

    start_time = time.perf_counter()

    try:
        output_path = run_pipeline(
            orchestrator
        )

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start_time

        logger.warning(
            "Pipeline interrupted by user after %.1f seconds.",
            elapsed,
        )

        print(
            "\n\n  Pipeline interrupted by user."
        )

        return 130

    except PipelineError as exc:
        elapsed = time.perf_counter() - start_time

        logger.error(
            "Pipeline execution failed after %.1f seconds: %s",
            elapsed,
            exc,
        )

        print_error(
            "PIPELINE ERROR",
            str(exc),
        )

        return 1

    except FalconAIException as exc:
        elapsed = time.perf_counter() - start_time

        logger.error(
            "FalconAI application error after %.1f seconds: %s",
            elapsed,
            exc,
        )

        print_error(
            "APPLICATION ERROR",
            str(exc),
        )

        return 1

    except Exception as exc:
        elapsed = time.perf_counter() - start_time

        logger.exception(
            "Unhandled pipeline exception after %.1f seconds.",
            elapsed,
        )

        print_error(
            "UNEXPECTED ERROR",
            str(exc),
        )

        return 1

    # ------------------------------------------------------------------
    # Validate pipeline result
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start_time

    try:
        output_path = Path(output_path)

    except (TypeError, ValueError) as exc:
        logger.error(
            "Pipeline returned an invalid output path: %s",
            exc,
        )

        print_error(
            "INVALID PIPELINE OUTPUT",
            str(exc),
        )

        return 1

    if not output_path.exists():
        logger.error(
            "Pipeline completed but output file was not found: %s",
            output_path,
        )

        print_error(
            "OUTPUT FILE NOT FOUND",
            (
                "The pipeline reported successful completion, "
                "but the expected output file does not exist."
            ),
        )

        return 1

    if not output_path.is_file():
        logger.error(
            "Pipeline output path is not a regular file: %s",
            output_path,
        )

        print_error(
            "INVALID OUTPUT FILE",
            str(output_path),
        )

        return 1

    if output_path.stat().st_size <= 0:
        logger.error(
            "Pipeline produced an empty output file: %s",
            output_path,
        )

        print_error(
            "EMPTY OUTPUT FILE",
            (
                "The pipeline completed, but the generated output "
                "file contains no data."
            ),
        )

        return 1

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------
    print_result(
        output_path=output_path,
        elapsed=elapsed,
    )

    logger.success(
        "Pipeline completed successfully in %.1f seconds: %s",
        elapsed,
        output_path,
    )

    return 0


# ============================================================================
# Python entry point
# ============================================================================

if __name__ == "__main__":
    sys.exit(
        main()
    )
