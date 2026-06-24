import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

from config.settings import (
    INPUT_DIR,
    OUTPUT_DIR,
    INPUT_VALIDATION,
    PIPELINE_CONFIG,
    LOG_CONFIG
)

from utils.logger import get_logger
from utils.validators import validate_inputs
from utils.exceptions import (
    FalconAIException,
    ValidationError,
    ModelLoadError,
    PipelineError
)
from pipeline.orchestrator import Orchestrator

logger = get_logger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="FalconAI Kids",
        description="Generate personalized stories for children (V1 - Story, Narration & Illustrations)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    required = parser.add_argument_group("Required Inputs")
    required.add_argument("--name", type=str, required=True, help="Child's name")
    required.add_argument("--birthday", type=str, required=True, help="Child's birthday (YYYY-MM-DD)")
    required.add_argument("--interests", type=str, required=True, help="Child's interests (e.g., 'space, dinosaurs, robots')")

    parser.add_argument("--language", type=str, default="English", help="Story language")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging mode")
    parser.add_argument("--no_audio", action="store_true", help="Disable audio generation (Narration)")
    parser.add_argument("--no_cleanup", action="store_true", help="Do not delete temporary working files")

    return parser

def print_banner():
    banner = """
    *****************************************
    * FALCONAI KIDS v1                      *
    * Personalized Story & Image Generator  *
    *****************************************
    """
    print(banner)

def print_summary(args: argparse.Namespace):
    try:
        bday = datetime.strptime(args.birthday, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - bday.year - ((today.month, today.day) < (bday.month, bday.day))
        age_str = f"{age} years old"
    except ValueError:
        age_str = "Unknown"
    
    print("─" * 54)
    print(f"  Child Name : {args.name}")
    print(f"  Birthday   : {args.birthday} ({age_str})")
    print(f"  Interests  : {args.interests}")
    print(f"  Language   : {args.language}")
    print(f"  Output Dir : {args.output_dir}")
    print(f"  Narration  : {'No' if args.no_audio else 'Yes'}")
    print("─" * 54)
    print()

def print_result(output_path: Path, elapsed: float):
    print()
    print("─" * 54)
    print("  STORY GENERATED SUCCESSFULLY!")
    print(f"  Output Folder : {output_path.name}")
    print(f"  Full Path     : {output_path}")
    print(f"  Execution Time: {elapsed:.1f} seconds")
    print("─" * 54)
    print()

def main():
    print_banner()

    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode activated")

    print_summary(args)

    logger.info("Validating inputs...")
    try:
        # Adapted for V1 by passing interests instead of an image path
        validated = validate_inputs(
            name=args.name,
            birthday=args.birthday,
            interests=args.interests
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    logger.info("Inputs are valid")

    pipeline_config = PIPELINE_CONFIG.copy()

    # Adjust pipeline steps dynamically for V1 (LLM -> Image -> Audio)
    if args.no_audio:
        pipeline_config["steps"] = [
            s for s in pipeline_config["steps"] if s != "audio_generator"
        ]

    if args.no_cleanup:
        pipeline_config["cleanup_temp"] = False

    logger.info("Loading AI models...")
    print("Loading AI models... (This may take a few minutes on the first run)")
    print()

    try:
        orchestrator = Orchestrator(
            pipeline_config=pipeline_config,
            output_dir=Path(args.output_dir),
            language=args.language,
            seed=args.seed,
        )
    except ModelLoadError as e:
        logger.error(f"Model loading error: {e}")
        print(f"\n  ERROR loading models: {e}\n")
        print("  Please verify that:\n"
              "     - You have an active internet connection for the initial download\n"
              "     - You have enough storage space on your disk\n"
              "     - Your GPU/CUDA or local environment configuration is correct\n")
        sys.exit(1)

    logger.info(f"Starting pipeline execution for: {args.name}")
    print(f"  Generating personalized story asset pack for {args.name}...")
    print()

    start_time = time.time()

    try:
        # Execute the orchestrator using V1 parameters
        output_path = orchestrator.run(
            name=validated["name"],
            birthday=validated["birthday"],
            age=validated["age"],
            interests=validated["interests"]
        )
    except PipelineError as e:
        logger.error(f"Pipeline processing error: {e}")
        print(f"\n  ERROR during generation: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.\n")
        sys.exit(0)
    except FalconAIException as e:
        logger.error(f"Unexpected application error: {e}")
        print(f"\n  UNEXPECTED ERROR: {e}\n")
        sys.exit(1)

    elapsed = time.time() - start_time
    print_result(output_path, elapsed)
    logger.info(f"Pipeline completed successfully in {elapsed:.1f}s -> {output_path}")

if __name__ == "__main__":
    main()
