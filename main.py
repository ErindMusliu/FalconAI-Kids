import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

from config.settings import (
    OUTPUT_DIR,
    PIPELINE_CONFIG,
)

from utils.logger import get_logger
from utils.validators import validate_name, validate_birthday, validate_photo
from utils.exceptions import (
    FalconAIException,
    ValidationError,
    InvalidPhotoError,
    ModelLoadError,
    PipelineError,
)
from pipeline.orchestrator import PipelineOrchestrator

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="FalconAI Kids",
        description="Generate a personalized animated storybook video for children",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = parser.add_argument_group("Required Inputs")
    required.add_argument("--name", type=str, required=True, help="Child's name")
    required.add_argument("--birthday", type=str, required=True, help="Child's birthday (YYYY-MM-DD)")

    optional = parser.add_argument_group("Optional Inputs")
    optional.add_argument("--photo", type=str, default=None,
                          help="Path to a portrait photo of the child (used so characters resemble them). "
                               "If omitted, characters are generated generically.")
    optional.add_argument("--gender", type=str, default=None, help="Child's gender (used for story phrasing)")
    optional.add_argument("--theme", type=str, default="magical adventure", help="Story theme")
    optional.add_argument("--favorite_animal", type=str, default="friendly creature", help="A favorite animal/creature to feature in the story")
    optional.add_argument("--trait", type=str, default="brave", help="A character trait to highlight in the story")
    optional.add_argument("--language", type=str, default="Albanian", help="Story & narration language")
    optional.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory path")
    optional.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    optional.add_argument("--verbose", action="store_true", help="Enable debug logging mode")
    optional.add_argument("--no_audio", action="store_true", help="Disable audio generation (narration + music)")
    optional.add_argument("--no_cleanup", action="store_true", help="Do not delete temporary working files")

    return parser


def print_banner():
    banner = """
    *****************************************
    * FALCONAI KIDS                         *
    * Personalized Animated Storybook Video *
    *****************************************
    """
    print(banner)


def print_summary(args):
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
    print(f"  Photo      : {args.photo or '(none — generic characters)'}")
    print(f"  Theme      : {args.theme}")
    print(f"  Animal     : {args.favorite_animal}")
    print(f"  Trait      : {args.trait}")
    print(f"  Language   : {args.language}")
    print(f"  Output Dir : {args.output_dir}")
    print(f"  Narration  : {'No' if args.no_audio else 'Yes'}")
    print("─" * 54)
    print()


def print_result(output_path: Path, elapsed: float):
    print()
    print("─" * 54)
    print("  STORYBOOK VIDEO GENERATED SUCCESSFULLY!")
    print(f"  Output File   : {output_path.name}")
    print(f"  Full Path     : {output_path}")
    print(f"  Execution Time: {elapsed:.1f} seconds")
    print("─" * 54)
    print()


def make_progress_callback():
    def _callback(current: int, total: int, label: str = ""):
        width = 30
        pct = current / total if total else 0
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        print(f"\r  [{bar}] {current}/{total}  {label}", end="", flush=True)
        if current >= total:
            print()
    return _callback


def validate_all_inputs(args) -> dict:
    errors = []
    result = {}

    try:
        result["name"] = validate_name(args.name)
    except ValidationError as e:
        errors.append(str(e))

    try:
        bday, age = validate_birthday(args.birthday)
        result["birthday"] = bday
        result["age"] = age
    except ValidationError as e:
        errors.append(str(e))

    if args.photo:
        try:
            result["photo_path"] = validate_photo(args.photo)
        except InvalidPhotoError as e:
            errors.append(str(e))
    else:
        result["photo_path"] = None

    if errors:
        if len(errors) == 1:
            raise ValidationError(errors[0])
        combined = "Multiple data payload schema exceptions flagged:\n" + "\n".join(
            f"  {i + 1}. {err}" for i, err in enumerate(errors)
        )
        raise ValidationError(combined)

    return result


def main():
    print_banner()

    parser = build_parser()
    
    # Trajtim inteligjent: Nëse jemi në Streamlit dhe mungojnë argumentet, përdorim vlera default
    if len(sys.argv) == 1 or any("streamlit" in arg for arg in sys.argv):
        class Args:
            name = "Kopr"
            birthday = "2020-01-01"
            photo = None
            gender = None
            theme = "magical adventure"
            favorite_animal = "friendly creature"
            trait = "brave"
            language = "Albanian"
            output_dir = str(OUTPUT_DIR)
            seed = None
            verbose = False
            no_audio = False
            no_cleanup = False
        args = Args()
    else:
        args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode activated")

    print_summary(args)

    logger.info("Validating inputs...")
    try:
        validated = validate_all_inputs(args)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    logger.info("Inputs are valid")

    context = {
        "name": validated["name"],
        "birthday": args.birthday,
        "photo": str(validated["photo_path"]) if validated["photo_path"] else None,
        "gender": args.gender,
        "language": args.language,
        "preferences": {
            "theme": args.theme,
            "favorite_animal": args.favorite_animal,
            "trait": args.trait,
        },
        "seed": args.seed,
        "output_dir": args.output_dir,
    }

    steps = list(PIPELINE_CONFIG["steps"])
    if args.no_audio:
        context["audio_paths"] = None
        steps = [s for s in steps if s != "audio_generator"]
    context["steps"] = steps

    if args.no_cleanup:
        cleanup_override = False
    else:
        cleanup_override = PIPELINE_CONFIG.get("cleanup_temp", True)

    logger.info("Loading AI models...")
    print("Loading AI models... (this may take a few minutes on the first run)")
    print()

    try:
        orchestrator = PipelineOrchestrator(context)
        orchestrator.cleanup_temp = cleanup_override
    except (ModelLoadError, ValueError) as e:
        logger.error(f"Model loading error: {e}")
        print(f"\n  ERROR loading models: {e}\n")
        print(
            "  Please verify that:\n"
            "     - You have an active internet connection for the initial download\n"
            "     - You have enough storage space on your disk\n"
            "     - Your GPU/CUDA or local environment configuration is correct\n"
        )
        sys.exit(1)

    logger.info(f"Starting pipeline execution for: {args.name}")
    print(f"  Generating personalized storybook video for {args.name}...")
    print()

    progress_callback = make_progress_callback()
    start_time = time.time()

    try:
        output_path = orchestrator.run(progress_callback=progress_callback)
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
