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
# Importojmë PipelineOrchestrator për t'iu përshtatur strukturës së re të korrigjuar
from pipeline.orchestrator import PipelineOrchestrator

logger = get_logger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="FalconAI Kids",
        description="Generate personalized children's movies with AI",
        formatter_class=argparse.RawTextHelpFormatter
    )

    required = parser.add_argument_group("Input")
    required.add_argument("--photo", type=str, required=True, help="Path te fotoja e fëmijës")
    required.add_argument("--name", type=str, required=True, help="Emri i fëmijës")
    required.add_argument("--birthday", type=str, required=True, help="Ditëlindja (YYYY-MM-DD)")

    parser.add_argument("--language", type=str, default="Albanian", help="Gjuha e filmit")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Folderi i output-it")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Debug mode")
    parser.add_argument("--no_audio", action="store_true", help="Mos gjenero audio")
    parser.add_argument("--no_upscale", action="store_true", help="Mos bëj upscale")
    parser.add_argument("--no_cleanup", action="store_true", help="Mos fshi skedarët e përkohshëm")

    return parser

def print_banner():
    banner = """
    *****************************************
    * FALCONAI KIDS                         *
    * AI Movie Generator for Children       *
    *****************************************
    """
    print(banner)

def print_summary(args: argparse.Namespace):
    try:
        bday = datetime.strptime(args.birthday, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - bday.year - ((today.month, today.day) < (bday.month, bday.day))
        age_str = f"{age} vjeç"
    except ValueError:
        age_str = "Unknown"
    
    print("─" * 54)
    print(f"  Fëmija     : {args.name}")
    print(f"  Ditëlindja : {args.birthday} ({age_str})")
    print(f"  Foto       : {args.photo}")
    print(f"  Gjuha      : {args.language}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Audio      : {'Jo' if args.no_audio else 'Po'}")
    print(f"  Upscale    : {'Jo' if args.no_upscale else 'Po'}")
    print("─" * 54)
    print()

def print_result(output_path: Path, elapsed: float):
    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
    print()
    print("─" * 54)
    print("  FILMI U GJENERUA ME SUKSES!")
    print(f"  File       : {output_path.name}")
    print(f"  Rruga      : {output_path}")
    print(f"  Madhësia   : {size_mb:.1f} MB")
    print(f"  Koha       : {elapsed:.1f} sekonda")
    print("─" * 54)
    print()

def main():
    print_banner()

    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode aktiv")

    print_summary(args)

    logger.info("Duke validuar inputet...")
    try:
        validated = validate_inputs(
            photo_path=args.photo,
            name=args.name,
            birthday=args.birthday,
        )
    except ValidationError as e:
        logger.error(f"Gabim validimi: {e}")
        print(f"\n  GABIM: {e}\n")
        sys.exit(1)

    logger.info("Inputet janë të vlefshme")

    pipeline_config = PIPELINE_CONFIG.copy()

    if args.no_audio:
        pipeline_config["steps"] = [
            s for s in pipeline_config["steps"] if s != "audio_generator"
        ]

    if args.no_upscale:
        pipeline_config["steps"] = [
            s for s in pipeline_config["steps"] if s != "upscaler"
        ]

    if args.no_cleanup:
        pipeline_config["cleanup_temp"] = False

    logger.info("Duke ngarkuar modelet AI...")
    print("Duke ngarkuar modelet AI... (mund të marrë disa minuta herën e parë)")
    print()

    # Paketojmë kontekstin fillestar për t'ia kaluar orkestratorit të ri
    context = {
        "name": validated["name"],
        "birthday": validated["birthday"],
        "age": validated.get("age"),
        "photo": validated["photo_path"],
        "language": args.language,
        "seed": args.seed,
        "output_dir": args.output_dir
    }

    try:
        # Inicializojmë orkestratorin duke ruajtur përputhshmërinë me të dyja anët
        orchestrator = PipelineOrchestrator(
            context=context,
            pipeline_config=pipeline_config,
            output_dir=Path(args.output_dir),
            language=args.language,
            seed=args.seed,
        )
    except ModelLoadError as e:
        logger.error(f"Gabim ngarkimi i modeleve: {e}")
        print(f"\n  GABIM duke ngarkuar modelet: {e}\n")
        print("  Sigurohu që:\n"
              "     - Ke internet për shkarkimin e modeleve\n"
              "     - Ke hapësirë të mjaftueshme në disk (>20GB)\n"
              "     - GPU/CUDA është i konfiguruar saktë\n")
        sys.exit(1)

    logger.info(f"Duke filluar pipeline për: {args.name}")
    print(f"  Duke gjeneruar filmin për {args.name}...")
    print()

    start_time = time.time()

    try:
        # Thërrasim funksionin pa i kaluar parametrat direkt në run(), 
        # pasi ato menaxhohen tashmë brenda kontekstit nga vetë orkestratori.
        output_path = orchestrator.run()
        
    except PipelineError as e:
        logger.error(f"Gabim në pipeline: {e}")
        print(f"\n  GABIM gjatë gjenerimit: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Pipeline u ndërpre nga përdoruesi.\n")
        sys.exit(0)
    except FalconAIException as e:
        logger.error(f"Gabim i papritur: {e}")
        print(f"\n  GABIM i papritur: {e}\n")
        sys.exit(1)

    elapsed = time.time() - start_time
    print_result(output_path, elapsed)
    logger.info(f"Pipeline përfundoi me sukses në {elapsed:.1f}s -> {output_path}")

if __name__ == "__main__":
    main()
