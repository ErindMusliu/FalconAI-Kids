import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    FACE_CONFIG,
    LLM_CONFIG,
    DIFFUSION_CONFIG,
    ANIMATOR_CONFIG,
    AUDIO_CONFIG,
    UPSCALER_CONFIG,
    MODELS_CACHE_DIR,
    DEVICE,
)


# ============================================================================
# Console UI
# ============================================================================

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BLUE = "\033[94m"


def ok(msg: str) -> None:
    print(f"  {C.GREEN}✓{C.RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")


def err(msg: str) -> None:
    print(f"  {C.RED}✗{C.RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {C.CYAN}→{C.RESET}  {msg}")


def head(msg: str) -> None:
    print(f"\n{C.BOLD}{C.BLUE}{msg}{C.RESET}")


def dim(msg: str) -> None:
    print(f"  {C.DIM}{msg}{C.RESET}")


# ============================================================================
# Configuration helpers
# ============================================================================

def _hf_token() -> Optional[str]:
    """
    Reads HUGGINGFACE_TOKEN from .env/environment.

    The downloader works without a token for public repositories.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
    return token or None


def _safe_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as exc:
        err(f"Nuk mund të krijohet direktoriumi '{path}': {exc}")
        return False


def _cleanup_memory() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ============================================================================
# Model registry
# ============================================================================

MODELS = {
    "face": {
        "name": FACE_CONFIG["model_name"],
        "description": "Face detection + ArcFace embedding",
        "size_gb": 0.5,
        "cache_dir": Path.home() / ".insightface",
    },
    "llm": {
        "name": LLM_CONFIG["model_name"],
        "description": "Story generation — Mistral 7B Instruct",
        "size_gb": 14.0,
        "cache_dir": Path(LLM_CONFIG["model_cache_dir"]),
    },
    "sd": {
        "name": DIFFUSION_CONFIG["model_name"],
        "description": "Frame generation — Stable Diffusion",
        "size_gb": 4.0,
        "cache_dir": Path(DIFFUSION_CONFIG["model_cache_dir"]),
    },
    "ip": {
        "name": DIFFUSION_CONFIG["ip_adapter_model"],
        "description": "Face integration — IP-Adapter SD1.5",
        "size_gb": 0.3,
        "cache_dir": Path(DIFFUSION_CONFIG["model_cache_dir"]),
    },
    "anim": {
        "name": ANIMATOR_CONFIG["model_name"],
        "description": "Frame animation — AnimateDiff",
        "size_gb": 1.8,
        "cache_dir": Path(ANIMATOR_CONFIG["model_cache_dir"]),
    },
    "tts": {
        "name": AUDIO_CONFIG["tts_model"],
        "description": "Text-to-Speech — Coqui TTS",
        "size_gb": 0.8,
        "cache_dir": Path(AUDIO_CONFIG["tts_cache_dir"]),
    },
    "esrgan": {
        "name": UPSCALER_CONFIG["model_name"],
        "description": "Video upscaling — RealESRGAN",
        "size_gb": 0.3,
        "cache_dir": Path(UPSCALER_CONFIG["model_cache_dir"]),
    },
}


# ============================================================================
# InsightFace
# ============================================================================

def download_insightface() -> bool:
    head("InsightFace — Face Detection & Embedding")

    model_name = FACE_CONFIG["model_name"]
    cache_dir = Path.home() / ".insightface"

    info(f"Model: {model_name}")
    info(f"Cache: {cache_dir}")

    try:
        from insightface.app import FaceAnalysis

        if not _safe_mkdir(cache_dir):
            return False

        info(f"Duke shkarkuar/verifikuar {model_name}...")
        start = time.time()

        app = FaceAnalysis(
            name=model_name,
            root=str(cache_dir),
            providers=["CPUExecutionProvider"],
        )

        app.prepare(
            ctx_id=-1,
            det_size=FACE_CONFIG.get("det_size", (320, 320)),
        )

        del app
        _cleanup_memory()

        elapsed = time.time() - start

        if _check_insightface():
            ok(f"InsightFace është gati ({elapsed:.0f}s)")
            return True

        err("InsightFace u inicializua, por modeli nuk u gjet në cache.")
        return False

    except ImportError:
        err("insightface nuk është instaluar.")
        dim("Ekzekuto: pip install insightface onnxruntime-gpu")
        return False

    except Exception as exc:
        err(f"Gabim InsightFace: {exc}")
        return False


# ============================================================================
# Hugging Face LLM
# ============================================================================

def download_llm() -> bool:
    model_name = LLM_CONFIG["model_name"]
    cache_dir = Path(LLM_CONFIG["model_cache_dir"])
    token = _hf_token()

    head(f"LLM — {model_name}")

    info(f"Cache: {cache_dir}")
    info("Modeli mund të kërkojë disa GB hapësirë.")

    if not token:
        warn("HUGGINGFACE_TOKEN nuk u gjet.")
        dim("Nëse repository është gated, duhet të pranosh licencën në Hugging Face dhe të vendosësh tokenin në .env.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not _safe_mkdir(cache_dir):
            return False

        start = time.time()

        info("Duke shkarkuar/verifikuar tokenizerin...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            token=token,
        )

        del tokenizer
        _cleanup_memory()

        ok(f"Tokenizeri është gati ({time.time() - start:.0f}s)")

        info("Duke shkarkuar/verifikuar peshat e LLM...")

        dtype = (
            torch.float16
            if DEVICE == "cuda"
            else torch.float32
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
            device_map="cpu",
            token=token,
            low_cpu_mem_usage=True,
        )

        del model
        _cleanup_memory()

        if _check_llm():
            ok(f"LLM është gati ({time.time() - start:.0f}s)")
            return True

        err("LLM u shkarkua, por cache nuk kaloi kontrollin.")
        return False

    except ImportError:
        err("transformers/accelerate nuk janë instaluar.")
        dim("Ekzekuto: pip install transformers accelerate")
        return False

    except Exception as exc:
        message = str(exc)

        if "401" in message or "403" in message or "gated" in message.lower():
            err("Hugging Face refuzoi aksesin në model.")
            info("Kontrollo licencën dhe HUGGINGFACE_TOKEN në .env.")
        elif "out of memory" in message.lower():
            err("Memoria RAM/VRAM nuk mjafton për inicializimin e modelit.")
        else:
            err(f"Gabim LLM: {exc}")

        return False


# ============================================================================
# Stable Diffusion
# ============================================================================

def download_stable_diffusion() -> bool:
    model_name = DIFFUSION_CONFIG["model_name"]
    cache_dir = Path(DIFFUSION_CONFIG["model_cache_dir"])
    token = _hf_token()

    head(f"Stable Diffusion — {model_name}")

    info(f"Cache: {cache_dir}")

    try:
        import torch
        from diffusers import StableDiffusionPipeline

        if not _safe_mkdir(cache_dir):
            return False

        info("Duke shkarkuar/verifikuar Stable Diffusion...")
        start = time.time()

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=(
                torch.float16
                if DEVICE == "cuda"
                else torch.float32
            ),
            safety_checker=None,
            requires_safety_checker=False,
            token=token,
        )

        del pipe
        _cleanup_memory()

        elapsed = time.time() - start

        if _check_sd():
            ok(f"Stable Diffusion është gati ({elapsed:.0f}s)")
            return True

        err("Stable Diffusion nuk kaloi kontrollin e cache.")
        return False

    except ImportError:
        err("diffusers/transformers nuk janë instaluar.")
        dim("Ekzekuto: pip install diffusers transformers accelerate")
        return False

    except Exception as exc:
        err(f"Gabim Stable Diffusion: {exc}")
        return False


# ============================================================================
# IP-Adapter
# ============================================================================

def download_ip_adapter() -> bool:
    model_name = DIFFUSION_CONFIG["ip_adapter_model"]
    cache_dir = Path(DIFFUSION_CONFIG["model_cache_dir"])
    token = _hf_token()

    head(f"IP-Adapter — {model_name}")

    info(f"Cache: {cache_dir}")

    try:
        from huggingface_hub import hf_hub_download

        if not _safe_mkdir(cache_dir):
            return False

        info("Duke shkarkuar ip-adapter_sd15.bin...")
        start = time.time()

        path = hf_hub_download(
            repo_id=model_name,
            filename="models/ip-adapter_sd15.bin",
            cache_dir=str(cache_dir),
            token=token,
        )

        elapsed = time.time() - start

        if not Path(path).exists():
            err("Hugging Face nuk raportoi një file ekzistues.")
            return False

        ok(
            f"IP-Adapter është gati → "
            f"{Path(path).name} ({elapsed:.0f}s)"
        )

        return _check_ip_adapter()

    except ImportError:
        err("huggingface_hub nuk është instaluar.")
        dim("Ekzekuto: pip install huggingface-hub")
        return False

    except Exception as exc:
        err(f"Gabim IP-Adapter: {exc}")
        return False


# ============================================================================
# AnimateDiff
# ============================================================================

def download_animatediff() -> bool:
    model_name = ANIMATOR_CONFIG["model_name"]
    cache_dir = Path(ANIMATOR_CONFIG["model_cache_dir"])
    token = _hf_token()

    head(f"AnimateDiff — {model_name}")

    info(f"Cache: {cache_dir}")

    try:
        import torch
        from diffusers import MotionAdapter

        if not _safe_mkdir(cache_dir):
            return False

        info("Duke shkarkuar/verifikuar AnimateDiff...")
        start = time.time()

        adapter = MotionAdapter.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=(
                torch.float16
                if DEVICE == "cuda"
                else torch.float32
            ),
            token=token,
        )

        del adapter
        _cleanup_memory()

        elapsed = time.time() - start

        if _check_animatediff():
            ok(f"AnimateDiff është gati ({elapsed:.0f}s)")
            return True

        err("AnimateDiff nuk kaloi kontrollin e cache.")
        return False

    except ImportError:
        err("diffusers nuk është instaluar.")
        dim("Ekzekuto: pip install diffusers")
        return False

    except Exception as exc:
        err(f"Gabim AnimateDiff: {exc}")
        return False


# ============================================================================
# TTS
# ============================================================================

def download_tts() -> bool:
    tts_model = AUDIO_CONFIG["tts_model"]
    cache_dir = Path(AUDIO_CONFIG["tts_cache_dir"])

    head(f"TTS — {tts_model}")

    info(f"Cache: {cache_dir}")

    try:
        from TTS.api import TTS

        if not _safe_mkdir(cache_dir):
            return False

        info(f"Duke shkarkuar/verifikuar {tts_model}...")
        start = time.time()

        tts = TTS(
            model_name=tts_model,
            progress_bar=True,
            gpu=False,
        )

        elapsed = time.time() - start
        ok(f"TTS është gati ({elapsed:.0f}s)")

        # Small runtime verification.
        test_path = cache_dir / "_falconai_tts_test.wav"

        try:
            info("Duke testuar TTS...")

            tts.tts_to_file(
                text="Përshëndetje.",
                file_path=str(test_path),
            )

            if not test_path.exists() or test_path.stat().st_size == 0:
                err("TTS u inicializua, por nuk krijoi audio valide.")
                return False

            ok("Testi TTS kaloi me sukses.")

        finally:
            test_path.unlink(missing_ok=True)

        del tts
        _cleanup_memory()

        return True

    except ImportError:
        err("TTS (Coqui) nuk është instaluar.")
        dim("Ekzekuto: pip install TTS")
        return False

    except Exception as exc:
        err(f"Gabim TTS: {exc}")

        warn("Po provohet modeli fallback në anglisht...")

        try:
            from TTS.api import TTS

            fallback = "tts_models/en/ljspeech/tacotron2-DDC"

            info(f"Fallback: {fallback}")

            tts = TTS(
                model_name=fallback,
                progress_bar=True,
                gpu=False,
            )

            del tts
            _cleanup_memory()

            ok(f"TTS fallback është gati: {fallback}")
            return True

        except Exception as fallback_exc:
            err(f"Fallback TTS dështoi: {fallback_exc}")
            return False


# ============================================================================
# RealESRGAN
# ============================================================================

def download_realesrgan() -> bool:
    model_name = UPSCALER_CONFIG["model_name"]
    cache_dir = Path(UPSCALER_CONFIG["model_cache_dir"])

    head(f"RealESRGAN — {model_name}")

    info(f"Cache: {cache_dir}")

    try:
        from huggingface_hub import hf_hub_download

        if not _safe_mkdir(cache_dir):
            return False

        filename = "RealESRGAN_x4plus_anime_6B.pth"

        info(f"Duke shkarkuar {filename}...")
        start = time.time()

        model_path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename=filename,
            cache_dir=str(cache_dir),
            token=_hf_token(),
        )

        elapsed = time.time() - start

        if not Path(model_path).exists():
            err("Modeli RealESRGAN nuk u gjet pas shkarkimit.")
            return False

        ok(
            f"RealESRGAN është gati → "
            f"{Path(model_path).name} ({elapsed:.0f}s)"
        )

        return _check_esrgan()

    except ImportError:
        err("huggingface_hub nuk është instaluar.")
        dim("Ekzekuto: pip install huggingface-hub")
        return False

    except Exception as exc:
        err(f"Gabim RealESRGAN: {exc}")
        return False


# ============================================================================
# Model checks
# ============================================================================

def _check_insightface() -> bool:
    cache = (
        Path.home()
        / ".insightface"
        / "models"
        / FACE_CONFIG["model_name"]
    )

    return cache.exists() and any(cache.iterdir())


def _check_llm() -> bool:
    cache = Path(LLM_CONFIG["model_cache_dir"])

    if not cache.exists():
        return False

    # A valid Transformers snapshot normally contains config.json.
    configs = list(cache.rglob("config.json"))

    if not configs:
        return False

    model_name = LLM_CONFIG["model_name"].split("/")[-1].lower()

    # Prefer a config that belongs to this model, but don't reject
    # Hugging Face cache layouts that use hashed snapshot directories.
    for config in configs:
        try:
            data = config.read_text(encoding="utf-8")
            if model_name in data.lower() or config.exists():
                return True
        except Exception:
            continue

    return False


def _check_sd() -> bool:
    cache = Path(DIFFUSION_CONFIG["model_cache_dir"])

    if not cache.exists():
        return False

    return any(cache.rglob("model_index.json"))


def _check_ip_adapter() -> bool:
    cache = Path(DIFFUSION_CONFIG["model_cache_dir"])

    if not cache.exists():
        return False

    return any(cache.rglob("ip-adapter_sd15.bin"))


def _check_animatediff() -> bool:
    cache = Path(ANIMATOR_CONFIG["model_cache_dir"])

    if not cache.exists():
        return False

    # AnimateDiff repositories may use either safetensors or bin weights.
    return (
        any(cache.rglob("*.safetensors"))
        or any(cache.rglob("*.bin"))
    )


def _check_tts() -> bool:
    """
    Do not instantiate the complete TTS model merely to check availability.

    The previous implementation created ModelManager(), which could report
    True even when the configured model itself was not downloaded.
    """
    try:
        from TTS.utils.manage import ModelManager

        manager = ModelManager()

        model_name = AUDIO_CONFIG["tts_model"]

        # Coqui's manager exposes model metadata differently across versions.
        # We therefore use its cache/model listing only when available.
        if hasattr(manager, "list_models"):
            models = manager.list_models()

            if isinstance(models, (list, tuple)):
                return model_name in models

        # Fallback: inspect the common Coqui cache locations.
        home = Path.home()

        possible_roots = [
            home / ".cache" / "tts",
            home / ".local" / "share" / "tts",
            Path(AUDIO_CONFIG["tts_cache_dir"]),
        ]

        model_fragment = model_name.replace("/", "--").lower()

        for root in possible_roots:
            if not root.exists():
                continue

            for path in root.rglob("*"):
                if model_fragment in str(path).lower():
                    return True

        return False

    except Exception:
        return False


def _check_esrgan() -> bool:
    cache = Path(UPSCALER_CONFIG["model_cache_dir"])

    if not cache.exists():
        return False

    return any(cache.rglob("*.pth"))


# ============================================================================
# Status
# ============================================================================

CHECKERS: Dict[str, Callable[[], bool]] = {
    "face": _check_insightface,
    "llm": _check_llm,
    "sd": _check_sd,
    "ip": _check_ip_adapter,
    "anim": _check_animatediff,
    "tts": _check_tts,
    "esrgan": _check_esrgan,
}


def check_models() -> Dict[str, bool]:
    head("Kontrollo modelet e shkarkuara")

    status: Dict[str, bool] = {}

    for key, check_fn in CHECKERS.items():
        model = MODELS[key]

        try:
            present = bool(check_fn())
        except Exception as exc:
            logger_message = f"Kontrolli i {model['name']} dështoi: {exc}"
            dim(logger_message)
            present = False

        status[key] = present

        icon = (
            f"{C.GREEN}✓{C.RESET}"
            if present
            else f"{C.RED}✗{C.RESET}"
        )

        size = f"~{model['size_gb']:.1f}GB"
        label = model["name"][:45].ljust(45)
        note = "" if present else f"{C.YELLOW} ← nevojitet{C.RESET}"

        print(
            f"  {icon}  {label}  "
            f"{C.DIM}{size}{C.RESET}{note}"
        )

    present_count = sum(status.values())
    total = len(status)

    missing_gb = sum(
        MODELS[key]["size_gb"]
        for key, present in status.items()
        if not present
    )

    print()

    if present_count == total:
        ok(f"Të gjitha {total} modelet janë të pranishme!")
    else:
        missing = total - present_count

        warn(f"{present_count}/{total} modele të pranishme")
        info(
            f"{missing} modele mungojnë "
            f"(~{missing_gb:.1f}GB sipas vlerësimit)"
        )

    return status


# ============================================================================
# Disk
# ============================================================================

def _print_disk_space() -> None:
    try:
        base = Path(MODELS_CACHE_DIR)

        if not base.exists():
            base = Path.home()

        total, used, free = shutil.disk_usage(base)

        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)

        if free_gb > 25:
            color = C.GREEN
        elif free_gb > 10:
            color = C.YELLOW
        else:
            color = C.RED

        print(
            f"\n  {C.DIM}Disk:{C.RESET} "
            f"{color}{free_gb:.1f}GB e lirë{C.RESET} "
            f"{C.DIM}nga {total_gb:.1f}GB totale{C.RESET}"
        )

        if free_gb < 10:
            warn("Hapësira e lirë është shumë e ulët.")
        elif free_gb < 25:
            warn(
                f"Hapësirë e kufizuar ({free_gb:.1f}GB). "
                f"Rekomandohen të paktën 25GB."
            )

    except Exception as exc:
        warn(f"Nuk mund të kontrollohet hapësira e diskut: {exc}")


# ============================================================================
# UI
# ============================================================================

def _print_banner() -> None:
    print(
        f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗
║       FalconAI Kids — Model Downloader               ║
╚══════════════════════════════════════════════════════╝{C.RESET}
"""
    )


# ============================================================================
# Download registry
# ============================================================================

DOWNLOADERS: Dict[str, Callable[[], bool]] = {
    "face": download_insightface,
    "llm": download_llm,
    "sd": download_stable_diffusion,
    "ip": download_ip_adapter,
    "anim": download_animatediff,
    "tts": download_tts,
    "esrgan": download_realesrgan,
}


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    _print_banner()

    parser = argparse.ArgumentParser(
        description="Shkarko dhe verifiko modelet AI për FalconAI Kids."
    )

    parser.add_argument(
        "--model",
        choices=list(DOWNLOADERS.keys()) + ["all"],
        default="all",
        help="Modeli që duhet shkarkuar. Default: all.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Vetëm kontrollo modelet ekzistuese.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Riekzekuto downloader-in edhe nëse modeli duket ekzistues.",
    )

    args = parser.parse_args()

    _print_disk_space()

    # ----------------------------------------------------------------------
    # Check-only mode
    # ----------------------------------------------------------------------

    if args.check:
        check_models()
        return

    # ----------------------------------------------------------------------
    # Determine download list
    # ----------------------------------------------------------------------

    if args.model == "all":
        to_download = list(DOWNLOADERS.keys())
    else:
        to_download = [args.model]

    results: Dict[str, bool] = {}

    start_total = time.time()

    # ----------------------------------------------------------------------
    # Download
    # ----------------------------------------------------------------------

    for key in to_download:
        model = MODELS[key]

        print()
        print(f"{C.BOLD}{'─' * 54}{C.RESET}")
        print(
            f"{C.BOLD}  Model: {model['name']}{C.RESET}"
        )
        print(f"{C.DIM}  {model['description']}{C.RESET}")
        print(f"{C.BOLD}{'─' * 54}{C.RESET}")

        if not args.force:
            try:
                if CHECKERS[key]():
                    ok(
                        f"{model['name']} tashmë ekziston. "
                        f"Po kalohet."
                    )
                    results[key] = True
                    continue
            except Exception:
                pass

        try:
            results[key] = bool(DOWNLOADERS[key]())

        except KeyboardInterrupt:
            warn("\nShkarkimi u ndërpre nga përdoruesi.")
            results[key] = False
            break

        except Exception as exc:
            err(
                f"Gabim i papritur gjatë shkarkimit të "
                f"{model['name']}: {exc}"
            )
            results[key] = False

        finally:
            _cleanup_memory()

    # ----------------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------------

    elapsed = time.time() - start_total

    success_count = sum(
        1 for value in results.values()
        if value
    )

    total_count = len(results)

    print()
    print(f"{'─' * 54}")
    print(f"{C.BOLD}  Rezultati Final{C.RESET}")
    print(f"{'─' * 54}")

    for key, success in results.items():
        icon = (
            f"{C.GREEN}✓{C.RESET}"
            if success
            else f"{C.RED}✗{C.RESET}"
        )

        print(
            f"  {icon}  {MODELS[key]['name']}"
        )

    print(f"{'─' * 54}")

    if total_count:
        print(
            f"  {success_count}/{total_count} "
            f"modele u përfunduan me sukses"
        )

    print(
        f"  Koha totale: "
        f"{elapsed:.0f}s ({elapsed / 60:.1f} min)"
    )

    # ----------------------------------------------------------------------
    # Success / failure
    # ----------------------------------------------------------------------

    if (
        total_count > 0
        and success_count == total_count
    ):
        print(
            f"\n  {C.BOLD}{C.GREEN}"
            f"Të gjitha modelet janë gati!"
            f"{C.RESET}"
        )

        print(
            f"  {C.DIM}Tani mund të ekzekutosh:{C.RESET}"
        )

        print(
            f"  {C.CYAN}"
            "python main.py --photo foto.jpg "
            "--name 'Emri' --birthday 2018-05-10"
            f"{C.RESET}\n"
        )

        return

    failed = [
        key
        for key, success in results.items()
        if not success
    ]

    if failed:
        warn(
            f"Modelet që dështuan: "
            f"{', '.join(failed)}"
        )

    info(
        "Për të provuar përsëri: "
        "python scripts/download_models.py"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()
