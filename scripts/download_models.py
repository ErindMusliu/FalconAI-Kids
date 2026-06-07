import argparse
import sys
import time
from pathlib import Path

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

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"
    BLUE   = "\033[94m"

def ok(msg):    print(f"  {C.GREEN}✓{C.RESET}  {msg}")
def warn(msg):  print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")
def err(msg):   print(f"  {C.RED}✗{C.RESET}  {msg}")
def info(msg):  print(f"  {C.CYAN}→{C.RESET}  {msg}")
def head(msg):  print(f"\n{C.BOLD}{C.BLUE}{msg}{C.RESET}")
def dim(msg):   print(f"  {C.DIM}{msg}{C.RESET}")

MODELS = {
    "face": {
        "name"       : "InsightFace (buffalo_l)",
        "description": "Face detection + ArcFace embedding",
        "size_gb"    : 0.5,
        "cache_dir"  : Path.home() / ".insightface",
    },
    "llm": {
        "name"       : LLM_CONFIG["model_name"],
        "description": "Story generation — Mistral 7B Instruct",
        "size_gb"    : 14.0,
        "cache_dir"  : Path(LLM_CONFIG["model_cache_dir"]),
    },
    "sd": {
        "name"       : DIFFUSION_CONFIG["model_name"],
        "description": "Frame generation — Stable Diffusion v1.5",
        "size_gb"    : 4.0,
        "cache_dir"  : Path(DIFFUSION_CONFIG["model_cache_dir"]),
    },
    "ip": {
        "name"       : DIFFUSION_CONFIG["ip_adapter_model"],
        "description": "Face integration — IP-Adapter SD1.5",
        "size_gb"    : 0.3,
        "cache_dir"  : Path(DIFFUSION_CONFIG["model_cache_dir"]),
    },
    "anim": {
        "name"       : ANIMATOR_CONFIG["model_name"],
        "description": "Frame animation — AnimateDiff v1.5",
        "size_gb"    : 1.8,
        "cache_dir"  : Path(ANIMATOR_CONFIG["model_cache_dir"]),
    },
    "tts": {
        "name"       : AUDIO_CONFIG["tts_model"],
        "description": "Text-to-Speech — Coqui TTS",
        "size_gb"    : 0.8,
        "cache_dir"  : Path(AUDIO_CONFIG["tts_cache_dir"]),
    },
    "esrgan": {
        "name"       : UPSCALER_CONFIG["model_name"],
        "description": "Video upscaling — RealESRGAN x4",
        "size_gb"    : 0.3,
        "cache_dir"  : Path(UPSCALER_CONFIG["model_cache_dir"]),
    },
}

def download_insightface() -> bool:
    head("InsightFace — Face Detection & Embedding")
    info(f"Model: {FACE_CONFIG['model_name']}")
    info(f"Cache: {Path.home() / '.insightface'}")

    try:
        import insightface
        from insightface.app import FaceAnalysis

        info("Duke shkarkuar buffalo_l...")
        start = time.time()

        app = FaceAnalysis(
            name=FACE_CONFIG["model_name"],
            root=str(Path.home() / ".insightface"),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))

        elapsed = time.time() - start
        ok(f"InsightFace u shkarkua ({elapsed:.0f}s)")
        return True

    except ImportError:
        err("insightface nuk është instaluar")
        dim("Ekzekuto: pip install insightface onnxruntime-gpu")
        return False
    except Exception as e:
        err(f"Gabim: {e}")
        return False

def download_llm() -> bool:
    model_name = LLM_CONFIG["model_name"]
    cache_dir  = LLM_CONFIG["model_cache_dir"]

    head(f"LLM — {model_name}")
    info(f"Madhësia: ~14GB")
    info(f"Cache: {cache_dir}")
    warn("Kjo mund të marrë 20-40 minuta sipas shpejtësisë së internetit")

    hf_token = _get_hf_token()
    if not hf_token:
        warn("HUGGINGFACE_TOKEN nuk u gjet në .env")
        warn("Mistral 7B kërkon pranim licence në huggingface.co")
        info("1. Shko te: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")
        info("2. Kliko 'Agree and access repository'")
        info("3. Shto HUGGINGFACE_TOKEN në .env file-in tënd")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        info("Duke shkarkuar tokenizerin...")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=hf_token or None,
        )

        ok(f"Tokenizeri u shkarkua ({time.time()-start:.0f}s)")

        info("Duke shkarkuar peshat e modelit (~14GB)...")
        info("Progres:")
        start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="cpu",
            token=hf_token or None,
            low_cpu_mem_usage=True,
        )

        elapsed = time.time() - start
        ok(f"Mistral 7B u shkarkua ({elapsed:.0f}s)")

        del model
        import gc
        gc.collect()

        return True

    except ImportError:
        err("transformers nuk është instaluar")
        dim("Ekzekuto: pip install transformers accelerate")
        return False
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower():
            err("Akses i refuzuar — nevojitet pranim licence")
            info("Shko te: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")
            info("Kliko 'Agree and access repository' dhe shto tokenin në .env")
        else:
            err(f"Gabim: {e}")
        return False

def download_stable_diffusion() -> bool:
    model_name = DIFFUSION_CONFIG["model_name"]
    cache_dir  = DIFFUSION_CONFIG["model_cache_dir"]

    head(f"Stable Diffusion — {model_name}")
    info(f"Madhësia: ~4GB")
    info(f"Cache: {cache_dir}")

    try:
        from diffusers import StableDiffusionPipeline
        import torch

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        info("Duke shkarkuar SD v1.5...")
        start = time.time()

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        elapsed = time.time() - start
        ok(f"Stable Diffusion u shkarkua ({elapsed:.0f}s)")

        del pipe
        import gc; gc.collect()
        return True

    except ImportError:
        err("diffusers nuk është instaluar")
        dim("Ekzekuto: pip install diffusers transformers accelerate")
        return False
    except Exception as e:
        err(f"Gabim: {e}")
        return False


def download_ip_adapter() -> bool:
    model_name = DIFFUSION_CONFIG["ip_adapter_model"]
    cache_dir  = DIFFUSION_CONFIG["model_cache_dir"]

    head(f"IP-Adapter — {model_name}")
    info(f"Madhësia: ~300MB")
    info(f"Cache: {cache_dir}")

    try:
        from huggingface_hub import hf_hub_download

        info("Duke shkarkuar ip-adapter_sd15.bin...")
        start = time.time()

        path = hf_hub_download(
            repo_id=model_name,
            filename="models/ip-adapter_sd15.bin",
            cache_dir=cache_dir,
            token=_get_hf_token() or None,
        )

        elapsed = time.time() - start
        ok(f"IP-Adapter u shkarkua → {Path(path).name} ({elapsed:.0f}s)")
        return True

    except ImportError:
        err("huggingface_hub nuk është instaluar")
        dim("Ekzekuto: pip install huggingface-hub")
        return False
    except Exception as e:
        err(f"Gabim: {e}")
        return False


def download_animatediff() -> bool:
    model_name = ANIMATOR_CONFIG["model_name"]
    cache_dir  = ANIMATOR_CONFIG["model_cache_dir"]

    head(f"AnimateDiff — {model_name}")
    info(f"Madhësia: ~1.8GB")
    info(f"Cache: {cache_dir}")

    try:
        from diffusers import MotionAdapter
        import torch

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        info("Duke shkarkuar AnimateDiff motion adapter...")
        start = time.time()

        adapter = MotionAdapter.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )

        elapsed = time.time() - start
        ok(f"AnimateDiff u shkarkua ({elapsed:.0f}s)")

        del adapter
        import gc; gc.collect()
        return True

    except ImportError:
        err("diffusers nuk është instaluar")
        return False
    except Exception as e:
        err(f"Gabim: {e}")
        return False

def download_tts() -> bool:
    tts_model = AUDIO_CONFIG["tts_model"]
    cache_dir = AUDIO_CONFIG["tts_cache_dir"]

    head(f"TTS (Coqui) — {tts_model}")
    info(f"Madhësia: ~800MB")
    info(f"Cache: {cache_dir}")

    try:
        from TTS.api import TTS

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        info(f"Duke shkarkuar {tts_model}...")
        start = time.time()

        tts = TTS(
            model_name=tts_model,
            progress_bar=True,
            gpu=False,
        )

        elapsed = time.time() - start
        ok(f"TTS u shkarkua ({elapsed:.0f}s)")

        info("Duke bërë provë TTS...")
        test_path = Path(cache_dir) / "test.wav"
        tts.tts_to_file(
            text="Përshëndetje nga FalconAI Kids!",
            file_path=str(test_path),
        )
        if test_path.exists():
            ok("Provë TTS kaloi me sukses")
            test_path.unlink()

        return True

    except ImportError:
        err("TTS (Coqui) nuk është instaluar")
        dim("Ekzekuto: pip install TTS")

        warn("Pa TTS, audio do të jetë tone placeholder")
        warn("Pipeline vazhdon, por pa zë tregimi")
        return False
    except Exception as e:
        err(f"Gabim duke shkarkuar TTS: {e}")

        warn("Duke provuar modelin fallback anglisht...")
        try:
            from TTS.api import TTS
            fallback = "tts_models/en/ljspeech/tacotron2-DDC"
            info(f"Fallback: {fallback}")
            tts = TTS(model_name=fallback, progress_bar=True, gpu=False)
            ok(f"TTS fallback u shkarkua: {fallback}")
            return True
        except Exception as e2:
            err(f"Fallback TTS gjithashtu deshtoi: {e2}")
            return False

def download_realesrgan() -> bool:
    model_name = UPSCALER_CONFIG["model_name"]
    cache_dir  = UPSCALER_CONFIG["model_cache_dir"]

    head(f"RealESRGAN — {model_name}")
    info(f"Madhësia: ~300MB")
    info(f"Cache: {cache_dir}")

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from huggingface_hub import hf_hub_download

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        info("Duke shkarkuar RealESRGAN_x4plus_anime_6B.pth...")
        start = time.time()

        model_path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x4plus_anime_6B.pth",
            cache_dir=cache_dir,
        )

        elapsed = time.time() - start
        ok(f"RealESRGAN u shkarkua → {Path(model_path).name} ({elapsed:.0f}s)")
        return True

    except ImportError:
        err("realesrgan/basicsr nuk janë instaluar")
        dim("Ekzekuto: pip install realesrgan basicsr")
        return False
    except Exception as e:
        err(f"Gabim: {e}")
        return False

def check_models() -> dict:
    head("Kontrollo modelet e shkarkuara")
    status = {}

    checks = {
        "face"  : _check_insightface,
        "llm"   : _check_llm,
        "sd"    : _check_sd,
        "ip"    : _check_ip_adapter,
        "anim"  : _check_animatediff,
        "tts"   : _check_tts,
        "esrgan": _check_esrgan,
    }

    for key, check_fn in checks.items():
        model   = MODELS[key]
        present = check_fn()
        status[key] = present

        icon   = f"{C.GREEN}✓{C.RESET}" if present else f"{C.RED}✗{C.RESET}"
        size   = f"~{model['size_gb']:.1f}GB"
        label  = model["name"][:45].ljust(45)
        note   = "" if present else f"{C.YELLOW}  ← nevojitet{C.RESET}"

        print(f"  {icon}  {label}  {C.DIM}{size}{C.RESET}{note}")

    present_count = sum(status.values())
    total         = len(status)
    total_gb      = sum(
        MODELS[k]["size_gb"] for k, v in status.items() if not v
    )

    print()
    if present_count == total:
        ok(f"Të gjitha {total} modelet janë të pranishme!")
    else:
        missing = total - present_count
        warn(f"{present_count}/{total} modele të pranishme")
        info(f"{missing} modele mungojnë (~{total_gb:.1f}GB për t'u shkarkuar)")
        info("Ekzekuto: python scripts/download_models.py")

    return status

def _check_insightface() -> bool:
    cache = Path.home() / ".insightface" / "models" / FACE_CONFIG["model_name"]
    return cache.exists() and any(cache.iterdir()) if cache.exists() else False

def _check_llm() -> bool:
    cache = Path(LLM_CONFIG["model_cache_dir"])
    if not cache.exists():
        return False
    pattern = f"*{LLM_CONFIG['model_name'].split('/')[-1].lower()}*"
    return any(cache.rglob("config.json"))

def _check_sd() -> bool:
    cache = Path(DIFFUSION_CONFIG["model_cache_dir"])
    return cache.exists() and any(cache.rglob("model_index.json"))

def _check_ip_adapter() -> bool:
    cache = Path(DIFFUSION_CONFIG["model_cache_dir"])
    return any(cache.rglob("ip-adapter_sd15.bin")) if cache.exists() else False

def _check_animatediff() -> bool:
    cache = Path(ANIMATOR_CONFIG["model_cache_dir"])
    return cache.exists() and any(cache.rglob("*.safetensors"))

def _check_tts() -> bool:
    try:
        from TTS.utils.manage import ModelManager
        manager = ModelManager()
        return True
    except Exception:
        return False

def _check_esrgan() -> bool:
    cache = Path(UPSCALER_CONFIG["model_cache_dir"])
    return any(cache.rglob("*.pth")) if cache.exists() else False

def _get_hf_token() -> str:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("HUGGINGFACE_TOKEN", "")

def _print_disk_space() -> None:
    import shutil
    total, used, free = shutil.disk_usage(MODELS_CACHE_DIR)
    free_gb  = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    color = C.GREEN if free_gb > 25 else C.YELLOW if free_gb > 10 else C.RED
    print(f"\n  {C.DIM}Disk:{C.RESET} {color}{free_gb:.1f}GB e lirë{C.RESET} "
          f"{C.DIM}nga {total_gb:.1f}GB totale{C.RESET}")

    if free_gb < 10:
        warn("Hapësirë e pamjaftueshme! Nevojiten të paktën 25GB")
    elif free_gb < 25:
        warn(f"Hapësirë e kufizuar ({free_gb:.1f}GB). Rekomandohen 25GB")

def _print_banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗
║       FalconAI Kids — Model Downloader               ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")

DOWNLOADERS = {
    "face"  : download_insightface,
    "llm"   : download_llm,
    "sd"    : download_stable_diffusion,
    "ip"    : download_ip_adapter,
    "anim"  : download_animatediff,
    "tts"   : download_tts,
    "esrgan": download_realesrgan,
}

def main():
    _print_banner()

    parser = argparse.ArgumentParser(
        description="Shkarko modelet AI për FalconAI Kids"
    )
    parser.add_argument(
        "--model",
        choices=list(DOWNLOADERS.keys()) + ["all"],
        default="all",
        help="Cili model të shkarkohet (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Vetëm kontrollo cilët modele janë të pranishëm",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Kalo modelet tashmë të shkarkuara (default: True)",
    )

    args = parser.parse_args()

    _print_disk_space()

    if args.check:
        check_models()
        return

    if args.model == "all":
        to_download = list(DOWNLOADERS.keys())
    else:
        to_download = [args.model]

    results  = {}
    start_t  = time.time()

    for key in to_download:
        model = MODELS[key]

        if args.skip_existing:
            check_fn_map = {
                "face": _check_insightface, "llm": _check_llm,
                "sd": _check_sd, "ip": _check_ip_adapter,
                "anim": _check_animatediff, "tts": _check_tts,
                "esrgan": _check_esrgan,
            }
            if check_fn_map[key]():
                print(f"\n  {C.DIM}⊘  {model['name']} tashmë ekziston, duke kaluar{C.RESET}")
                results[key] = True
                continue
        try:
            results[key] = DOWNLOADERS[key]()
        except KeyboardInterrupt:
            warn(f"\nShkarkimi u ndërpre nga përdoruesi")
            break
        except Exception as e:
            err(f"Gabim i papritur duke shkarkuar {model['name']}: {e}")
            results[key] = False

    elapsed     = time.time() - start_t
    success_n   = sum(results.values())
    total_n     = len(results)

    print(f"\n{'─'*54}")
    print(f"{C.BOLD}  Rezultati Final{C.RESET}")
    print(f"{'─'*54}")

    for key, success in results.items():
        icon  = f"{C.GREEN}✓{C.RESET}" if success else f"{C.RED}✗{C.RESET}"
        label = MODELS[key]["name"]
        print(f"  {icon}  {label}")

    print(f"{'─'*54}")
    print(f"  {success_n}/{total_n} modele u shkarkuan me sukses")
    print(f"  Koha totale: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if success_n == total_n:
        print(f"\n  {C.BOLD}{C.GREEN}Të gjitha modelet janë gati!{C.RESET}")
        print(f"  {C.DIM}Tani mund të ekzekutosh:{C.RESET}")
        print(f"  {C.CYAN}python main.py --photo foto.jpg --name 'Emri' --birthday 2018-05-10{C.RESET}\n")
    else:
        failed = [k for k, v in results.items() if not v]
        warn(f"Modelet që dështuan: {', '.join(failed)}")
        info("Provo sërish: python scripts/download_models.py")
        sys.exit(1)

if __name__ == "__main__":
    main()