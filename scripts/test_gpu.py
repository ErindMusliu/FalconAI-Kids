import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"

def ok(msg):   print(f"  {C.GREEN}✓{C.RESET}  {msg}")
def warn(msg): print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")
def err(msg):  print(f"  {C.RED}✗{C.RESET}  {msg}")
def info(msg): print(f"  {C.CYAN}→{C.RESET}  {msg}")
def head(msg): print(f"\n{C.BOLD}{C.BLUE}{'─'*54}\n  {msg}\n{'─'*54}{C.RESET}")
def row(label, value, color=C.WHITE):
    label_str = f"{C.DIM}{label}{C.RESET}".ljust(40)
    print(f"  {label_str}{color}{value}{C.RESET}")

def check_system() -> dict:
    head("Sistemi")

    results = {}

    os_name = f"{platform.system()} {platform.release()}"
    row("Sistemi Operativ", os_name)
    results["os"] = os_name

    try:
        import psutil
        cpu_name  = platform.processor() or "i panjohur"
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_logic = psutil.cpu_count(logical=True)
        ram_gb    = psutil.virtual_memory().total / (1024**3)
        row("CPU", f"{cpu_name}")
        row("CPU Cores", f"{cpu_cores} fizike / {cpu_logic} logjike")
        row("RAM", f"{ram_gb:.1f} GB")
        results["ram_gb"] = ram_gb
        if ram_gb < 16:
            warn(f"RAM e ulët ({ram_gb:.0f}GB). Rekomandohet ≥16GB për LLM")
    except ImportError:
        cpu_cores = 1
        row("CPU", platform.processor() or "i panjohur")
        info("pip install psutil për info të detajuar CPU/RAM")

    py_version = sys.version.split()[0]
    py_ok      = sys.version_info >= (3, 10)
    color      = C.GREEN if py_ok else C.RED
    row("Python", py_version, color)
    if not py_ok:
        err(f"Python {py_version} nuk mbështetet. Nevojitet ≥ 3.10")
    results["python_ok"] = py_ok

    return results

def check_pytorch() -> dict:
    head("PyTorch & CUDA")

    results = {}

    try:
        import torch

        pt_version = torch.__version__
        row("PyTorch", pt_version)
        results["torch_version"] = pt_version

        cuda_available = torch.cuda.is_available()
        results["cuda_available"] = cuda_available

        if cuda_available:
            cuda_version = torch.version.cuda or "i panjohur"
            row("CUDA", cuda_version, C.GREEN)
            results["cuda_version"] = cuda_version

            cudnn_version = torch.backends.cudnn.version()
            row("cuDNN", str(cudnn_version), C.GREEN)

            gpu_count = torch.cuda.device_count()
            row("GPU count", str(gpu_count), C.GREEN)
            results["gpu_count"] = gpu_count

            for i in range(gpu_count):
                props     = torch.cuda.get_device_properties(i)
                vram_gb   = props.total_memory / (1024**3)
                vram_color = (
                    C.GREEN  if vram_gb >= 16 else
                    C.YELLOW if vram_gb >= 8  else
                    C.RED
                )

                row(f"GPU {i} — Modeli",  props.name, C.CYAN)
                row(f"GPU {i} — VRAM",    f"{vram_gb:.1f} GB", vram_color)
                row(f"GPU {i} — Compute", f"{props.major}.{props.minor}")
                row(f"GPU {i} — SMs",     str(props.multi_processor_count))

                results[f"gpu_{i}_name"]    = props.name
                results[f"gpu_{i}_vram_gb"] = vram_gb

                if vram_gb >= 24:
                    ok(f"GPU {i}: Shkëlqyer! Mund të ekzekutojë të gjitha modelet")
                elif vram_gb >= 16:
                    ok(f"GPU {i}: Shumë mirë! Mbështet plotësisht FalconAI Kids")
                elif vram_gb >= 8:
                    warn(f"GPU {i}: Mjaftueshëm ({vram_gb:.0f}GB). "
                         f"SD + AnimateDiff do punojnë, LLM do jetë i ngadaltë")
                else:
                    warn(f"GPU {i}: VRAM e ulët ({vram_gb:.0f}GB). "
                         f"Rekomandohet ≥8GB VRAM")

            free_vram, total_vram = torch.cuda.mem_get_info(0)
            free_gb  = free_vram  / (1024**3)
            total_gb = total_vram / (1024**3)
            row("VRAM e lirë tani", f"{free_gb:.1f} / {total_gb:.1f} GB")

        else:
            warn("CUDA nuk është e disponueshme")
            row("CUDA", "Jo disponueshme", C.YELLOW)

            info("Arsyet e mundshme:")
            info("  1. Nuk ke GPU NVIDIA")
            info("  2. CUDA Toolkit nuk është instaluar")
            info("  3. PyTorch u instalua pa CUDA support")
            info("")
            info("Zgjidhja:")
            info("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

            results["cuda_available"] = False

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            row("Apple MPS (M1/M2)", "Disponueshme", C.GREEN)
            warn("MPS mbështetet pjesërisht — disa modele mund të dështojnë")

        if cuda_available:
            try:
                test = torch.zeros(1, dtype=torch.float16, device="cuda")
                row("Float16 (fp16)", "Mbështetet", C.GREEN)
                del test
            except Exception:
                warn("Float16 nuk mbështetet — do të përdoret float32 (2x më shumë VRAM)")

        ok("PyTorch u importua me sukses")

    except ImportError:
        err("PyTorch nuk është instaluar!")
        info("Instalo: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        results["torch_installed"] = False

    return results

def check_packages() -> dict:
    head("Paketat e Instaluara")

    packages = [
        ("insightface",     "insightface",      True,  "Face detection"),
        ("cv2",             "opencv-python",    True,  "Computer vision"),
        ("PIL",             "Pillow",           True,  "Image I/O"),
        ("diffusers",       "diffusers",        True,  "Stable Diffusion"),
        ("transformers",    "transformers",     True,  "LLM / Tokenizers"),
        ("accelerate",      "accelerate",       True,  "Model loading"),
        ("safetensors",     "safetensors",      True,  "Model weights"),
        ("huggingface_hub", "huggingface-hub",  True,  "HF model download"),
        ("TTS",             "TTS",              False, "Text-to-Speech"),
        ("numpy",           "numpy",            True,  "Array operations"),
        ("realesrgan",      "realesrgan",       False, "Upscaling"),
        ("basicsr",         "basicsr",          False, "RealESRGAN backend"),
        ("boto3",           "boto3",            False, "AWS S3 (opsionale)"),
        ("dotenv",          "python-dotenv",    True,  "Config .env"),
        ("tqdm",            "tqdm",             True,  "Progress bars"),
        ("sentencepiece",   "sentencepiece",    True,  "LLM tokenizer"),
        ("einops",          "einops",           True,  "Tensor ops"),
        ("psutil",          "psutil",           False, "System info"),
    ]

    results   = {}
    missing_r = []
    missing_o = []

    for import_name, pip_name, required, description in packages:
        try:
            mod     = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            label   = f"{pip_name:<20} {C.DIM}{description}{C.RESET}"
            row(label, f"v{version}", C.GREEN)
            results[pip_name] = True
        except ImportError:
            color = C.RED if required else C.YELLOW
            req_str = "kërkohet" if required else "opsionale"
            row(f"{pip_name:<20} {C.DIM}{description}{C.RESET}",
                f"MUNGON ({req_str})", color)
            results[pip_name] = False
            if required:
                missing_r.append(pip_name)
            else:
                missing_o.append(pip_name)

    if missing_r:
        print()
        err(f"Paketa të detyrueshme mungojnë: {', '.join(missing_r)}")
        info(f"Instalo: pip install {' '.join(missing_r)}")

    if missing_o:
        print()
        warn(f"Paketa opsionale mungojnë: {', '.join(missing_o)}")
        info("Pipeline mund të vazhdojë pa to, por me funksionalitet të kufizuar")

    return results

def check_ffmpeg() -> dict:
    head("FFmpeg")

    results = {}

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0].replace("ffmpeg version ", "")
            row("ffmpeg", version, C.GREEN)
            ok("ffmpeg disponueshëm")
            results["ffmpeg"] = True
        else:
            err("ffmpeg nuk funksionon")
            results["ffmpeg"] = False
    except FileNotFoundError:
        err("ffmpeg nuk është instaluar")
        info("Ubuntu/Debian: sudo apt install ffmpeg")
        info("macOS:         brew install ffmpeg")
        info("Windows:       https://ffmpeg.org/download.html")
        results["ffmpeg"] = False

    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0].replace("ffprobe version ", "")
            row("ffprobe", version, C.GREEN)
            results["ffprobe"] = True
        else:
            warn("ffprobe nuk disponueshëm (jo kritike)")
            results["ffprobe"] = False
    except FileNotFoundError:
        warn("ffprobe nuk disponueshëm (instalohet me ffmpeg)")
        results["ffprobe"] = False

    return results

def check_disk() -> dict:
    head("Hapësira e Diskut")

    results = {}

    try:
        import shutil
        from config.settings import MODELS_CACHE_DIR, DATA_DIR, BASE_DIR

        dirs_to_check = [
            (BASE_DIR,          "Projekti",       5.0),
            (MODELS_CACHE_DIR,  "Models Cache",   25.0),
            (DATA_DIR,          "Data (outputs)", 10.0),
        ]

        for check_dir, label, min_gb in dirs_to_check:
            check_dir.mkdir(parents=True, exist_ok=True)
            total, used, free = shutil.disk_usage(check_dir)
            free_gb  = free  / (1024**3)
            total_gb = total / (1024**3)
            used_gb  = used  / (1024**3)

            color = (
                C.GREEN  if free_gb >= min_gb * 2 else
                C.YELLOW if free_gb >= min_gb     else
                C.RED
            )

            row(
                f"{label} ({check_dir})",
                f"{free_gb:.1f}GB e lirë / {total_gb:.1f}GB total",
                color
            )

            if free_gb < min_gb:
                err(f"Hapësirë e pamjaftueshme për {label}! "
                    f"Nevojiten ≥{min_gb:.0f}GB, disponueshme {free_gb:.1f}GB")

            results[f"disk_{label.lower().replace(' ', '_')}"] = free_gb

    except Exception as e:
        warn(f"Nuk u kontrollua hapësira: {e}")

    return results

def run_benchmark() -> dict:
    head("GPU Benchmark")

    results = {}

    try:
        import torch

        if not torch.cuda.is_available():
            warn("CUDA nuk disponueshme — benchmark kalohet")
            return results

        device = torch.device("cuda")
        info("Duke ekzekutuar benchmark...")

        sizes = [512, 1024, 2048, 4096]
        print()
        print(f"  {'Madhësia':<12} {'Koha (ms)':<14} {'TFLOPS':<12} {'Vlerësim'}")
        print(f"  {'─'*52}")

        for n in sizes:
            a = torch.randn(n, n, device=device, dtype=torch.float16)
            b = torch.randn(n, n, device=device, dtype=torch.float16)

            torch.matmul(a, b)
            torch.cuda.synchronize()

            iterations = 10
            start = time.perf_counter()
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

            tflops = 2 * n**3 / (elapsed_ms / 1000 * 1e12)

            rating = (
                "Shkëlqyer" if tflops > 100 else
                "Shumë mirë" if tflops > 50 else
                "Mirë"      if tflops > 20 else
                "Mesatar"
            )
            color = (
                C.GREEN  if tflops > 100 else
                C.GREEN  if tflops > 50  else
                C.YELLOW if tflops > 20  else
                C.RED
            )

            print(f"  {n}x{n:<8} {elapsed_ms:<14.2f} {color}{tflops:<12.1f}{C.RESET} {rating}")

            del a, b, c

        torch.cuda.empty_cache()

        print()
        info("Memory bandwidth test...")
        size_mb = 512
        n_elem  = size_mb * 1024 * 1024 // 4

        src = torch.randn(n_elem, device=device)
        dst = torch.empty_like(src)

        torch.cuda.synchronize()
        start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed_s = (time.perf_counter() - start) / iterations

        bandwidth_gbs = (size_mb * 2) / (elapsed_s * 1024)
        color = (
            C.GREEN  if bandwidth_gbs > 500 else
            C.YELLOW if bandwidth_gbs > 200 else
            C.RED
        )
        row("Memory Bandwidth", f"{color}{bandwidth_gbs:.1f} GB/s{C.RESET}")

        del src, dst
        torch.cuda.empty_cache()

        print()
        info("Simulim Stable Diffusion (UNet-like)...")

        batch     = 1
        channels  = 4
        h = w     = 64
        time_emb  = 320

        latent  = torch.randn(batch, channels, h, w, device=device, dtype=torch.float16)
        t_emb   = torch.randn(batch, time_emb, device=device, dtype=torch.float16)

        start = time.perf_counter()
        for _ in range(5):
            x = torch.nn.functional.conv2d(
                latent,
                torch.randn(320, 4, 3, 3, device=device, dtype=torch.float16),
                padding=1
            )
            x = torch.nn.functional.group_norm(
                x.float(), 32
            ).half()
            x = torch.nn.functional.silu(x)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / 5

        color  = C.GREEN if elapsed_ms < 50 else C.YELLOW if elapsed_ms < 150 else C.RED
        row("SD UNet simulim", f"{color}{elapsed_ms:.1f}ms / step{C.RESET}")

        if elapsed_ms < 50:
            ok("GPU është i shkëlqyer për Stable Diffusion!")
        elif elapsed_ms < 150:
            ok("GPU mbështet mirë Stable Diffusion")
        else:
            warn("GPU mund të jetë i ngadaltë për SD — konsidero zvogëlimin e rezolucionit")

        del latent, t_emb, x
        torch.cuda.empty_cache()

        results["benchmark_ok"] = True

    except ImportError:
        warn("PyTorch nuk disponueshëm — benchmark kalohet")
    except Exception as e:
        warn(f"Benchmark deshtoi: {e}")

    return results

def check_config() -> dict:
    head("Konfigurimi i Projektit")

    results = {}

    try:
        from config.settings import (
            DEVICE, BASE_DIR, MODELS_CACHE_DIR,
            LLM_CONFIG, DIFFUSION_CONFIG, FACE_CONFIG,
            VIDEO_CONFIG
        )

        row("Base directory",     str(BASE_DIR))
        row("Models cache",       str(MODELS_CACHE_DIR))
        row("Device",             DEVICE,
            C.GREEN if DEVICE == "cuda" else C.YELLOW)
        row("LLM model",          LLM_CONFIG["model_name"])
        row("Diffusion model",    DIFFUSION_CONFIG["model_name"])
        row("Face model",         FACE_CONFIG["model_name"])
        row("Video resolution",   f"{VIDEO_CONFIG['resolution'][0]}x{VIDEO_CONFIG['resolution'][1]}")
        row("Video FPS",          str(VIDEO_CONFIG["fps"]))

        env_path = BASE_DIR / ".env"
        if env_path.exists():
            ok(".env file ekziston")
            results["env_exists"] = True
        else:
            warn(".env file mungon")
            info("Krijo me: cp .env.example .env")
            results["env_exists"] = False

        ok("Config u importua me sukses")
        results["config_ok"] = True

    except Exception as e:
        err(f"Gabim duke importuar config: {e}")
        results["config_ok"] = False

    return results

def print_final_report(all_results: dict) -> bool:
    head("Raporti Final")

    checks = {
        "Python ≥ 3.10"           : all_results.get("system", {}).get("python_ok", False),
        "PyTorch i instaluar"     : all_results.get("pytorch", {}).get("cuda_available") is not None,
        "CUDA disponueshme"       : all_results.get("pytorch", {}).get("cuda_available", False),
        "FFmpeg i instaluar"      : all_results.get("ffmpeg", {}).get("ffmpeg", False),
        "Paketat kryesore"        : all(
            v for k, v in all_results.get("packages", {}).items()
            if k in ["insightface", "opencv-python", "Pillow", "diffusers",
                     "transformers", "accelerate", "numpy"]
        ),
        "Config i ngarkuar"       : all_results.get("config", {}).get("config_ok", False),
    }

    all_ok = True
    for label, status in checks.items():
        if status:
            ok(label)
        else:
            err(label)
            all_ok = False

    print()

    gpu_vram = all_results.get("pytorch", {}).get("gpu_0_vram_gb", 0)
    if gpu_vram > 0:
        if gpu_vram >= 24:
            ok(f"GPU VRAM: {gpu_vram:.0f}GB — Shkëlqyer për FalconAI Kids")
        elif gpu_vram >= 16:
            ok(f"GPU VRAM: {gpu_vram:.0f}GB — Shumë mirë")
        elif gpu_vram >= 8:
            warn(f"GPU VRAM: {gpu_vram:.0f}GB — Mjaftueshëm (mund të jetë i ngadaltë)")
        else:
            err(f"GPU VRAM: {gpu_vram:.0f}GB — Jo i mjaftueshëm (rekomandohet ≥8GB)")
            all_ok = False

    print()
    print(f"{'─'*54}")

    if all_ok:
        print(f"\n  {C.BOLD}{C.GREEN}Sistemi është gati për FalconAI Kids!{C.RESET}")
        print(f"\n  {C.DIM}Hapi tjetër:{C.RESET}")
        print(f"  {C.CYAN}python scripts/download_models.py{C.RESET}")
        print(f"  {C.DIM}Pastaj:{C.RESET}")
        print(f"  {C.CYAN}python main.py --photo foto.jpg --name 'Emri' --birthday 2018-05-10{C.RESET}\n")
    else:
        print(f"\n  {C.BOLD}{C.YELLOW}Sistemi ka probleme që duhen rregulluar.{C.RESET}")
        print(f"  {C.DIM}Shiko gabimet lart dhe ndiq udhëzimet.{C.RESET}\n")

    return all_ok

def main():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗
║       FalconAI Kids — System & GPU Tester            ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")

    parser = argparse.ArgumentParser(
        description="Kontrollo sistemin për FalconAI Kids"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Vetëm info bazike (pa benchmark)",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Ekzekuto benchmark të detajuar",
    )
    args = parser.parse_args()

    all_results = {}

    all_results["system"]   = check_system()
    all_results["pytorch"]  = check_pytorch()
    all_results["packages"] = check_packages()
    all_results["ffmpeg"]   = check_ffmpeg()
    all_results["disk"]     = check_disk()
    all_results["config"]   = check_config()

    if args.bench or not args.quick:
        all_results["benchmark"] = run_benchmark()

    ready = print_final_report(all_results)

    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()