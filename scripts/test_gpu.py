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

def ok(msg: str) -> None:
    print(f"  {C.GREEN}✓{C.RESET}  {msg}")

def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")

def err(msg: str) -> None:
    print(f"  {C.RED}✗{C.RESET}  {msg}")

def info(msg: str) -> None:
    print(f"  {C.CYAN}→{C.RESET}  {msg}")

def head(msg: str) -> None:
    print(f"\n{C.BOLD}{C.BLUE}{'─'*54}\n  {msg}\n{'─'*54}{C.RESET}")

def row(label: str, value: str, color: str = C.WHITE) -> None:
    label_str = f"{C.DIM}{label}{C.RESET}".ljust(40)
    print(f"  {label_str}{color}{value}{C.RESET}")

def check_system() -> dict:
    head("Host System Environment")
    results = {}

    os_name = f"{platform.system()} {platform.release()}"
    row("Operating System", os_name)
    results["os"] = os_name

    try:
        import psutil
        cpu_name = platform.processor() or "Unknown Architecture"
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_logic = psutil.cpu_count(logical=True) or 1
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        row("Processor (CPU)", f"{cpu_name}")
        row("CPU Cores Configuration", f"{cpu_cores} Physical / {cpu_logic} Logical Threads")
        row("Total System Memory (RAM)", f"{ram_gb:.1f} GB")
        
        results["ram_gb"] = ram_gb
        if ram_gb < 16:
            warn(f"Low system memory detected ({ram_gb:.1f}GB). Optimal pipeline execution requires ≥16GB RAM.")
    except ImportError:
        cpu_cores = 1
        row("Processor (CPU)", platform.processor() or "Unknown Architecture")
        info("Missing 'psutil' package tracking framework. Run: pip install psutil for detailed metrics.")
        results["ram_gb"] = 8.0

    py_version = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 10)
    color = C.GREEN if py_ok else C.RED
    row("Python Version", py_version, color)
    if not py_ok:
        err(f"Python execution instance runtime version {py_version} is unsupported. Requires version ≥ 3.10")
    results["python_ok"] = py_ok

    return results

def check_pytorch() -> dict:
    head("PyTorch & CUDA Compute Layer")
    results = {}

    try:
        import torch
        pt_version = torch.__version__
        row("PyTorch Framework", pt_version)
        results["torch_version"] = pt_version

        cuda_available = torch.cuda.is_available()
        results["cuda_available"] = cuda_available

        if cuda_available:
            cuda_version = torch.version.cuda or "Unknown System Driver Mapping"
            row("CUDA Compute Runtime", cuda_version, C.GREEN)
            results["cuda_version"] = cuda_version

            cudnn_version = torch.backends.cudnn.version()
            row("cuDNN Engine Layer", str(cudnn_version), C.GREEN)

            gpu_count = torch.cuda.device_count()
            row("Available Discrete GPUs", str(gpu_count), C.GREEN)
            results["gpu_count"] = gpu_count

            current_device = torch.cuda.current_device()

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                vram_color = (
                    C.GREEN  if vram_gb >= 16 else
                    C.YELLOW if vram_gb >= 8  else
                    C.RED
                )

                row(f"GPU [{i}] — Model Identifier", props.name, C.CYAN)
                row(f"GPU [{i}] — Hardware VRAM", f"{vram_gb:.1f} GB", vram_color)
                row(f"GPU [{i}] — Compute Capability", f"{props.major}.{props.minor}")
                row(f"GPU [{i}] — Multiprocessors (SMs)", str(props.multi_processor_count))

                results[f"gpu_{i}_name"] = props.name
                results[f"gpu_{i}_vram_gb"] = vram_gb

                if vram_gb >= 24:
                    ok(f"GPU [{i}]: Excellent specifications. Fully capable of running top-tier models locally.")
                elif vram_gb >= 16:
                    ok(f"GPU [{i}]: Recommended baseline specs verified. Full pipeline execution supported natively.")
                elif vram_gb >= 8:
                    warn(f"GPU [{i}]: Marginal footprint detected ({vram_gb:.1f}GB VRAM). SD/AnimateDiff pipelines function, LLM inference tasks may experience throttling.")
                else:
                    warn(f"GPU [{i}]: Critical VRAM constraint ({vram_gb:.1f}GB VRAM). System requires ≥8GB VRAM for standard deployment configurations.")

            free_vram, total_vram = torch.cuda.mem_get_info(current_device)
            free_gb = free_vram / (1024**3)
            total_gb = total_vram / (1024**3)
            row("Available Volatile VRAM Allocation", f"{free_gb:.1f} / {total_gb:.1f} GB")

        else:
            warn("CUDA acceleration layers are inaccessible within the current execution environment context.")
            row("CUDA Availability", "Not Available / Missing Driver Tracks", C.YELLOW)

            info("Potential Root Causes:")
            info("  1. No compatible NVIDIA graphics processing unit discovered on host system.")
            info("  2. System Environmental paths to the local CUDA Toolkit binaries are broken or unlinked.")
            info("  3. Active PyTorch package build binaries lack pre-compiled source CUDA support hooks.")
            info("")
            info("Recommended Resolution Step:")
            info("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            results["cuda_available"] = False

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            row("Apple Silicon MPS Runtime", "Available", C.GREEN)
            warn("MPS backend tracking displays partial infrastructure cross-compilation errors — fallback warnings enabled.")

        if cuda_available:
            try:
                test_tensor = torch.zeros(1, dtype=torch.float16, device="cuda")
                row("Float16 (fp16) Mixed Precision", "Supported Natively", C.GREEN)
                del test_tensor
            except Exception:
                warn("Float16 precision formats unsupported — defaulting execution layers to float32 modes (double VRAM overhead).")

        ok("PyTorch verification completed cleanly.")

    except ImportError:
        err("PyTorch library dependency is missing or uninstalled from the target Python package ecosystem.")
        info("Execution Resolution: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        results["torch_installed"] = False

    return results

def check_packages() -> dict:
    head("Dependency Architecture Manifest")

    packages = [
        ("insightface",     "insightface",      True,  "Face detection / alignment"),
        ("cv2",             "opencv-python",    True,  "Computer vision algorithms"),
        ("PIL",             "Pillow",           True,  "Image raster input/output processing"),
        ("diffusers",       "diffusers",        True,  "Stable Diffusion engine orchestration"),
        ("transformers",    "transformers",     True,  "LLM compilation & tokenizer managers"),
        ("accelerate",      "accelerate",       True,  "Advanced weights distribution maps"),
        ("safetensors",     "safetensors",      True,  "Secure structured weight storage format"),
        ("huggingface_hub", "huggingface-hub",  True,  "Remote asset hub access tools"),
        ("TTS",             "TTS",              False, "Text-to-Speech synthesizer engine"),
        ("numpy",           "numpy",            True,  "Linear matrix algebra calculations"),
        ("realesrgan",      "realesrgan",       False, "Visual upscaling enhancement algorithms"),
        ("basicsr",         "basicsr",          False, "RealESRGAN underlying backend frameworks"),
        ("boto3",           "boto3",            False, "AWS Cloud S3 pipeline access drivers"),
        ("dotenv",          "python-dotenv",    True,  "Environmental configuration manager"),
        ("tqdm",            "tqdm",             True,  "Asynchronous command loop progress trackers"),
        ("sentencepiece",   "sentencepiece",    True,  "Advanced semantic tokenizer support models"),
        ("einops",          "einops",           True,  "Matrix multidimensional transposition layers"),
        ("psutil",          "psutil",           False, "Hardware resource utility logging metrics"),
    ]

    results = {}
    missing_required = []
    missing_optional = []

    for import_name, pip_name, required, description in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "Unknown Version")
            label = f"{pip_name:<20} {C.DIM}{description}{C.RESET}"
            row(label, f"v{version}", C.GREEN)
            results[pip_name] = True
        except ImportError:
            color = C.RED if required else C.YELLOW
            req_str = "Required Baseline" if required else "Optional Extendor"
            row(f"{pip_name:<20} {C.DIM}{description}{C.RESET}", f"MISSING ({req_str})", color)
            results[pip_name] = False
            if required:
                missing_required.append(pip_name)
            else:
                missing_optional.append(pip_name)

    if missing_required:
        print()
        err(f"Pipeline initialization terminated; critical dependencies are missing: {', '.join(missing_required)}")
        info(f"Resolution Command: pip install {' '.join(missing_required)}")

    if missing_optional:
        print()
        warn(f"Optional extensions are unpopulated within environmental spaces: {', '.join(missing_optional)}")
        info("Pipeline processes will continue executing, though select non-critical features may be hidden.")

    return results

def check_ffmpeg() -> dict:
    head("System Binary Pipeline Tools (FFmpeg)")
    results = {}

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split("\n")[0].replace("ffmpeg version ", "")
            row("ffmpeg executable path", version, C.GREEN)
            ok("FFmpeg multiplexing framework verified successfully.")
            results["ffmpeg"] = True
        else:
            err("FFmpeg execution diagnostic command reported an unhealthy status code.")
            results["ffmpeg"] = False
    except FileNotFoundError:
        err("FFmpeg binary utilities could not be mapped within current operating system environmental paths.")
        info("Platform Recovery Command Scripts:")
        info("  Ubuntu/Debian: sudo apt install ffmpeg")
        info("  macOS:         brew install ffmpeg")
        info("  Windows:       Download distributions via: https://ffmpeg.org/download.html")
        results["ffmpeg"] = False

    try:
        result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split("\n")[0].replace("ffprobe version ", "")
            row("ffprobe executable path", version, C.GREEN)
            results["ffprobe"] = True
        else:
            warn("FFprobe system analytic asset checks failed (non-critical pipeline layer).")
            results["ffprobe"] = False
    except FileNotFoundError:
        warn("FFprobe media stream probe execution tool not discovered (typically deployed along with standard FFmpeg bundles).")
        results["ffprobe"] = False

    return results

def check_disk() -> dict:
    head("Persistent Storage Array Free Space Constraints")
    results = {}

    try:
        import shutil
        from config.settings import MODELS_CACHE_DIR, DATA_DIR, BASE_DIR

        dirs_to_check = [
            (BASE_DIR,          "Principal Project Workspace", 5.0),
            (MODELS_CACHE_DIR,  "Local Inference Weights Cache Location", 25.0),
            (DATA_DIR,          "Pipeline Final Outputs Workspace", 10.0),
        ]

        for check_dir, label, min_gb in dirs_to_check:
            check_dir.mkdir(parents=True, exist_ok=True)
            total, used, free = shutil.disk_usage(check_dir)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)

            color = (
                C.GREEN  if free_gb >= min_gb * 2 else
                C.YELLOW if free_gb >= min_gb     else
                C.RED
            )

            row(f"{label} Target Space Space", f"{free_gb:.1f} GB Free Space / {total_gb:.1f} GB Volume Size", color)

            if free_gb < min_gb:
                err(f"Severe storage shortage mapped for: {label}! Infrastructure deployment limits require ≥{min_gb:.1f}GB allocation maps; Only {free_gb:.1f}GB is ready.")

            results[f"disk_{label.lower().replace(' ', '_')}"] = free_gb

    except Exception as storage_err:
        warn(f"Dynamic disk infrastructure verification loop bypassed due to interface layer anomalies: {storage_err}")

    return results

def run_benchmark() -> dict:
    head("Raw Hardware Engine Floating Point Stress Test Benchmarks")
    results = {}

    try:
        import torch

        if not torch.cuda.is_available():
            warn("CUDA acceleration is missing — calculation benchmark phase skipped.")
            return results

        device = torch.device("cuda")
        info("Initializing hardware floating point engine calculations stress matrix loop...")

        sizes = [512, 1024, 2048, 4096]
        print()
        print(f"  {'Matrix Size':<14} {'Compute (ms)':<15} {'Thruput (TFLOPS)':<18} {'System Rating'}")
        print(f"  {'─'*65}")

        for n in sizes:
            a = torch.randn(n, n, device=device, dtype=torch.float16)
            b = torch.randn(n, n, device=device, dtype=torch.float16)

            torch.matmul(a, b)
            torch.cuda.synchronize()

            loops = 10
            start_stamp = time.perf_counter()
            for _ in range(loops):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_stamp) * 1000 / loops

            tflops = 2 * (n**3) / (elapsed_ms / 1000 * 1e12)

            rating = (
                "Excellent Performance" if tflops > 100 else
                "Highly Competent"      if tflops > 50  else
                "Standard Capacity"     if tflops > 20  else
                "Baseline Structural Capacity"
            )
            color = (
                C.GREEN  if tflops > 50  else
                C.YELLOW if tflops > 20  else
                C.RED
            )

            print(f"  {n} x {n:<8} {elapsed_ms:<15.2f} {color}{tflops:<18.1f}{C.RESET} {rating}")
            del a, b, c

        torch.cuda.empty_cache()

        print()
        info("Evaluating volatile physical hardware memory interface bus bandwidth capacity...")
        payload_mb = 512
        element_counts = payload_mb * 1024 * 1024 // 4

        src_buf = torch.randn(element_counts, device=device)
        dst_buf = torch.empty_like(src_buf)

        torch.cuda.synchronize()
        start_stamp = time.perf_counter()
        transfer_loops = 20
        for _ in range(transfer_loops):
            dst_buf.copy_(src_buf)
        torch.cuda.synchronize()
        transfer_elapsed = (time.perf_counter() - start_stamp) / transfer_loops

        effective_bandwidth = (payload_mb * 2) / (transfer_elapsed * 1024)
        color = (
            C.GREEN  if effective_bandwidth > 500 else
            C.YELLOW if effective_bandwidth > 200 else
            C.RED
        )
        row("Calculated Memory Interface Bandwidth", f"{color}{effective_bandwidth:.1f} GB/s{C.RESET}")

        del src_buf, dst_buf
        torch.cuda.empty_cache()

        print()
        info("Simulating Stable Diffusion neural network engine workload blocks (UNet-like layers)...")

        batch, channels, h, w = 1, 4, 64, 64
        latent_space = torch.randn(batch, channels, h, w, device=device, dtype=torch.float16)

        start_stamp = time.perf_counter()
        simulation_steps = 5
        for _ in range(simulation_steps):
            mock_conv = torch.nn.functional.conv2d(
                latent_space,
                torch.randn(320, 4, 3, 3, device=device, dtype=torch.float16),
                padding=1
            )
            mock_norm = torch.nn.functional.group_norm(mock_conv.float(), 32).half()
            mock_act  = torch.nn.functional.silu(mock_norm)
            
        torch.cuda.synchronize()
        sim_elapsed_ms = (time.perf_counter() - start_stamp) * 1000 / simulation_steps

        color = C.GREEN if sim_elapsed_ms < 50 else C.YELLOW if sim_elapsed_ms < 150 else C.RED
        row("SD UNet Synthesis Latency Approximation", f"{color}{sim_elapsed_ms:.1f} ms / processing step{C.RESET}")

        if sim_elapsed_ms < 50:
            ok("Hardware acceleration profiles are exceptionally well suited for local Stable Diffusion generation tasks.")
        elif sim_elapsed_ms < 150:
            ok("Local compute performance matches core framework execution baseline requirements.")
        else:
            warn("Processing latencies are elevated. Consider decreasing processing resolution schemas inside config mappings.")

        del latent_space, mock_conv, mock_norm, mock_act
        torch.cuda.empty_cache()
        results["benchmark_ok"] = True

    except Exception as bench_err:
        warn(f"Computational stress benchmarking task sequence aborted unexpectedly: {bench_err}")

    return results

def check_config() -> dict:
    head("Local Project Settings Verification Map")
    results = {}

    try:
        from config.settings import (
            DEVICE, BASE_DIR, MODELS_CACHE_DIR,
            LLM_CONFIG, DIFFUSION_CONFIG, FACE_CONFIG,
            VIDEO_CONFIG
        )

        row("Base Project Directory Path", str(BASE_DIR))
        row("Models Cache Allocation Target", str(MODELS_CACHE_DIR))
        row("Assigned Pipeline Target Device", DEVICE, C.GREEN if DEVICE == "cuda" else C.YELLOW)
        row("Selected Language Model (LLM)", LLM_CONFIG["model_name"])
        row("Stable Diffusion Base Weights Model", DIFFUSION_CONFIG["model_name"])
        row("InsightFace Network Weights Model", FACE_CONFIG["model_name"])
        row("Target Export Video Resolution Frame Size", f"{VIDEO_CONFIG['resolution'][0]} x {VIDEO_CONFIG['resolution'][1]}")
        row("Video Target Frame Rate Metric (FPS)", f"{VIDEO_CONFIG['fps']} frames per second")

        env_file_link = BASE_DIR / ".env"
        if env_file_link.exists():
            ok("Configuration interface file (.env) exists on system.")
            results["env_exists"] = True
        else:
            warn("Target infrastructure configuration settings track file (.env) is missing.")
            info("Execution Tip: Instantiate configuration maps using: cp .env.example .env")
            results["env_exists"] = False

        ok("Local pipeline architecture configurations compiled successfully.")
        results["config_ok"] = True

    except Exception as config_err:
        err(f"System initialization interrupted while parsing internal setup criteria: {config_err}")
        results["config_ok"] = False

    return results

def print_final_report(all_results: dict) -> bool:
    head("System Diagnostic Evaluation Abstract Summary")

    verifications = {
        "Python Runtime Compatibility (≥ 3.10)" : all_results.get("system", {}).get("python_ok", False),
        "PyTorch Model Engine Package Stack"     : all_results.get("pytorch", {}).get("torch_version") is not None,
        "CUDA Compute Layer Engine Verification" : all_results.get("pytorch", {}).get("cuda_available", False),
        "FFmpeg Binary Integration Driver Modules" : all_results.get("ffmpeg", {}).get("ffmpeg", False),
        "Core Multi-Model Python Packages"      : all(
            all_results.get("packages", {}).get(p, False)
            for p in ["insightface", "opencv-python", "Pillow", "diffusers", "transformers", "accelerate", "numpy"]
        ),
        "Local Engine Project Settings Registry" : all_results.get("config", {}).get("config_ok", False),
    }

    operational_readiness = True
    for label, validation_state in verifications.items():
        if validation_state:
            ok(label)
        else:
            err(label)
            operational_readiness = False

    print()

    gpu_vram_metric = all_results.get("pytorch", {}).get("gpu_0_vram_gb", 0.0)
    if gpu_vram_metric > 0:
        if gpu_vram_metric >= 24:
            ok(f"Hardware Compute Engine Capacity VRAM Allocation: {gpu_vram_metric:.1f} GB — Top Tier Framework Standard Compatibility.")
        elif gpu_vram_metric >= 16:
            ok(f"Hardware Compute Engine Capacity VRAM Allocation: {gpu_vram_metric:.1f} GB — High Configuration Structural Verification Safe Baseline.")
        elif gpu_vram_metric >= 8:
            warn(f"Hardware Compute Engine Capacity VRAM Allocation: {gpu_vram_metric:.1f} GB — Minimal Operations Map; processing speeds may drop under heavy task cycles.")
        else:
            err(f"Hardware Compute Engine Capacity VRAM Allocation: {gpu_vram_metric:.1f} GB — Resource constraints do not pass standard threshold specs.")
            operational_readiness = False

    print(f"\n{'─'*54}")

    if operational_readiness:
        print(f"\n  {C.BOLD}{C.GREEN}System state parameters verified ready for FalconAI Kids orchestration!{C.RESET}")
        print(f"\n  {C.DIM}Next Step Workflow Command:{C.RESET}")
        print(f"  {C.CYAN}python scripts/download_models.py{C.RESET}")
        print(f"  {C.DIM}Then Execute Principal Core Pipeline Run:{C.RESET}")
        print(f"  {C.CYAN}python main.py --photo input_portrait.jpg --name 'Alex' --birthday 2018-05-10{C.RESET}\n")
    else:
        print(f"\n  {C.BOLD}{C.YELLOW}System diagnostic checks isolated configuration parameters needing resolution attention.{C.RESET}")
        print(f"  {C.DIM}Review error diagnostic outputs listed sequentially up-stack to address environment discrepancies.{C.RESET}\n")

    return operational_readiness

def main():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗
║         FalconAI Kids — System & GPU Tester          ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")

    parser = argparse.ArgumentParser(description="Validates local compute engine criteria frameworks for FalconAI pipeline operations.")
    parser.add_argument("--quick", action="store_true", help="Bypasses intense compute profiling steps; logs structural parameters quickly.")
    parser.add_argument("--bench", action="store_true", help="Forces explicit extended micro-benchmarking calculation cycles.")
    args = parser.parse_args()

    runtime_analytics = {
        "system":   check_system(),
        "pytorch":  check_pytorch(),
        "packages": check_packages(),
        "ffmpeg":   check_ffmpeg(),
        "disk":     check_disk(),
        "config":   check_config()
    }

    if args.bench or not args.quick:
        runtime_analytics["benchmark"] = run_benchmark()

    system_pass_status = print_final_report(runtime_analytics)

    sys.exit(0 if system_pass_status else 1)

if __name__ == "__main__":
    main()
