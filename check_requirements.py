#!/usr/bin/env python3
"""
DeepSeek-V3 Inference Requirements Checker

Validates whether the current machine meets the hardware and software
requirements for running DeepSeek-V3 inference at various precision levels.

Usage:
    python check_requirements.py [--precision fp8|bf16|int8|int4]

Exit codes:
    0 - All checks passed (or warnings only)
    1 - One or more critical requirements not met
"""

import argparse
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DeepSeek-V3 Hardware Requirements (based on README examples)
# =============================================================================

WEIGHT_SIZES_GB = {
    "fp8": 685,
    "bf16": 1300,
    "int8": 700,
    "int4": 350,
}

VRAM_REQUIREMENTS_GB = {
    "fp8": 640,
    "bf16": 1280,
    "int8": 640,
    "int4": 320,
}

MIN_CUDA_VERSION = "12.1"
MIN_PYTHON_VERSION = (3, 10)

REQUIRED_PACKAGES = [
    ("torch", "2.4.1", False),
    ("triton", "3.0.0", False),
    ("transformers", "4.46.3", False),
    ("safetensors", "0.4.5", False),
]

OPTIONAL_PACKAGES = [
    ("sglang", None, True),
    ("vllm", None, True),
    ("lmdeploy", None, True),
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "WARN", "FAIL", "INFO", "SKIP"
    message: str
    details: List[str] = field(default_factory=list)


class Colors:
    PASS = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    INFO = "\033[94m"
    SKIP = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    """Apply ANSI color codes if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


# =============================================================================
# Check Functions
# =============================================================================

def check_python_version() -> CheckResult:
    current = sys.version_info[:2]
    if current >= MIN_PYTHON_VERSION:
        return CheckResult(
            name="Python Version",
            status="PASS",
            message=f"Python {current[0]}.{current[1]} meets minimum requirement (>=3.10)",
        )
    else:
        return CheckResult(
            name="Python Version",
            status="FAIL",
            message=f"Python {current[0]}.{current[1]} is too old. Minimum required: 3.10",
            details=["Consider using conda, pyenv, or uv to install a newer Python version."],
        )


def check_os() -> CheckResult:
    system = platform.system()
    if system == "Linux":
        distro = ""
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        distro = line.split("=", 1)[1].strip().strip('"')
                        break
        return CheckResult(
            name="Operating System",
            status="PASS",
            message=f"Linux detected{f' ({distro})' if distro else ''}. DeepSeek-V3 inference is fully supported.",
        )
    elif system == "Darwin":
        return CheckResult(
            name="Operating System",
            status="FAIL",
            message="macOS detected. DeepSeek-V3 inference is NOT supported on macOS.",
            details=["Use a Linux machine with NVIDIA GPUs for inference."],
        )
    elif system == "Windows":
        return CheckResult(
            name="Operating System",
            status="FAIL",
            message="Windows detected. DeepSeek-V3 inference is NOT supported on Windows.",
            details=["Use WSL2 or a native Linux machine with NVIDIA GPUs."],
        )
    else:
        return CheckResult(
            name="Operating System",
            status="WARN",
            message=f"{system} detected. Linux is the only officially supported platform.",
        )


def check_cuda() -> Tuple[CheckResult, Optional[dict]]:
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="CUDA / GPUs",
            status="SKIP",
            message="PyTorch not installed. Cannot detect CUDA/GPUs.",
            details=["Install PyTorch first: pip install torch>=2.4.1"],
        ), None

    if not torch.cuda.is_available():
        return CheckResult(
            name="CUDA / GPUs",
            status="FAIL",
            message="CUDA is not available. No NVIDIA GPUs detected.",
            details=[
                "Ensure NVIDIA drivers are installed.",
                "For CPU-only inference, consider using smaller quantized models or API access.",
            ],
        ), None

    cuda_version = torch.version.cuda or "unknown"
    gpu_count = torch.cuda.device_count()

    gpus = []
    total_vram_gb = 0
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        total_vram_gb += vram_gb
        gpus.append({
            "index": i,
            "name": props.name,
            "vram_gb": round(vram_gb, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })

    info = {
        "cuda_version": cuda_version,
        "gpu_count": gpu_count,
        "total_vram_gb": total_vram_gb,
        "gpus": gpus,
    }

    try:
        major, minor = map(int, cuda_version.split(".")[:2])
        min_major, min_minor = map(int, MIN_CUDA_VERSION.split(".")[:2])
        cuda_ok = (major > min_major) or (major == min_major and minor >= min_minor)
    except (ValueError, AttributeError):
        cuda_ok = True

    details = [
        f"CUDA Version: {cuda_version}",
        f"GPU Count: {gpu_count}",
        f"Total VRAM: {total_vram_gb:.1f} GB",
    ]
    for g in gpus:
        details.append(f"  [{g['index']}] {g['name']} - {g['vram_gb']} GB (SM {g['compute_capability']})")

    if not cuda_ok:
        return CheckResult(
            name="CUDA / GPUs",
            status="WARN",
            message=f"CUDA {cuda_version} may be outdated. Recommended: >={MIN_CUDA_VERSION}",
            details=details,
        ), info

    if gpu_count == 0:
        return CheckResult(
            name="CUDA / GPUs",
            status="FAIL",
            message="No GPUs detected despite CUDA being available.",
            details=details,
        ), info

    return CheckResult(
        name="CUDA / GPUs",
        status="PASS",
        message=f"{gpu_count} GPU(s) detected with {total_vram_gb:.1f} GB total VRAM (CUDA {cuda_version})",
        details=details,
    ), info


def check_vram_against_precision(precision: str, total_vram_gb: float) -> CheckResult:
    required = VRAM_REQUIREMENTS_GB.get(precision, 640)

    if total_vram_gb >= required:
        return CheckResult(
            name=f"VRAM for {precision.upper()}",
            status="PASS",
            message=f"{total_vram_gb:.1f} GB VRAM meets requirement (>={required} GB for {precision.upper()})",
        )
    elif total_vram_gb >= required * 0.75:
        return CheckResult(
            name=f"VRAM for {precision.upper()}",
            status="WARN",
            message=f"{total_vram_gb:.1f} GB VRAM is below recommended ({required} GB) but may work with offloading/quantization.",
            details=[
                "Consider using pipeline parallelism or CPU offloading.",
                f"Try a smaller quantization: INT8/INT4 instead of {precision.upper()}.",
            ],
        )
    else:
        return CheckResult(
            name=f"VRAM for {precision.upper()}",
            status="FAIL",
            message=f"{total_vram_gb:.1f} GB VRAM is insufficient for {precision.upper()} (requires ~{required} GB)",
            details=[
                f"Required: ~{required} GB total VRAM across all GPUs",
                "Consider using cloud instances (e.g., 8x H100, 2x H20x8)",
                "Or use a quantized variant (INT8/INT4) which needs less VRAM",
            ],
        )


def check_disk_space(precision: str) -> CheckResult:
    required_gb = WEIGHT_SIZES_GB.get(precision, 685)
    required_with_buffer = required_gb + 50

    paths_to_check = [
        os.getcwd(),
        os.path.expanduser("~"),
        os.path.expanduser("~/.cache"),
        "/tmp",
    ]

    best_free_gb = 0
    best_path = ""

    for path in paths_to_check:
        if os.path.exists(path):
            try:
                stat = shutil.disk_usage(path)
                free_gb = stat.free / (1024 ** 3)
                if free_gb > best_free_gb:
                    best_free_gb = free_gb
                    best_path = path
            except (OSError, PermissionError):
                continue

    details = [
        f"Model weights ({precision.upper()}): ~{required_gb} GB",
        f"Recommended free space: ~{required_with_buffer} GB (including buffer)",
        f"Largest free space found: {best_free_gb:.1f} GB at '{best_path}'",
    ]

    if best_free_gb >= required_with_buffer:
        return CheckResult(
            name="Disk Space",
            status="PASS",
            message=f"{best_free_gb:.1f} GB free at '{best_path}' - sufficient for {precision.upper()} weights",
            details=details,
        )
    elif best_free_gb >= required_gb:
        return CheckResult(
            name="Disk Space",
            status="WARN",
            message=f"{best_free_gb:.1f} GB free at '{best_path}' - tight but may fit {precision.upper()} weights",
            details=details,
        )
    else:
        return CheckResult(
            name="Disk Space",
            status="FAIL",
            message=f"{best_free_gb:.1f} GB free at '{best_path}' - insufficient for {precision.upper()} weights (~{required_gb} GB needed)",
            details=details + ["Free up disk space or mount a larger volume before downloading weights."],
        )


def check_python_packages() -> List[CheckResult]:
    results = []

    all_packages = REQUIRED_PACKAGES + OPTIONAL_PACKAGES

    for pkg_name, min_version, is_optional in all_packages:
        try:
            module = __import__(pkg_name)
            version = None
            for attr in ("__version__", "VERSION", "version"):
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    if callable(version):
                        version = version()
                    break

            if version is None:
                version = "unknown"

            if min_version and version != "unknown":
                def parse_ver(v):
                    return tuple(int(x) for x in str(v).split(".")[:3] if x.isdigit())

                try:
                    current_ver = parse_ver(version)
                    required_ver = parse_ver(min_version)
                    if current_ver >= required_ver:
                        status = "PASS"
                        msg = f"{pkg_name} {version} installed (>={min_version})"
                    else:
                        status = "WARN" if is_optional else "FAIL"
                        msg = f"{pkg_name} {version} is below recommended {min_version}"
                except ValueError:
                    status = "PASS"
                    msg = f"{pkg_name} {version} installed (could not compare with {min_version})"
            else:
                status = "PASS"
                msg = f"{pkg_name} {version} installed"

            results.append(CheckResult(
                name=f"Package: {pkg_name}",
                status=status,
                message=msg,
            ))

        except ImportError:
            status = "SKIP" if is_optional else "FAIL"
            msg = f"{pkg_name} not installed"
            if not is_optional:
                msg += f" (required: >={min_version})"
            results.append(CheckResult(
                name=f"Package: {pkg_name}",
                status=status,
                message=msg,
                details=[f"Install with: pip install {pkg_name}{'>=' + min_version if min_version else ''}"] if not is_optional else [],
            ))

    return results


def check_nccl() -> CheckResult:
    nccl_path = None
    for env_var in ["LD_LIBRARY_PATH", "NCCL_HOME", "NCCL_DIR"]:
        if env_var in os.environ:
            paths = os.environ[env_var].split(":")
            for p in paths:
                candidate = os.path.join(p, "libnccl.so")
                if os.path.exists(candidate) or os.path.exists(os.path.join(p, "lib", "libnccl.so")):
                    nccl_path = p
                    break

    standard_paths = [
        "/usr/lib/x86_64-linux-gnu/libnccl.so",
        "/usr/local/cuda/lib64/libnccl.so",
        "/usr/local/lib/libnccl.so",
    ]
    for p in standard_paths:
        if os.path.exists(p):
            nccl_path = p
            break

    if nccl_path:
        return CheckResult(
            name="NCCL (Multi-node)",
            status="PASS",
            message=f"NCCL library found at {nccl_path}",
            details=["Required for multi-node tensor/pipeline parallelism."],
        )
    else:
        return CheckResult(
            name="NCCL (Multi-node)",
            status="WARN",
            message="NCCL library not found in standard locations.",
            details=[
                "NCCL is required for multi-node inference (e.g., 2x H20x8 nodes).",
                "Single-node inference may still work without it.",
                "Install: apt install libnccl2 libnccl-dev (Ubuntu/Debian)",
            ],
        )


# =============================================================================
# Main Runner
# =============================================================================

def print_banner():
    banner = """
================================================================================
          DeepSeek-V3 Inference Requirements Checker
          Validates hardware & software before downloading 685GB+ weights
================================================================================
"""
    print(colorize(banner, Colors.BOLD))


def print_result(result: CheckResult):
    status_colors = {
        "PASS": Colors.PASS,
        "WARN": Colors.WARN,
        "FAIL": Colors.FAIL,
        "INFO": Colors.INFO,
        "SKIP": Colors.SKIP,
    }

    color = status_colors.get(result.status, Colors.RESET)

    print(f"  {colorize(f'[{result.status}]', color)} {colorize(result.name, Colors.BOLD)}")
    print(f"      {result.message}")

    for detail in result.details:
        print(f"      -> {detail}")
    print()


def print_summary(results: List[CheckResult]):
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0, "SKIP": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print(colorize("-" * 80, Colors.BOLD))
    print(colorize("  SUMMARY", Colors.BOLD))
    print(colorize("-" * 80, Colors.BOLD))

    total = sum(counts.values())
    print(f"  Total checks: {total}")
    pass_count = counts["PASS"]
    warn_count = counts["WARN"]
    fail_count = counts["FAIL"]
    skip_count = counts["SKIP"]
    print(f"    {colorize(f'PASS: {pass_count}', Colors.PASS)}")
    print(f"    {colorize(f'WARN: {warn_count}', Colors.WARN)}")
    print(f"    {colorize(f'FAIL: {fail_count}', Colors.FAIL)}")
    print(f"    {colorize(f'SKIP: {skip_count}', Colors.SKIP)}")
    print()

    if counts["FAIL"] > 0:
        print(colorize("  RESULT: Cannot run DeepSeek-V3 inference. Fix the FAIL items above.", Colors.FAIL))
        return 1
    elif counts["WARN"] > 0:
        print(colorize("  RESULT: May work with caveats. Review WARN items above.", Colors.WARN))
        return 0
    else:
        print(colorize("  RESULT: Ready to run DeepSeek-V3 inference!", Colors.PASS))
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check if your machine can run DeepSeek-V3 inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_requirements.py              # Check with default FP8 precision
  python check_requirements.py --precision bf16   # Check for BF16 inference
  python check_requirements.py --precision int8   # Check for INT8 quantized inference
        """,
    )
    parser.add_argument(
        "--precision",
        choices=["fp8", "bf16", "int8", "int4"],
        default="fp8",
        help="Target inference precision (default: fp8)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    if args.no_color:
        global Colors
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    print_banner()

    print(colorize(f"  Target Precision: {args.precision.upper()}", Colors.BOLD))
    print(colorize(f"  Required Weights: ~{WEIGHT_SIZES_GB[args.precision]} GB", Colors.BOLD))
    print(colorize(f"  Required VRAM:    ~{VRAM_REQUIREMENTS_GB[args.precision]} GB", Colors.BOLD))
    print()

    results: List[CheckResult] = []

    results.append(check_python_version())
    results.append(check_os())

    cuda_result, cuda_info = check_cuda()
    results.append(cuda_result)

    if cuda_info and cuda_info.get("total_vram_gb"):
        results.append(check_vram_against_precision(args.precision, cuda_info["total_vram_gb"]))
    else:
        results.append(CheckResult(
            name=f"VRAM for {args.precision.upper()}",
            status="SKIP",
            message="Skipped - no CUDA GPUs detected.",
        ))

    results.append(check_disk_space(args.precision))
    results.extend(check_python_packages())
    results.append(check_nccl())

    print(colorize("-" * 80, Colors.BOLD))
    print(colorize("  DETAILED RESULTS", Colors.BOLD))
    print(colorize("-" * 80, Colors.BOLD))
    print()

    for result in results:
        print_result(result)

    exit_code = print_summary(results)

    print()
    print(colorize("  Next steps:", Colors.BOLD))
    if args.precision == "fp8":
        print("    1. Download weights: huggingface-cli download deepseek-ai/DeepSeek-V3")
    elif args.precision == "bf16":
        print("    1. Convert FP8 -> BF16: python inference/fp8_cast_bf16.py")
        print("    2. Download weights: huggingface-cli download deepseek-ai/DeepSeek-V3")
    else:
        print(f"    1. Download quantized weights for {args.precision.upper()}")
    print("    2. Choose inference engine: SGLang, vLLM, LMDeploy, or TRT-LLM")
    print("    3. Follow README.md Section 6 for launch instructions")
    print()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
