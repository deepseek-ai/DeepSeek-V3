from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch


_EXTENSION = None
_LOAD_ERROR: Optional[Exception] = None


def _load_extension():
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None or _LOAD_ERROR is not None:
        return _EXTENSION

    try:
        from torch.utils.cpp_extension import load

        base_dir = Path(__file__).resolve().parent
        source = base_dir / "native" / "gate_scan_mps.mm"
        build_dir = base_dir / "native" / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        venv_bin = str(Path(sys.prefix) / "bin")
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        _EXTENSION = load(
            name="gate_scan_mps_ext",
            sources=[str(source)],
            extra_cflags=["-std=c++17", "-fobjc-arc"],
            extra_ldflags=[
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "MetalPerformanceShadersGraph",
            ],
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - exercised only on toolchain failures
        _LOAD_ERROR = exc
        _EXTENSION = None
    return _EXTENSION


def mps_gate_scan_available() -> bool:
    if not torch.backends.mps.is_available():
        return False
    return _load_extension() is not None


def mps_gate_scan_error() -> Optional[Exception]:
    _load_extension()
    return _LOAD_ERROR


def run_mps_gate_scan(
    x: torch.Tensor,
    weights: torch.Tensor,
    context: torch.Tensor,
    membrane: torch.Tensor,
    trace: torch.Tensor,
    threshold: torch.Tensor,
    leak: float,
    context_leak: float,
    context_scale: float,
    threshold_lr: float,
    target_rate: float,
    trace_decay: float,
    active_bonus: float,
    min_active: int,
    max_active: int,
) -> Optional[torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.gate_scan_mps(
        x,
        weights,
        context,
        membrane,
        trace,
        threshold,
        float(leak),
        float(context_leak),
        float(context_scale),
        float(threshold_lr),
        float(target_rate),
        float(trace_decay),
        float(active_bonus),
        int(min_active),
        int(max_active),
    )
