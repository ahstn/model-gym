"""Runtime environment report: versions, device, and GPU memory.

Deliberately import-guarded so ``model-gym info`` works before the ``train`` or
``export`` extras are installed.
"""

from __future__ import annotations

import platform
import shutil
import sys

from importlib import metadata
from pathlib import Path
from typing import Any

OPTIONAL_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "scikit-learn",
    "optimum",
    "optimum-onnx",
    "onnx",
    "onnxruntime",
)


def package_versions() -> dict[str, str]:
    """Installed versions of the packages this project depends on."""
    versions: dict[str, str] = {}
    for name in OPTIONAL_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def gpu_report() -> dict[str, Any]:
    """CUDA/MPS availability, device names, and total memory in GiB."""
    report: dict[str, Any] = {"cuda": False, "mps": False, "devices": [], "cuda_version": "", "torch": ""}
    try:
        import torch
    except ImportError:
        return report
    report["torch"] = torch.__version__
    report["cuda_version"] = torch.version.cuda or ""
    report["cuda"] = bool(torch.cuda.is_available())
    report["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if report["cuda"]:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": f"{properties.major}.{properties.minor}",
                    "memory_gib": round(properties.total_memory / 1024**3, 2),
                }
            )
    return report


def environment_info() -> dict[str, Any]:
    """Full report: platform, package versions, device, and disk headroom."""
    usage = shutil.disk_usage(Path.cwd())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "gpu": gpu_report(),
        "disk_free_gib": round(usage.free / 1024**3, 2),
    }


def format_info(info: dict[str, Any] | None = None) -> str:
    """Human-readable rendering of :func:`environment_info`."""
    info = info or environment_info()
    lines = [
        f"python     {info['python']}  ({info['machine']})",
        f"platform   {info['platform']}",
        f"disk free  {info['disk_free_gib']} GiB",
        "",
        "packages:",
    ]
    packages = info["packages"]
    for name in OPTIONAL_PACKAGES:
        lines.append(f"  {packages.get(name, '-'):>10}  {name}")
    gpu = info["gpu"]
    lines.extend(
        [
            "",
            "compute:",
            f"  torch            {gpu['torch'] or 'not installed'}",
            f"  cuda runtime     {gpu['cuda_version'] or '-'}",
            f"  cuda available   {gpu['cuda']}",
            f"  mps available    {gpu['mps']}",
        ]
    )
    for device in gpu["devices"]:
        capability = device["capability"].replace(".", "")
        lines.append(f"  gpu {device['index']}            {device['name']} sm_{capability} {device['memory_gib']} GiB")
    if gpu["cuda"] and gpu["devices"]:
        capability = gpu["devices"][0]["capability"]
        if capability == "8.6":
            lines.append("  note             sm_86 (RTX 3090): bf16 and flash-attn 2 ok, flash-attn 3 is Hopper only")
        else:
            short = capability.replace(".", "")
            lines.append(f"  note             sm_{short}: check bf16 and flash-attn support before enabling either")
    return "\n".join(lines)
