#!/usr/bin/env python3
"""Verify the pinned DeepCA/ImageCAS environment without training a model."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


EXPECTED_VERSIONS = {
    "imageio": "2.34.0",
    "lazy_loader": "0.3",
    "networkx": "3.2.1",
    "numpy": "1.23.5",
    "packaging": "24.0",
    "Pillow": "10.2.0",
    "PyWavelets": "1.5.0",
    "PyYAML": "6.0.1",
    "scipy": "1.10.1",
    "scikit-image": "0.21.0",
    "tifffile": "2023.12.9",
    "torch": "2.1.1",
    "torchvision": "0.16.1",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for the tiny construction check (default: cuda:0).",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Also run a no-gradient 32^3 generator forward pass.",
    )
    return parser.parse_args(argv)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def _print_nvidia_smi() -> bool:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"nvidia-smi query: FAILED ({error})")
        return False
    lines = completed.stdout.strip().splitlines()
    print("nvidia-smi GPUs:")
    for line in lines or ["<no devices returned>"]:
        print(f"  {line}")
    return bool(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    failures: list[str] = []

    print(f"repository: {REPOSITORY_ROOT}")
    print(f"python executable: {sys.executable}")
    print(f"python version: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    if sys.version_info[:2] != (3, 9):
        failures.append(
            f"expected Python 3.9, found {platform.python_version()}"
        )

    print("package versions:")
    for distribution, expected in EXPECTED_VERSIONS.items():
        installed = _distribution_version(distribution)
        print(f"  {distribution}: {installed} (expected {expected})")
        if _base_version(installed) != expected:
            failures.append(
                f"expected {distribution} {expected}, found {installed}"
            )

    try:
        import numpy as np
        import scipy
        import skimage
        import torch
        import torchvision
        import yaml

        from deepca.metrics import skeletonization_metadata
        from deepca.modeling import architecture_metadata
        from train_models.networks.discriminator import Discriminator
        from train_models.networks.generator import Generator
    except Exception as error:
        print(f"required import check: FAILED ({type(error).__name__}: {error})")
        return 1

    print("required import check: OK")
    print(f"numpy import version: {np.__version__}")
    print(f"scipy import version: {scipy.__version__}")
    print(f"PyYAML import version: {yaml.__version__}")
    print(f"scikit-image import version: {skimage.__version__}")
    print(f"torch import version: {torch.__version__}")
    print(f"torchvision import version: {torchvision.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    smi_ok = _print_nvidia_smi()
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"torch GPU {index}: {properties.name}; "
            f"capability={properties.major}.{properties.minor}; "
            f"memory={properties.total_memory / 2**30:.2f} GiB"
        )

    if _base_version(str(torch.__version__)) != "2.1.1":
        failures.append(f"unexpected imported torch version {torch.__version__}")
    if str(torch.version.cuda) != "12.1":
        failures.append(
            f"expected the cu121 PyTorch runtime, found CUDA {torch.version.cuda}"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            failures.append(
                f"requested {device}, but torch.cuda.is_available() is False"
            )
        if not smi_ok:
            failures.append("nvidia-smi did not return a GPU")
    elif device.type != "cpu":
        failures.append(f"unsupported verification device {device}")

    if failures:
        print("environment verification: FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    generator = Generator(num_filters=8, volume_size=32).to(device).eval()
    critic = Discriminator(device, channels=2).to(device).eval()
    metadata = architecture_metadata(generator, critic)
    print(
        "tiny model construction: OK "
        f"(volume={metadata['volume_size']}, "
        f"base_filters={metadata['base_filters']}, "
        f"critic_channels={metadata['discriminator_input_channels']})"
    )
    print(f"skeletonization: {skeletonization_metadata()}")

    if args.forward:
        sample = torch.zeros((1, 1, 32, 32, 32), device=device)
        with torch.no_grad():
            output = generator(sample)
        expected_shape = (1, 1, 32, 32, 32)
        if tuple(output.shape) != expected_shape:
            print(
                f"tiny generator forward: FAILED; expected {expected_shape}, "
                f"got {tuple(output.shape)}"
            )
            return 1
        if not bool(torch.isfinite(output).all()):
            print("tiny generator forward: FAILED; output is non-finite")
            return 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(f"tiny generator forward: OK ({tuple(output.shape)})")

    print("environment verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
