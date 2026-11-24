#!/usr/bin/env python3
"""Report whether the local environment is ready for official PaiNN training."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


PACKAGES = ("torch", "schnetpack", "ase", "h5py", "numpy")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    print(f"python      : {sys.version.split()[0]}")
    print(f"platform    : {platform.platform()}")

    missing: list[str] = []
    for package in PACKAGES:
        version = package_version(package)
        print(f"{package:<12}: {version or 'MISSING'}")
        if version is None:
            missing.append(package)

    torch_version = package_version("torch")
    if torch_version is not None:
        import torch

        print(f"cuda        : {torch.cuda.is_available()}")
        print(f"mps         : {torch.backends.mps.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device : {torch.cuda.get_device_name(0)}")

    if missing:
        print("\nEnvironment is not ready. Missing:", ", ".join(missing))
        print("Install with: python -m pip install -r requirements-official-painn.txt")
        return 1

    print("\nEnvironment is ready for the official PaiNN Learning Test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

