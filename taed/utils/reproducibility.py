"""Seeds, portable checkpoint state, and run provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def require_int(value, name, minimum=1, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bounds = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be an integer {bounds}.")
    return value


def seed_everything(seed: int, deterministic: bool = True) -> None:
    require_int(seed, "seed", minimum=0, maximum=2**32 - 1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state() -> dict:
    kind, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            kind,
            torch.from_numpy(keys.astype(np.int64)),
            position,
            has_gauss,
            cached_gaussian,
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    kind, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state(
        (kind, keys.cpu().numpy().astype(np.uint32), position, has_gauss, cached_gaussian)
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Exact resume requires the same number of CUDA devices.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in ("adapter_compvis.py", "run_rl.py", "run_eval.py")]
    for directory in ("models", "dasp", "training", "config", "utils", "tools"):
        files.extend(sorted((root / directory).rglob("*.py")))
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "diffusers",
                "lpips",
                "timm",
                "pillow",
                "matplotlib",
            )
        },
        "source_sha256": {str(path.relative_to(root)): file_sha256(path) for path in files},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def save_checkpoint(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or infinity.")
    return value
