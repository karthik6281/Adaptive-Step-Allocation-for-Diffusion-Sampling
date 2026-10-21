"""Build a checksum-recorded feature bank from a local, licensed image dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

from training.advanced_reward import _preprocess_for_encoder, load_manifold_encoder
from utils.reproducibility import (
    file_sha256,
    model_sha256,
    require_finite,
    save_checkpoint,
    seed_everything,
)


@torch.no_grad()
def build_bank(
    image_dir,
    encoder_name="dinov2b",
    device="cpu",
    batch_size=32,
    class_folders=False,
    encoder=None,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(image_dir).resolve()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    if not files:
        raise ValueError(f"No supported images found in {root}.")
    labels = None
    if class_folders:
        labels = []
        for path in files:
            relative = path.relative_to(root)
            if len(relative.parts) < 2 or not relative.parts[0].isdecimal():
                raise ValueError(
                    "Class folders must be nonnegative integer labels, such as 0/ and 1/."
                )
            labels.append(int(relative.parts[0]))
    model = encoder if encoder is not None else load_manifold_encoder(encoder_name, device)
    model = model.to(device).eval().requires_grad_(False)
    batches = []
    for first in range(0, len(files), batch_size):
        images = []
        for path in files[first : first + batch_size]:
            with Image.open(path) as image:
                tensor = to_tensor(image.convert("RGB"))[None]
            images.append(_preprocess_for_encoder(tensor))
        features = require_finite(model(torch.cat(images).to(device)).float(), "bank features")
        if features.ndim != 2 or torch.any(features.norm(dim=-1) == 0):
            raise ValueError("The encoder must return nonzero [batch, features] vectors.")
        batches.append(F.normalize(features, dim=-1).cpu())
    embeddings = torch.cat(batches)
    result = {
        "encoder": encoder_name,
        "encoder_state_sha256": model_sha256(model),
        "preprocessing": "RGB; bilinear 224x224, align_corners=False; ImageNet mean/std",
        "images_sha256": {str(path.relative_to(root)): file_sha256(path) for path in files},
        "is_class_conditional": class_folders,
    }
    if labels is None:
        result["embeds"] = embeddings
    else:
        labels = torch.tensor(labels)
        result["class_banks"] = {
            int(label): embeddings[labels == label] for label in labels.unique()
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--encoder", choices=("resnet18", "resnet50", "dinov2s", "dinov2b"), default="dinov2b"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-folders", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}.")
    seed_everything(42)
    bank = build_bank(args.images, args.encoder, args.device, args.batch_size, args.class_folders)
    save_checkpoint(bank, args.output)
    print(f"Saved {len(bank['images_sha256'])} embeddings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
