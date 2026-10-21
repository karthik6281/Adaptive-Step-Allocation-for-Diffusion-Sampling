"""Per-sample rewards. Optional image-quality heuristics are disabled by default."""

from __future__ import annotations

import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from utils.reproducibility import file_sha256, model_sha256, require_finite, require_int

COMPONENTS = (
    "perceptual",
    "object",
    "focus",
    "color",
    "manifold",
    "quality",
    "dynamic_range",
    "cost",
)


def validate_reward_schedule(schedule, num_episodes):
    if not isinstance(schedule, list):
        raise ValueError("reward_schedule must be a list of stages.")
    previous = -1
    for stage in schedule:
        if not isinstance(stage, dict) or set(stage) != {"episode", "weights"}:
            raise ValueError("Each reward stage must contain episode and weights.")
        episode = require_int(
            stage["episode"], "reward stage episode", minimum=0, maximum=num_episodes - 1
        )
        if episode <= previous:
            raise ValueError("Reward stages must have strictly increasing episode indices.")
        previous = episode
        if not isinstance(stage["weights"], dict):
            raise ValueError("Scheduled weights must be a mapping.")
        for name, weight in stage["weights"].items():
            if name not in {f"w_{component}" for component in COMPONENTS}:
                raise ValueError(f"Unknown scheduled weight: {name}")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError("Scheduled reward weights must be finite and nonnegative.")


class GPURunningStats:
    """Population statistics, frozen after a fixed calibration sample count."""

    def __init__(self, device="cpu", warmup_samples=20, epsilon=1e-8):
        if warmup_samples < 1 or epsilon <= 0:
            raise ValueError("warmup_samples and epsilon must be positive")
        self.mean = torch.tensor(0.0, device=device)
        self.variance = torch.tensor(1.0, device=device)
        self.count = 0
        self.warmup_samples, self.epsilon = warmup_samples, epsilon

    @torch.no_grad()
    def update_batch(self, values):
        values = require_finite(values.detach().float(), "normalization values").flatten()
        values = values[: max(0, self.warmup_samples - self.count)]
        if values.numel() == 0:
            return
        count = values.numel()
        mean, variance = values.mean(), values.var(unbiased=False)
        if self.count:
            total = self.count + count
            delta = mean - self.mean
            variance = (
                self.variance * self.count
                + variance * count
                + delta.square() * self.count * count / total
            ) / total
            mean = self.mean + delta * count / total
        self.mean, self.variance = mean, variance
        self.count += count

    def normalize_batch(self, values):
        if self.count < self.warmup_samples:
            return values
        return (values - self.mean) / (self.variance + self.epsilon).sqrt()

    def state_dict(self):
        return {"mean": self.mean, "variance": self.variance, "count": self.count}

    def load_state_dict(self, state):
        self.mean = state["mean"].to(self.mean.device)
        self.variance = state["variance"].to(self.variance.device)
        self.count = int(state["count"])


class _ManifoldBank:
    def __init__(self, embeds, device="cpu"):
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 2 or min(embeds.shape) < 1:
            raise ValueError("A manifold bank must be a nonempty [samples, features] tensor.")
        require_finite(embeds, "manifold bank")
        if torch.any(embeds.norm(dim=-1) == 0):
            raise ValueError("Manifold embeddings must have nonzero norm.")
        self.bank = F.normalize(embeds.detach().float().to(device), dim=-1)
        self.num_embeds = self.bank.shape[0]
        self.is_class_conditional = False

    def topk_mean_cosine(self, queries, k=20, target_class=None):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if queries.ndim != 2 or queries.shape[1] != self.bank.shape[1]:
            raise ValueError("Query and bank feature dimensions do not match.")
        require_finite(queries, "manifold queries")
        if torch.any(queries.norm(dim=-1) == 0):
            raise ValueError("Manifold queries must have nonzero norm.")
        normalized = F.normalize(queries.float().to(self.bank.device), dim=-1)
        values = (normalized @ self.bank.T).topk(min(k, self.num_embeds), dim=-1).values
        return values.mean(-1).to(queries.device)


class _ClassConditionalManifoldBank:
    def __init__(self, class_banks, class_names=None, device="cpu"):
        if not class_banks:
            raise ValueError("Class-conditional banks cannot be empty.")
        self.class_banks = {
            int(key): _ManifoldBank(value, device) for key, value in class_banks.items()
        }
        self.is_class_conditional = True

    def topk_mean_cosine(self, queries, k=20, target_class=None):
        if target_class is None:
            raise ValueError("Class-conditional manifold rewards require target labels.")
        labels = torch.as_tensor(target_class, device=queries.device).expand(queries.shape[0])
        result = queries.new_empty(queries.shape[0])
        for label in labels.unique().tolist():
            if label not in self.class_banks:
                raise ValueError(f"No manifold bank for class {label}.")
            mask = labels == label
            result[mask] = self.class_banks[label].topk_mean_cosine(queries[mask], k)
        return result


def _preprocess_for_encoder(images, size=224):
    images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


def load_manifold_encoder(name, device):
    if name in ("resnet18", "resnet50"):
        from torchvision import models

        if name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    elif name in ("dinov2s", "dinov2b"):
        import timm

        variant = "small" if name == "dinov2s" else "base"
        model = timm.create_model(
            f"vit_{variant}_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224
        )
    else:
        raise ValueError(f"Unsupported manifold encoder: {name}")
    return model.to(device).eval().requires_grad_(False)


class AdvancedRewardV3:
    def __init__(
        self,
        device="cpu",
        dataset="cifar10",
        w_perceptual=0.0,
        w_object=0.0,
        w_focus=0.0,
        w_color=0.0,
        w_cost=0.0,
        w_manifold=0.0,
        w_quality=0.0,
        w_dynamic_range=0.0,
        manifold_bank_path=None,
        manifold_encoder="dinov2b",
        manifold_k=25,
        manifold_bank_device=None,
        quality_patch_size=32,
        obj_alpha=0.75,
        obj_power=2.5,
        perceptual_mode="delta",
        use_normalization=False,
        normalization_samples=20,
        perceptual_model=None,
        object_model=None,
        manifold_model=None,
    ):
        self.device = torch.device(device)
        self.dataset = dataset
        weights = (
            w_perceptual,
            w_object,
            w_focus,
            w_color,
            w_manifold,
            w_quality,
            w_dynamic_range,
            w_cost,
        )
        for name, weight in zip(COMPONENTS, weights):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"w_{name} must be finite and nonnegative")
            setattr(self, f"w_{name}", float(weight))
        if perceptual_mode not in ("delta", "absolute"):
            raise ValueError("perceptual_mode must be 'delta' or 'absolute'")
        if not isinstance(manifold_k, int) or manifold_k < 1 or quality_patch_size < 3:
            raise ValueError("manifold_k must be positive; quality_patch_size must be at least 3")
        if not 0 <= obj_alpha <= 1 or not math.isfinite(obj_power) or obj_power <= 0:
            raise ValueError("Invalid object-reward parameters.")
        self.perceptual_mode = perceptual_mode
        self.quality_patch_size = quality_patch_size
        self.manifold_k, self.manifold_encoder = manifold_k, manifold_encoder
        self.manifold_device = torch.device(manifold_bank_device or device)
        self.obj_alpha, self.obj_power = obj_alpha, obj_power
        self.use_normalization = use_normalization
        self.stats = {
            name: GPURunningStats(self.device, normalization_samples) for name in COMPONENTS
        }
        self.lpips_model = perceptual_model
        self.feature_extractor = object_model
        self._manifold_enc = manifold_model
        self._manifold_bank = None
        self.configuration = {
            "dataset": dataset,
            "perceptual_mode": perceptual_mode,
            "weights": dict(zip(COMPONENTS, weights)),
            "quality_patch_size": quality_patch_size,
            "manifold_encoder": manifold_encoder,
            "manifold_k": manifold_k,
            "obj_alpha": obj_alpha,
            "obj_power": obj_power,
            "use_normalization": use_normalization,
            "normalization_samples": normalization_samples,
            "manifold_bank_sha256": file_sha256(manifold_bank_path)
            if w_manifold and manifold_bank_path
            else None,
        }
        if w_perceptual and self.lpips_model is None:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex", verbose=False)
        if w_object and self.feature_extractor is None:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            self.feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for model in (self.lpips_model, self.feature_extractor):
            if model is not None:
                model.to(self.device).eval().requires_grad_(False)
        if w_manifold:
            if not manifold_bank_path:
                raise ValueError("w_manifold requires a manifold_bank_path.")
            payload = torch.load(manifold_bank_path, map_location="cpu", weights_only=True)
            if (
                isinstance(payload, dict)
                and payload.get("encoder", manifold_encoder) != manifold_encoder
            ):
                raise ValueError("The requested encoder does not match the manifold bank.")
            if isinstance(payload, dict) and payload.get("is_class_conditional", False):
                self._manifold_bank = _ClassConditionalManifoldBank(
                    payload["class_banks"], device=self.manifold_device
                )
            else:
                embeds = payload["embeds"] if isinstance(payload, dict) else payload
                self._manifold_bank = _ManifoldBank(embeds, self.manifold_device)
            if self._manifold_enc is None:
                self._manifold_enc = load_manifold_encoder(manifold_encoder, self.manifold_device)
            self._manifold_enc.to(self.manifold_device).eval().requires_grad_(False)
            if isinstance(payload, dict) and "encoder_state_sha256" in payload:
                if payload["encoder_state_sha256"] != model_sha256(self._manifold_enc):
                    raise ValueError("The manifold bank was built with different encoder weights.")
            else:
                warnings.warn(
                    "Legacy bank has no encoder checksum; rebuild it to verify feature compatibility.",
                    UserWarning,
                    stacklevel=2,
                )
        self.configuration["model_sha256"] = {
            name: model_sha256(model)
            for name, model in (
                ("perceptual", self.lpips_model),
                ("object", self.feature_extractor),
                ("manifold", self._manifold_enc),
            )
            if model is not None
        }

    @property
    def needs_reference(self):
        return self.w_perceptual != 0

    @property
    def needs_images(self):
        return any(getattr(self, f"w_{name}") != 0 for name in COMPONENTS if name != "cost")

    @staticmethod
    def _rgb(images):
        if images.shape[1] == 1:
            return images.expand(-1, 3, -1, -1)
        if images.shape[1] != 3:
            raise ValueError("Image rewards require decoded grayscale or RGB images.")
        return images

    @staticmethod
    def _gray(images):
        if images.shape[1] == 1:
            return images
        weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (AdvancedRewardV3._rgb(images) * weights).sum(1, keepdim=True)

    def _compute_perceptual_reward(self, generated, reference):
        if reference is None or reference.shape != generated.shape:
            raise ValueError("Perceptual reward requires a shape-matched, paired reference.")
        if self.lpips_model is None:
            raise RuntimeError("The LPIPS model is not initialized.")
        value = self.lpips_model(
            self._rgb(generated).clamp(-1, 1), self._rgb(reference).clamp(-1, 1)
        )
        return -value.reshape(generated.shape[0])

    def _compute_object_reward(self, logits):
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("Object logits must have shape [batch, classes>=2].")
        probabilities = logits.float().softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        confidence = 1 - entropy / math.log(logits.shape[1])
        combined = self.obj_alpha * probabilities.max(-1).values + (1 - self.obj_alpha) * confidence
        return combined.clamp(0, 1).pow(self.obj_power)

    def _compute_focus_reward(self, images):
        gray = self._gray(images)
        batch, _, height, width = gray.shape
        kernel = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)
        padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        saliency = (
            (
                F.conv2d(padded, kernel).square()
                + F.conv2d(padded, kernel.transpose(-1, -2)).square()
            )
            .sqrt()
            .squeeze(1)
        )
        threshold = saliency.flatten(1).sort(descending=True).values[:, int(0.1 * height * width)]
        top = torch.sigmoid(5 * (saliency - threshold[:, None, None]))
        y = torch.arange(height, device=images.device).view(1, height, 1) - height // 2
        x = torch.arange(width, device=images.device).view(1, 1, width) - width // 2
        distance = (x.square() + y.square() + 1e-8).sqrt()
        distance = distance / (math.hypot(height // 2, width // 2) + 1e-8)
        return (top * torch.sigmoid(3 * (0.7 - distance))).reshape(batch, -1).mean(1)

    @staticmethod
    def _compute_color_reward(images):
        horizontal = (images[..., 1:] - images[..., :-1]).abs().sum((1, 2, 3))
        vertical = (images[:, :, 1:] - images[:, :, :-1]).abs().sum((1, 2, 3))
        return -(horizontal + vertical)

    @staticmethod
    def _compute_dynamic_range_reward(images):
        return images.flatten(1).std(1, unbiased=False)

    def _embed_norm(self, images):
        if self._manifold_enc is None:
            raise RuntimeError("The manifold encoder is not initialized.")
        inputs = _preprocess_for_encoder((self._rgb(images).clamp(-1, 1) + 1) / 2)
        embeddings = self._manifold_enc(inputs.to(self.manifold_device)).float()
        return F.normalize(require_finite(embeddings, "manifold features"), dim=-1)

    def _compute_manifold_delta(self, before, after, k=None, target_class=None):
        if self._manifold_bank is None:
            raise RuntimeError("The manifold bank is not initialized.")
        first, second = self._embed_norm(torch.cat((before, after))).chunk(2)
        k = self.manifold_k if k is None else k
        score_before = self._manifold_bank.topk_mean_cosine(first, k, target_class)
        score_after = self._manifold_bank.topk_mean_cosine(second, k, target_class)
        return (score_after - score_before).to(self.device)

    def compute_sparse_quality(
        self, images, timestep_ratio=None, patch_size=None, return_raw=False
    ):
        size = min(patch_size or self.quality_patch_size, *images.shape[-2:])
        if size < 3:
            raise ValueError("Quality rewards require images and patches at least 3x3.")
        gray = self._gray((images.clamp(-1, 1) + 1) / 2)
        ratio = None
        if timestep_ratio is not None:
            ratio = torch.as_tensor(timestep_ratio, device=images.device, dtype=torch.float32)
            ratio = ratio.expand(images.shape[0])
            if not torch.all((ratio >= 0) & (ratio <= 1)):
                raise ValueError("timestep_ratio must be in [0, 1]")
        filtered = F.avg_pool2d(gray, 3, stride=1, padding=1)
        if ratio is not None:
            filtered = torch.where(
                (ratio > 0.3)[:, None, None, None],
                F.avg_pool2d(gray, 5, stride=1, padding=2),
                filtered,
            )
        patches = filtered.unfold(2, size, size).unfold(3, size, size)
        patches = patches.reshape(images.shape[0], -1, size, size).float()
        singular = torch.linalg.svdvals(patches)
        structure = singular[..., 0] / singular.sum(-1).clamp_min(1e-8)
        detail = torch.log1p((patches[..., 2:] - patches[..., :-2]).square().mean((-1, -2)))
        texture = patches.diff(dim=-2).std((-1, -2), unbiased=False)
        if ratio is None:
            scores = (structure * detail * texture).mean(1)
            scale = 0.1
        else:
            scores = (
                (0.6 + 0.3 * ratio[:, None]) * structure
                + (0.3 - 0.2 * ratio[:, None]) * detail
                + (0.1 - 0.05 * ratio[:, None]) * texture
            ).mean(1)
            scale = 0.05 + 0.15 * ratio
        return require_finite(100 * scores if return_raw else (scores / scale).sigmoid(), "quality")

    @torch.no_grad()
    def compute_comprehensive_rewards(
        self,
        generated_image,
        target_image,
        actions,
        x_pre=None,
        force_recache=False,
        target_class=None,
        timestep_ratio=None,
    ):
        if generated_image.ndim != 4 or generated_image.shape[0] < 1:
            raise ValueError(
                "generated_image must be a nonempty [batch, channels, height, width] tensor"
            )
        batch = generated_image.shape[0]
        if actions.shape != (batch,) or not torch.all((actions == 0) | (actions == 1)):
            raise ValueError("actions must be binary with shape [batch]")
        require_finite(generated_image, "reward images")
        if x_pre is not None and x_pre.shape != generated_image.shape:
            raise ValueError("Pre-action and post-action image shapes must match.")
        raw = {name: generated_image.new_zeros(batch) for name in COMPONENTS}
        if self.w_perceptual:
            raw["perceptual"] = self._compute_perceptual_reward(generated_image, target_image)
            if self.perceptual_mode == "delta":
                if x_pre is None:
                    raise ValueError("Perceptual delta requires the pre-action image.")
                raw["perceptual"] -= self._compute_perceptual_reward(x_pre, target_image)
        if self.w_object:
            inputs = _preprocess_for_encoder((self._rgb(generated_image).clamp(-1, 1) + 1) / 2)
            raw["object"] = self._compute_object_reward(self.feature_extractor(inputs))
        if self.w_focus:
            raw["focus"] = self._compute_focus_reward(generated_image)
        if self.w_color:
            raw["color"] = self._compute_color_reward(generated_image)
        if self.w_manifold:
            if x_pre is None:
                raise ValueError("Manifold delta requires the pre-action image.")
            raw["manifold"] = self._compute_manifold_delta(
                x_pre, generated_image, target_class=target_class
            )
        if self.w_quality:
            after = self.compute_sparse_quality(
                generated_image, timestep_ratio, return_raw=x_pre is not None
            )
            before = (
                0
                if x_pre is None
                else self.compute_sparse_quality(x_pre, timestep_ratio, return_raw=True)
            )
            raw["quality"] = after - before
        if self.w_dynamic_range:
            raw["dynamic_range"] = self._compute_dynamic_range_reward(generated_image)
        raw["cost"] = 1 - actions.float()
        results = {}
        deltas = {"manifold", "quality"}
        if self.perceptual_mode == "delta":
            deltas.add("perceptual")
        for name, value in raw.items():
            require_finite(value, f"{name} reward")
            results[f"raw_{name}"] = value
            if (
                self.use_normalization
                and name not in deltas | {"cost"}
                and getattr(self, f"w_{name}")
            ):
                self.stats[name].update_batch(value)
                value = self.stats[name].normalize_batch(value)
            results[name] = value
        results["total"] = sum(
            (-1 if name == "cost" else 1) * getattr(self, f"w_{name}") * results[name]
            for name in COMPONENTS
        )
        require_finite(results["total"], "total reward")
        return results

    def compute_rewards(self, x_t, x_t_denoised, noise_t, noise_t_prev, actions, x_ref, **kwargs):
        return self.compute_comprehensive_rewards(x_t_denoised, x_ref, actions, x_pre=x_t, **kwargs)

    def state_dict(self):
        return {name: stats.state_dict() for name, stats in self.stats.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.stats):
            raise ValueError("Checkpoint reward components do not match.")
        for name, value in state.items():
            self.stats[name].load_state_dict(value)
