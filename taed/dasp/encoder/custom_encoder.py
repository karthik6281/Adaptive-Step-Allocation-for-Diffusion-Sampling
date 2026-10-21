"""TAED Visual Encoder with Configuration Support"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..diffusion_transformer.module_attr_mixin import ModuleAttrMixin


class CustomEncoder(ModuleAttrMixin):
    """Configurable visual encoder for TAED"""

    def __init__(self, config: Dict, dataset: str = "cifar10"):
        """Initialize visual encoder from configuration"""
        super().__init__()
        self.config = config
        self.dataset = dataset.lower()
        self.input_channels = config.get("input_channels", self._get_default_channels())
        self.output_dim = config.get("output_dim", 256)
        self.image_size = config.get("image_size", self._get_default_image_size())
        self.arch_type = config.get("arch_type", "simple_cnn")
        self.channels = config.get("channels", self._get_default_channels_list())
        self.kernel_sizes = config.get("kernel_sizes", [3] * len(self.channels))
        self.strides = config.get("strides", [2] * len(self.channels))
        self.use_batch_norm = config.get("use_batch_norm", True)
        self.use_group_norm = config.get("use_group_norm", False)
        self.activation = config.get("activation", "GELU")
        self.dropout_rate = config.get("dropout_rate", 0.1)
        self.use_adaptive_pool = config.get("use_adaptive_pool", True)
        self.final_pool_size = config.get("final_pool_size", (1, 1))
        self.use_projection = config.get("use_projection", True)
        self.projection_layers = config.get("projection_layers", [self.output_dim])
        if not self.channels or not len(self.channels) == len(self.kernel_sizes) == len(
            self.strides
        ):
            raise ValueError(
                "channels, kernel_sizes, and strides must be nonempty and equally sized"
            )
        if (
            min(
                self.input_channels,
                self.output_dim,
                self.image_size,
                *self.channels,
                *self.kernel_sizes,
                *self.strides,
            )
            < 1
        ):
            raise ValueError("Encoder dimensions must be positive.")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.backbone = self._build_backbone()
        self.feature_dim = self._compute_feature_dim()
        if not self.use_projection and self.output_dim != self.feature_dim:
            raise ValueError(
                "Without projection, output_dim must equal the backbone feature dimension."
            )
        self.projection = self._build_projection() if self.use_projection else nn.Identity()

    def _get_default_channels(self) -> int:
        """Get default input channels for dataset"""
        if self.dataset == "mnist":
            return 1
        else:
            return 3

    def _get_default_image_size(self) -> int:
        """Get default image size for dataset"""
        if self.dataset == "mnist":
            return 28
        elif self.dataset == "cifar10":
            return 32
        elif self.dataset in ["cat256", "church256"]:
            return 256
        else:
            return 32

    def _get_default_channels_list(self) -> List[int]:
        """Get default channel progression for dataset"""
        if self.dataset == "mnist":
            return [32, 64, 128, 256]
        elif self.dataset == "cifar10":
            return [64, 128, 256, 512]
        elif self.dataset in ["cat256", "church256"]:
            return [32, 64, 128, 256, 512, 1024]
        else:
            return [64, 128, 256, 512]

    def _get_activation(self) -> nn.Module:
        """Get activation function from config"""
        activation_map = {
            "GELU": nn.GELU(),
            "ReLU": nn.ReLU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(0.1, inplace=True),
            "Mish": nn.Mish(),
            "SiLU": nn.SiLU(),
            "ELU": nn.ELU(inplace=True),
        }
        if self.activation not in activation_map:
            raise ValueError(f"Unknown activation: {self.activation}")
        return activation_map[self.activation]

    def _build_normalization(self, channels: int) -> nn.Module:
        """Build normalization layer"""
        if self.use_group_norm:
            num_groups = min(32, channels // 4) if channels >= 4 else 1
            num_groups = math.gcd(num_groups, channels)
            return nn.GroupNorm(num_groups, channels)
        elif self.use_batch_norm:
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()

    def _build_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_residual: bool = False,
    ) -> nn.Module:
        """Build a single convolution block"""
        layers = []
        padding = kernel_size // 2
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(self._build_normalization(out_channels))
        layers.append(self._get_activation())
        if self.dropout_rate > 0:
            layers.append(nn.Dropout2d(self.dropout_rate))
        conv_block = nn.Sequential(*layers)
        if use_residual and in_channels == out_channels and (stride == 1):
            return ResidualBlock(conv_block)
        else:
            return conv_block

    def _build_resnet_block(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> nn.Module:
        """Build ResNet-style residual block"""
        main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            self._build_normalization(out_channels),
            self._get_activation(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            self._build_normalization(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            skip_conn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                self._build_normalization(out_channels),
            )
        else:
            skip_conn = nn.Identity()
        return ResidualBlock(main_path, skip_conn, self._get_activation())

    def _build_backbone(self) -> nn.Module:
        """Build the main backbone network"""
        layers = []
        in_channels = self.input_channels
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(self.channels, self.kernel_sizes, self.strides)
        ):
            if self.arch_type == "resnet":
                block = self._build_resnet_block(in_channels, out_channels, stride)
            elif self.arch_type == "simple_cnn":
                use_residual = self.config.get("use_residual", False)
                block = self._build_conv_block(
                    in_channels, out_channels, kernel_size, stride, use_residual
                )
            else:
                raise ValueError(f"Unknown architecture type: {self.arch_type}")
            layers.append(block)
            in_channels = out_channels
        if self.use_adaptive_pool:
            layers.append(nn.AdaptiveAvgPool2d(self.final_pool_size))
        else:
            pool_size = max(1, self.image_size // 2 ** len(self.channels))
            layers.append(nn.AvgPool2d(pool_size))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def _compute_feature_dim(self) -> int:
        """Compute feature dimension after backbone"""
        if self.use_adaptive_pool:
            pool_height, pool_width = self.final_pool_size
            return self.channels[-1] * pool_height * pool_width
        else:
            mode = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                dimension = self.backbone(
                    torch.zeros(1, self.input_channels, self.image_size, self.image_size)
                ).shape[1]
            self.backbone.train(mode)
            return dimension

    def _build_projection(self) -> nn.Module:
        """Build projection head to desired output dimension"""
        if not self.projection_layers:
            return nn.Linear(self.feature_dim, self.output_dim)
        layers = []
        in_dim = self.feature_dim
        needs_final_projection = self.projection_layers[-1] != self.output_dim
        for i, out_dim in enumerate(self.projection_layers):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(self.projection_layers) - 1 or needs_final_projection:
                layers.append(self._get_activation())
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
            in_dim = out_dim
        if needs_final_projection:
            layers.append(nn.Linear(self.projection_layers[-1], self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [batch, {self.input_channels}, height, width] observations."
            )
        features = self.backbone(x)
        if self.use_projection:
            features = self.projection(features)
        return features

    def get_output_dim(self) -> int:
        """Return output feature dimension"""
        return self.output_dim


class ResidualBlock(nn.Module):
    """Residual block with optional skip connection"""

    def __init__(
        self,
        main_path: nn.Module,
        skip_conn: Optional[nn.Module] = None,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.main_path = main_path
        self.skip_conn = skip_conn if skip_conn is not None else nn.Identity()
        self.final_activation = final_activation

    def forward(self, x):
        identity = self.skip_conn(x)
        out = self.main_path(x)
        out = out + identity
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out


def create_taed_visual_encoder(
    dataset: str = "cifar10", config: Optional[Dict] = None, **kwargs
) -> CustomEncoder:
    default_configs = {
        "mnist": {
            "input_channels": 1,
            "image_size": 28,
            "output_dim": 128,
            "channels": [32, 64, 128, 256],
            "kernel_sizes": [3, 3, 3, 3],
            "strides": [2, 2, 2, 2],
            "arch_type": "simple_cnn",
            "use_batch_norm": True,
            "activation": "GELU",
            "dropout_rate": 0.1,
        },
        "cifar10": {
            "input_channels": 3,
            "image_size": 32,
            "output_dim": 48,
            "channels": [24, 48, 96],
            "kernel_sizes": [3, 3, 3],
            "strides": [2, 2, 2],
            "arch_type": "simple_cnn",
            "use_batch_norm": True,
            "activation": "GELU",
            "dropout_rate": 0.1,
        },
        "cat256": {
            "input_channels": 3,
            "image_size": 256,
            "output_dim": 512,
            "channels": [32, 64, 128, 256, 512, 1024],
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "strides": [2, 2, 2, 2, 2, 2],
            "arch_type": "resnet",
            "use_batch_norm": True,
            "activation": "GELU",
            "dropout_rate": 0.1,
        },
        "church256": {
            "input_channels": 3,
            "image_size": 256,
            "output_dim": 512,
            "channels": [32, 64, 128, 256, 512, 1024],
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "strides": [2, 2, 2, 2, 2, 2],
            "arch_type": "resnet",
            "use_batch_norm": True,
            "activation": "GELU",
            "dropout_rate": 0.1,
        },
    }
    if config is None and dataset.lower() not in default_configs:
        raise ValueError(f"An explicit encoder configuration is required for {dataset}.")
    final_config = default_configs.get(dataset.lower(), {}).copy()
    if config is not None:
        final_config.update(config)
    final_config.update(kwargs)
    if "channels" in final_config:
        explicit = {**(config or {}), **kwargs}
        final_config["kernel_sizes"] = explicit.get(
            "kernel_sizes", [3] * len(final_config["channels"])
        )
        final_config["strides"] = explicit.get("strides", [2] * len(final_config["channels"]))
    return CustomEncoder(final_config, dataset)


def get_encoder_config(
    dataset: str = "cifar10", arch_type: str = "simple_cnn", output_dim: int = 256, **overrides
) -> Dict:
    """Get encoder configuration with easy parameter overrides"""
    base_configs = {
        "simple_cnn": {
            "arch_type": "simple_cnn",
            "use_batch_norm": True,
            "use_residual": False,
            "activation": "GELU",
            "dropout_rate": 0.1,
        },
        "resnet": {
            "arch_type": "resnet",
            "use_batch_norm": True,
            "activation": "GELU",
            "dropout_rate": 0.1,
        },
    }
    dataset_configs = {
        "mnist": {"input_channels": 1, "image_size": 28, "channels": [32, 64, 128, 256]},
        "cifar10": {"input_channels": 3, "image_size": 32, "channels": [64, 128, 256, 512]},
        "cat256": {
            "input_channels": 3,
            "image_size": 256,
            "channels": [32, 64, 128, 256, 512, 1024],
        },
        "church256": {
            "input_channels": 3,
            "image_size": 256,
            "channels": [32, 64, 128, 256, 512, 1024],
        },
    }
    if arch_type not in base_configs or dataset.lower() not in dataset_configs:
        raise ValueError(f"Unknown encoder architecture or dataset: {arch_type}, {dataset}")
    config = base_configs[arch_type].copy()
    config.update(dataset_configs[dataset.lower()])
    config["output_dim"] = output_dim
    num_layers = len(config["channels"])
    config.setdefault("kernel_sizes", [3] * num_layers)
    config.setdefault("strides", [2] * num_layers)
    config.update(overrides)
    return config


def create_encoder_from_config(config: Dict, dataset: str = "cifar10") -> CustomEncoder:
    return CustomEncoder(config, dataset)
