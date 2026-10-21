"""CIFAR-10 policy presets and historical training settings."""

import torch

DIFFUSION_CONFIG = {"img_size": (3, 32, 32), "img_size_condition": (3, 32, 32), "eval_steps": 250}
POLICY_CONFIG = {
    "input_channels": 3,
    "hidden_dim": 64,
    "feature_dim": 32,
    "num_classes": None,
    "class_embedding_dim": 16,
    "use_convlstm": False,
}
MAMBA_CONFIG = {
    "input_channels": 3,
    "d_state": 32,
    "d_expand": 2,
    "d_conv": 4,
    "hidden_dim": 96,
    "feature_dim": 48,
}
TRAINING_CONFIG = {
    "num_episodes": 1000,
    "batch_size": 32,
    "learning_rate": 5e-05,
    "entropy_weight": 0.02,
    "max_consecutive_skips": 1001,
    "warmup_ratio": 0.8,
    "warmup_initial_ratio": 0.4,
    "warmup_final_ratio": 0.0,
    "warmup_anneal_episodes": 800,
    "reward_alpha": 1.5,
    "reward_beta": 15,
    "reward_gamma": 0.07,
    "discount_factor": 1.0,
    "use_curriculum": True,
    "phase1_episodes": 100,
    "phase2_episodes": 400,
    "use_advanced_rewards": True,
    "w_perceptual": 0.0,
    "w_object": 2.5,
    "w_focus": 0.0,
    "w_color": 0.0,
    "w_cost": 0.003,
    "w_quality": 0.0,
    "w_dynamic_range": 0.0,
    "quality_patch_size": 32,
    "w_manifold": 2,
    "manifold_encoder": "resnet18",
    "manifold_bank_path": "data/cifar10_bank_resnet18.pt",
    "manifold_k": 50,
    "img_size": (3, 32, 32),
    "debug_mode": False,
    "debug_save_plots": True,
    "debug_check_gradients": True,
    "debug_validate_images": True,
    "debug_reward_analysis": True,
    "debug_ppo_ratios": True,
    "debug_memory_usage": False,
}
EVAL_CONFIG = {
    "num_samples": 64,
    "batch_size": 32,
    "visualize": True,
    "save_dir": "evaluation_results_cifar10",
    "num_warmup_steps": 200,
    "eval_specific_classes": False,
}
USE_MIXED_PRECISION = True
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
DIFFUSION_POLICY_CONFIG = {
    "dataset": "cifar10",
    "action_horizon": 15,
    "planning_interval": 5,
    "n_obs_steps": 1,
    "num_inference_steps": 5,
    "scheduler_type": "DDPM",
    "max_diffusion_steps": 1000,
    "obs_feature_dim": 32,
    "n_layer": 2,
    "n_head": 2,
    "n_emb": 32,
    "p_drop_emb": 0.0,
    "p_drop_attn": 0.1,
    "use_ema": False,
    "ema_decay": 0.995,
    "causal_attn": False,
    "time_as_cond": True,
    "obs_as_cond": True,
    "n_cond_layers": 1,
    "num_classes": None,
    "class_embedding_dim": 16,
    "learning_rate": 0.0001,
    "weight_decay": 0.001,
    "batch_size": 32,
    "prediction_type": "epsilon",
    "loss_reduction": "mean",
}
ENCODER_CONFIG = {
    "dataset": "cifar10",
    "input_channels": 3,
    "image_size": 32,
    "output_dim": 32,
    "arch_type": "simple_cnn",
    "channels": [16, 32, 64],
    "kernel_sizes": [3, 3, 3],
    "strides": [2, 2, 2],
    "use_batch_norm": True,
    "use_group_norm": False,
    "activation": "GELU",
    "dropout_rate": 0.1,
    "use_adaptive_pool": True,
    "final_pool_size": (1, 1),
    "use_projection": True,
    "projection_layers": [128],
    "use_residual": False,
}
ENCODER_CONFIG_RESNET = {
    **ENCODER_CONFIG,
    "arch_type": "resnet",
    "channels": [16, 32, 64],
    "use_batch_norm": True,
    "activation": "GELU",
    "dropout_rate": 0.1,
}
PHASE2_TRAINING_CONFIG = {
    **{k: v for k, v in TRAINING_CONFIG.items() if not k.startswith("w_")},
    "w_perceptual": 0.0,
    "w_object": 2,
    "w_focus": 0.2,
    "w_color": 0.1,
    "w_cost": 0.05,
    "w_manifold": 0.7,
    "manifold_encoder": "resnet18",
    "manifold_bank_path": "data/cifar10_bank_resnet18.pt",
    "manifold_k": 50,
}
"CURRICULUM LEARNING INSTRUCTIONS:"


def build_diffusion_policy(**overrides):
    from ..policy.diffusion_policy import DiffusionPolicy

    config = DIFFUSION_POLICY_CONFIG.copy()
    config.update(overrides)
    config.setdefault("encoder_config", ENCODER_CONFIG.copy())
    return DiffusionPolicy(**config)


def build_cifar10_encoder(config_type="default", **overrides):
    from ..encoder.custom_encoder import CustomEncoder

    if config_type not in ("default", "resnet"):
        raise ValueError("Unsupported encoder configuration: " + str(config_type))
    config = (ENCODER_CONFIG_RESNET if config_type == "resnet" else ENCODER_CONFIG).copy()
    config.update(overrides)
    return CustomEncoder(config, dataset="cifar10")


def build_complete_system(**policy_overrides):
    dimension = policy_overrides.get("obs_feature_dim", DIFFUSION_POLICY_CONFIG["obs_feature_dim"])
    encoder = build_cifar10_encoder(output_dim=dimension)
    policy = build_diffusion_policy(**policy_overrides)
    policy.visual_encoder = encoder
    return (encoder, policy)
