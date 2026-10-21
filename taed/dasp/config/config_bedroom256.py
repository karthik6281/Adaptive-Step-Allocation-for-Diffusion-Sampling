"""LSUN-bedroom policy presets and historical training settings."""

import os

import torch

_TAED_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_CONFIG = {
    "num_episodes": 400,
    "batch_size": 32,
    "learning_rate": 5e-05,
    "entropy_weight": 0.02,
    "max_consecutive_skips": 1001,
    "warmup_ratio": 0.8,
    "warmup_initial_ratio": 0.4,
    "warmup_final_ratio": 0.0,
    "warmup_anneal_episodes": 160,
    "reward_alpha": 1.5,
    "reward_beta": 15,
    "reward_gamma": 0.07,
    "discount_factor": 1.0,
    "use_curriculum": False,
    "phase1_episodes": 100,
    "phase2_episodes": 400,
    "use_advanced_rewards": True,
    "w_perceptual": 2.0,
    "perceptual_mode": "delta",
    "w_object": 0.0,
    "w_focus": 0.0,
    "w_color": 0.0,
    "w_cost": 0.02,
    "w_quality": 0.0,
    "w_dynamic_range": 0.0,
    "quality_patch_size": 32,
    "w_manifold": 0.0,
    "manifold_encoder": "dinov2b",
    "manifold_bank_path": os.path.join(
        _TAED_ROOT, "data", "manifold_lsun-bedrooms_dinov2_vitb14_224px.pt"
    ),
    "manifold_k": 25,
    "img_size": (3, 64, 64),
    "debug_mode": False,
}
EVAL_CONFIG = {
    "num_samples": 64,
    "batch_size": 32,
    "visualize": True,
    "save_dir": "evaluation_results_bedroom     ",
}
USE_MIXED_PRECISION = True
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIFFUSION_POLICY_CONFIG = {
    "dataset": "bedroom_256",
    "action_horizon": 50,
    "planning_interval": 5,
    "n_obs_steps": 1,
    "num_inference_steps": 10,
    "scheduler_type": "DDPM",
    "max_diffusion_steps": 1000,
    "obs_feature_dim": 32,
    "n_layer": 3,
    "n_head": 4,
    "n_emb": 32,
    "p_drop_emb": 0.0,
    "p_drop_attn": 0.1,
    "use_ema": False,
    "causal_attn": False,
    "time_as_cond": True,
    "obs_as_cond": True,
    "n_cond_layers": 1,
    "num_classes": None,
    "class_embedding_dim": 16,
    "budget_aware": True,
    "horizon_aware": True,
    "learning_rate": 0.0001,
    "weight_decay": 0.001,
    "batch_size": TRAINING_CONFIG["batch_size"],
    "prediction_type": "epsilon",
    "loss_reduction": "mean",
}
ENCODER_CONFIG = {
    "dataset": "bedroom_256",
    "input_channels": 3,
    "image_size": 64,
    "output_dim": 32,
    "arch_type": "simple_cnn",
    "channels": [16, 24, 32],
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
    "channels": [16, 24, 32],
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
    "manifold_bank_path": TRAINING_CONFIG["manifold_bank_path"],
    "manifold_k": 50,
}


def build_diffusion_policy(**overrides):
    from ..policy.diffusion_policy import DiffusionPolicy

    config = DIFFUSION_POLICY_CONFIG.copy()
    config.update(overrides)
    config.setdefault("encoder_config", ENCODER_CONFIG.copy())
    return DiffusionPolicy(**config)


def build_bedroom256_encoder(**overrides):
    from ..encoder.custom_encoder import CustomEncoder

    config = ENCODER_CONFIG.copy()
    config.update(overrides)
    return CustomEncoder(config, dataset="bedroom256")


def build_complete_system(**policy_overrides):
    dimension = policy_overrides.get("obs_feature_dim", DIFFUSION_POLICY_CONFIG["obs_feature_dim"])
    encoder = build_bedroom256_encoder(output_dim=dimension)
    policy = build_diffusion_policy(**policy_overrides)
    policy.visual_encoder = encoder
    return (encoder, policy)
