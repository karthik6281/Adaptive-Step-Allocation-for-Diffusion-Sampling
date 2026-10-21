"""Load explicit experiment settings and construct matching runtime components."""

from __future__ import annotations

import copy
import importlib
import inspect
from pathlib import Path

import torch
import yaml

from models.diffusion import create_toy_diffusion
from training.advanced_reward import COMPONENTS, AdvancedRewardV3, validate_reward_schedule
from training.budget import validate_rollout_settings
from training.rl_trainer import RLTrainer
from training.tbptt_trainer import TBPTTRLTrainer
from utils.reproducibility import file_sha256, require_int, seed_everything

DATASETS = (
    "toy",
    "mnist",
    "cifar10",
    "cat256",
    "celeb64",
    "church256",
    "bedroom256",
    "celeb256",
    "ffhq256",
    "cin256",
)


def load_experiment(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        experiment = yaml.safe_load(stream)
    if not isinstance(experiment, dict):
        raise ValueError("The experiment file must contain a mapping.")
    allowed = {
        "dataset",
        "seed",
        "deterministic",
        "device",
        "threads",
        "model",
        "policy",
        "reward",
        "training",
    }
    if set(experiment) - allowed:
        raise ValueError(f"Unknown experiment keys: {sorted(set(experiment) - allowed)}")
    if experiment.get("dataset") not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    for name in ("model", "policy", "reward", "training"):
        if name not in experiment:
            raise ValueError(f"The experiment must explicitly define {name}.")
        if not isinstance(experiment[name], dict):
            raise ValueError(f"{name} must be a mapping")
    experiment = copy.deepcopy(experiment)
    for section, keys in (
        ("model", ("config_path", "checkpoint_path", "compvis_root")),
        ("reward", ("manifold_bank_path",)),
    ):
        for key in keys:
            value = experiment[section].get(key)
            if value:
                candidate = Path(value).expanduser()
                experiment[section][key] = str((path.parent / candidate).resolve())
    model_id = experiment["model"].get("model_id")
    if model_id and (model_id.startswith((".", "/")) or (path.parent / model_id).is_dir()):
        experiment["model"]["model_id"] = str((path.parent / Path(model_id).expanduser()).resolve())
    return experiment


def select_device(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda[:index].")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device is not available: {name}")
        return torch.device("cuda", index)
    return device


def create_model(specification, dataset, device, steps):
    from adapter_compvis import create_compvis_adapter, resolve_paths
    from models.pretrained_diffusion import create_diffusers_model

    spec = copy.deepcopy(specification)
    allowed = {
        "kind",
        "model_id",
        "revision",
        "config_path",
        "checkpoint_path",
        "compvis_root",
        "trust_checkpoint",
        "asset_sha256",
    }
    if set(spec) - allowed:
        raise ValueError(f"Unknown model keys: {sorted(set(spec) - allowed)}")
    kind = spec.get("kind")
    backend_keys = {
        "toy": {"kind", "asset_sha256"},
        "compvis": {
            "kind",
            "config_path",
            "checkpoint_path",
            "compvis_root",
            "trust_checkpoint",
            "asset_sha256",
        },
        "diffusers": {"kind", "model_id", "revision", "asset_sha256"},
    }
    if kind in backend_keys and set(spec) - backend_keys[kind]:
        raise ValueError(
            f"Unsupported {kind} model settings: {sorted(set(spec) - backend_keys[kind])}"
        )
    assets = {}
    if kind == "toy":
        if dataset != "toy" or set(spec) - {"kind", "asset_sha256"}:
            raise ValueError(
                "The toy model must use dataset=toy and cannot take pretrained-model settings."
            )
        model = create_toy_diffusion(steps).to(device)
    elif kind == "compvis":
        config, checkpoint, repo = resolve_paths(
            dataset, spec.get("config_path"), spec.get("checkpoint_path"), spec.get("compvis_root")
        )
        assets = {"config": file_sha256(config), "checkpoint": file_sha256(checkpoint)}
        if "asset_sha256" in spec and spec["asset_sha256"] != assets:
            raise ValueError("Pretrained model files do not match their recorded checksums.")
        model = create_compvis_adapter(
            config,
            checkpoint,
            device=device,
            n_timesteps=steps,
            compvis_root=repo,
            trust_checkpoint=spec.get("trust_checkpoint", False),
        )
        spec.update(
            config_path=str(config), checkpoint_path=str(checkpoint), compvis_root=str(repo)
        )
    elif kind == "diffusers":
        model_id = spec.get("model_id")
        if model_id and Path(model_id).is_dir():
            for path in sorted(Path(model_id).rglob("*")):
                if path.is_file() and path.suffix in (".json", ".safetensors", ".bin"):
                    assets[str(path.relative_to(model_id))] = file_sha256(path)
        if "asset_sha256" in spec and spec["asset_sha256"] != assets:
            raise ValueError("Local Diffusers files do not match their recorded checksums.")
        model = create_diffusers_model(model_id, spec.get("revision"), device, steps)
    else:
        raise ValueError("model.kind must be 'toy', 'compvis', or 'diffusers'.")
    spec["asset_sha256"] = assets
    return model, spec


def create_policy(specification, dataset, model):
    spec = copy.deepcopy(specification)
    kind = spec.pop("type", None)
    inferred = {
        "input_channels": model.sample_shape[0],
        "num_classes": model.num_classes,
        "max_diffusion_steps": model.num_timesteps,
    }
    for name, value in inferred.items():
        if name in spec and spec[name] != value:
            raise ValueError(f"Policy {name}={spec[name]} disagrees with the model value {value}.")
    if kind in ("diffusion", "direct"):
        from dasp.adapters.policy_adapter import DATASETS as DASP_DATASETS
        from dasp.adapters.policy_adapter import DASPPolicyAdapter

        if dataset in DASP_DATASETS:
            preset = importlib.import_module(f"dasp.config.config_{dataset}")
            parameters = copy.deepcopy(preset.DIFFUSION_POLICY_CONFIG)
            encoder = copy.deepcopy(preset.ENCODER_CONFIG)
        else:
            parameters = {
                "action_horizon": 8,
                "planning_interval": 2,
                "obs_feature_dim": 32,
                "n_layer": 2,
                "n_head": 4,
                "n_emb": 32,
                "n_cond_layers": 1,
                "num_inference_steps": 5,
            }
            encoder = {"channels": [16, 24, 32], "use_batch_norm": False, "dropout_rate": 0.0}
        optimizer_keys = {"learning_rate", "weight_decay", "batch_size"}
        if optimizer_keys & spec.keys():
            raise ValueError("Optimizer settings belong in training, not policy.")
        for key in optimizer_keys:
            parameters.pop(key, None)
        encoder_overrides = spec.pop("encoder_config", {})
        for name, value in (
            ("input_channels", model.sample_shape[0]),
            ("image_size", model.sample_shape[1]),
        ):
            if name in encoder_overrides and encoder_overrides[name] != value:
                raise ValueError(f"Encoder {name} disagrees with the model shape.")
        encoder.update(encoder_overrides)
        planning_mode = spec.pop("planning_mode", "multi")
        parameters.update(spec)
        parameters.pop("dataset", None)
        expected_output = parameters.get("obs_feature_dim", 256)
        if "output_dim" in encoder_overrides and encoder_overrides["output_dim"] != expected_output:
            raise ValueError("Encoder output_dim must equal the policy obs_feature_dim.")
        encoder["output_dim"] = expected_output
        parameters["max_diffusion_steps"] = model.num_timesteps
        parameters["num_classes"] = model.num_classes
        encoder["input_channels"] = model.sample_shape[0]
        encoder["image_size"] = model.sample_shape[1]
        policy = DASPPolicyAdapter(
            dataset, parameters, encoder, planning_mode=planning_mode, head=kind
        )
        resolved = {
            "type": kind,
            **parameters,
            "encoder_config": encoder,
            "planning_mode": planning_mode,
        }
    else:
        constructors = {
            "lstm": ("models.policy_network", "PolicyNetwork"),
            "lstm_cifar": ("models.policy_network_cifar", "PolicyNetworkCIFAR"),
            "mamba": ("models.policy_network_mamba", "PolicyNetworkMamba"),
            "mamba_cifar": ("models.policy_network_cifar_mamba", "PolicyNetworkMambaCIFAR"),
        }
        if kind not in constructors:
            raise ValueError(f"Unknown policy type: {kind}")
        module, name = constructors[kind]
        constructor = getattr(importlib.import_module(module), name)
        spec.update(inferred)
        policy = constructor(**spec)
        resolved = {"type": kind, **spec}
    return policy.to(model.device), resolved


def build_experiment(experiment, device_override=None, with_reward=True):
    resolved = copy.deepcopy(experiment)
    device = select_device(device_override or resolved.get("device", "cpu"))
    seed_everything(resolved.get("seed", 42), resolved.get("deterministic", True))
    threads = resolved.get("threads", 1)
    require_int(threads, "threads")
    torch.set_num_threads(threads)
    resolved.update(
        device=str(device),
        seed=resolved.get("seed", 42),
        deterministic=resolved.get("deterministic", True),
        threads=threads,
    )
    training = resolved["training"]
    extra_keys = {"batch_size", "trainer", "checkpoint_interval", "tbptt_chunk_size"}
    allowed = set(inspect.signature(RLTrainer).parameters) - {
        "diffusion_model",
        "policy_network",
        "reward_function",
        "device",
        "experiment",
    }
    if set(training) - allowed - extra_keys:
        raise ValueError(
            f"Unknown training settings: {sorted(set(training) - allowed - extra_keys)}"
        )
    training.setdefault("inference_steps", 8 if resolved["model"].get("kind") == "toy" else 200)
    training.setdefault("batch_size", 16)
    training.setdefault("num_episodes", 100)
    for name in ("batch_size", "num_episodes", "inference_steps"):
        require_int(training[name], name)
    for name in (
        "reference_steps",
        "ppo_epochs",
        "grpo_group_size",
        "checkpoint_interval",
        "tbptt_chunk_size",
        "truncation_steps",
    ):
        if training.get(name) is not None:
            require_int(training[name], name)
    require_int(training.get("warmup_anneal_episodes", 0), "warmup_anneal_episodes", minimum=0)
    validate_reward_schedule(training.get("reward_schedule", []), training["num_episodes"])
    ratios = (
        (training.get("warmup_initial_ratio"), training.get("warmup_final_ratio", 0.0))
        if training.get("warmup_initial_ratio") is not None
        and training.get("warmup_anneal_episodes", 0)
        else (training.get("warmup_ratio", 0.0),)
    )
    for ratio in ratios:
        validate_rollout_settings(
            training["inference_steps"],
            training.get("denoising_budget"),
            training.get("budget_mode", "cap"),
            ratio,
            training.get("max_consecutive_skips"),
        )
    model, resolved["model"] = create_model(
        resolved["model"], resolved["dataset"], device, training["inference_steps"]
    )
    policy, resolved["policy"] = create_policy(resolved["policy"], resolved["dataset"], model)
    reward = None
    if with_reward:
        actual = resolved["reward"]
        initialized = copy.deepcopy(actual)
        for stage in training.get("reward_schedule", []):
            for key, value in stage["weights"].items():
                initialized[key] = max(initialized.get(key, 0), value)
        reward = AdvancedRewardV3(device=device, dataset=resolved["dataset"], **initialized)
        for name in COMPONENTS:
            setattr(reward, f"w_{name}", actual.get(f"w_{name}", 0.0))
    return resolved, model, policy, reward


def create_trainer(resolved, model, policy, reward):
    options = copy.deepcopy(resolved["training"])
    options.pop("batch_size")
    options.pop("checkpoint_interval", None)
    kind = options.pop("trainer", "standard")
    chunk = options.pop("tbptt_chunk_size", None)
    if kind not in ("standard", "tbptt"):
        raise ValueError("training.trainer must be 'standard' or 'tbptt'.")
    if kind == "standard" and chunk is not None:
        raise ValueError("tbptt_chunk_size requires training.trainer=tbptt.")
    constructor = RLTrainer if kind == "standard" else TBPTTRLTrainer
    if kind == "tbptt":
        options["tbptt_chunk_size"] = 100 if chunk is None else chunk
    return constructor(model, policy, reward, device=model.device, experiment=resolved, **options)
