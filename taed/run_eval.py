"""Evaluate a clean-format policy checkpoint on paired initial noise."""

import argparse
import json
from pathlib import Path


def model_identity(specification):
    identity = {
        "kind": specification["kind"],
        "asset_sha256": specification.get("asset_sha256", {}),
    }
    if specification["kind"] == "diffusers" and not identity["asset_sha256"]:
        identity.update(
            model_id=specification.get("model_id"), revision=specification.get("revision")
        )
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", help="Override asset locations using an equivalent policy configuration"
    )
    parser.add_argument("--output-dir", required=True, help="New evaluation directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budget", type=int, help="Per-sample denoising budget; defaults to the training setting"
    )
    parser.add_argument(
        "--budget-mode", choices=("cap", "exact"), help="Defaults to the training setting"
    )
    parser.add_argument("--metric", choices=("none", "mse", "lpips"), default="none")
    parser.add_argument(
        "--warmup-ratio", type=float, help="Override the warmup ratio at the saved episode"
    )
    parser.add_argument(
        "--deterministic-actions",
        action="store_true",
        help="Threshold probabilities instead of Bernoulli sampling",
    )
    args = parser.parse_args(argv)

    import torch

    from config.experiment import build_experiment, load_experiment
    from training.rl_trainer import warmup_ratio_at
    from utils.evaluation import evaluate
    from utils.reproducibility import file_sha256, provenance, save_json, seed_everything

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(
            "Expected a clean-format checkpoint; historical checkpoints require explicit migration."
        )
    experiment = load_experiment(args.config) if args.config else payload["experiment"]
    resolved, model, policy, _ = build_experiment(experiment, args.device, with_reward=False)
    if resolved["policy"] != payload["experiment"]["policy"]:
        raise ValueError("Evaluation policy settings must match the saved checkpoint.")
    saved = payload["experiment"]
    if model_identity(resolved["model"]) != model_identity(saved["model"]):
        raise ValueError("Evaluation backbone assets must match the saved checkpoint.")
    if resolved["dataset"] != saved["dataset"] or resolved["training"] != saved["training"]:
        raise ValueError(
            "Use explicit evaluation flags to override sampling settings, not --config."
        )
    policy.load_state_dict(payload["policy_network_state_dict"], strict=True)
    seed_everything(args.seed, resolved["deterministic"])
    training = resolved["training"]
    warmup = args.warmup_ratio
    if warmup is None:
        warmup = warmup_ratio_at(
            max(0, payload["episode"] - 1),
            training.get("warmup_ratio", 0.0),
            training.get("warmup_initial_ratio"),
            training.get("warmup_final_ratio", 0.0),
            training.get("warmup_anneal_episodes", 0),
        )
    summary = evaluate(
        model,
        policy,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=training["inference_steps"],
        budget=args.budget if args.budget is not None else training.get("denoising_budget"),
        budget_mode=args.budget_mode or training.get("budget_mode", "cap"),
        reference_steps=training.get("reference_steps"),
        seed=args.seed,
        metric=args.metric,
        stochastic=not args.deterministic_actions,
        output_dir=args.output_dir,
        warmup_ratio=warmup,
        max_consecutive_skips=training.get("max_consecutive_skips"),
    )
    save_json(
        {
            "experiment": resolved,
            "environment": provenance(),
            "policy_checkpoint_sha256": file_sha256(args.checkpoint),
            "evaluation": summary,
            "deterministic_actions": args.deterministic_actions,
        },
        Path(args.output_dir) / "run.json",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
