"""Train a policy from an explicit experiment file."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML or JSON file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New run directory, or its original directory when resuming",
    )
    parser.add_argument("--device", help="cpu or cuda[:index]; overrides the experiment setting")
    parser.add_argument("--resume", help="Resume a clean-format checkpoint with identical settings")
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop at this absolute episode count without changing the LR schedule",
    )
    args = parser.parse_args(argv)

    from config.experiment import build_experiment, create_trainer, load_experiment
    from utils.reproducibility import provenance, save_json

    output = Path(args.output_dir)
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop-after must be a positive episode count.")
    if output.exists() and not (args.resume and (output / "run.json").is_file()):
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new run directory.")
    if output.exists() and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("An existing run can only resume a checkpoint from that run directory.")
    resolved, model, policy, reward = build_experiment(load_experiment(args.config), args.device)
    if output.exists():
        metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if metadata.get("experiment") != resolved or metadata.get("environment") != provenance():
            raise ValueError(
                "The existing run directory does not match this experiment and environment."
            )
    trainer = create_trainer(resolved, model, policy, reward)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    limit = resolved["training"]["num_episodes"] if args.stop_after is None else args.stop_after
    if not trainer.episode < limit <= trainer.num_episodes:
        raise ValueError(
            "stop-after must exceed the current episode and not exceed the configured final episode."
        )
    output.mkdir(parents=True, exist_ok=True)
    save_json({"experiment": resolved, "environment": provenance()}, output / "run.json")
    save_json(resolved, output / "config.json")
    trainer.train(
        limit,
        resolved["training"]["batch_size"],
        checkpoint_dir=output,
        checkpoint_interval=resolved["training"].get("checkpoint_interval", 1),
    )
    save_json({"episodes": trainer.history}, output / "history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
