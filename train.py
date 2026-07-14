#!/usr/bin/env python3
"""WAXAL ASR — Training Script.

Fine-tunes a Gemma 3n model on the WaxalNLP dataset using LoRA (PEFT) and
the TRL SFTTrainer.  Integrates experiment tracking, rich logging,
visualization, and reproducibility snapshots.

Usage::

    python train.py                                    # default config
    python train.py --config configs/sna.yaml          # Shona override
    python train.py --config configs/multilingual.yaml  # all languages
    python train.py --name "exp-lr-sweep-1" --notes "Testing lr=5e-4"

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.dataset import load_datasets_from_config, log_dataset_statistics
from src.experiment import Experiment
from src.logging_utils import (
    log_best_checkpoint,
    log_gpu_memory_bar,
    log_training_summary,
    setup_colored_logging,
    TrainingTimer,
)
from src.model import load_model_and_processor
from src.trainer import build_trainer
from src.utils import (
    log_environment,
    log_gpu_info,
    log_memory_usage,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3n for African language ASR.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML override config (merged on top of configs/default.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated from timestamp).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text notes for the experiment registry.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()

    # Build CLI overrides dict
    cli_overrides: dict[str, object] = {}
    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.max_steps is not None:
        cli_overrides["training.max_steps"] = args.max_steps
    if args.learning_rate is not None:
        cli_overrides["training.learning_rate"] = args.learning_rate

    config = load_config(override_path=args.config, cli_overrides=cli_overrides)
    seed = config["seed"]
    training_cfg = config["training"]
    logging_cfg = config.get("logging", {})
    experiment_cfg = config.get("experiment", {})

    # Setup logging — use colored output if configured
    if logging_cfg.get("colored", True):
        setup_colored_logging()
    else:
        setup_logging()

    set_seed(seed)
    log_environment()
    log_gpu_info()

    # ---- Create experiment ----
    experiment: Experiment | None = None
    if experiment_cfg.get("enabled", True):
        experiment = Experiment.create(
            config=config,
            name=args.name or experiment_cfg.get("name"),
            notes=args.notes or experiment_cfg.get("notes", ""),
        )
        logger.info("Experiment directory: %s", experiment.dir)

    # ---- Load data ----
    logger.info("Loading datasets...")
    splits = load_datasets_from_config(config)
    train_ds = splits["train"]
    val_ds = splits["validation"]

    if logging_cfg.get("log_dataset_stats", True):
        log_dataset_statistics(train_ds, "train")

    # Shuffle and repeat training data for streaming
    shuffled_train = train_ds.shuffle(
        buffer_size=training_cfg.get("shuffle_buffer_size", 1000),
        seed=seed,
    ).repeat(None)

    # Fixed validation slice for reproducibility
    num_val = training_cfg.get("num_validation_examples", 200)
    val_ds_fixed = val_ds.take(num_val)

    # ---- Load model ----
    logger.info("Loading model and processor...")
    model, processor = load_model_and_processor(config)
    log_gpu_memory_bar()

    # ---- Build trainer and train ----
    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=shuffled_train,
        eval_dataset=val_ds_fixed,
        config=config,
    )

    logger.info("Starting fine-tuning...")
    timer = TrainingTimer(total_steps=training_cfg["max_steps"])
    start_time = time.monotonic()

    train_result = trainer.train()

    elapsed = time.monotonic() - start_time

    # Extract final loss from training history
    final_loss = None
    if train_result and hasattr(train_result, "training_loss"):
        final_loss = train_result.training_loss

    # ---- Save training log from trainer state ----
    if experiment and hasattr(trainer, "state") and trainer.state.log_history:
        for entry in trainer.state.log_history:
            step = entry.get("step", 0)
            experiment.log_step(step, entry)

            # Track best checkpoint by eval loss (WER computed during eval)
            if "eval_loss" in entry:
                checkpoint_path = str(Path(training_cfg["output_dir"]) / f"checkpoint-{step}")
                experiment.update_best(
                    step=step,
                    wer=entry.get("eval_loss", float("inf")),
                    checkpoint_path=checkpoint_path,
                )

        if experiment_cfg.get("save_training_log", True):
            experiment.save_training_log()

    # ---- Generate plots ----
    if experiment and experiment_cfg.get("generate_plots", True):
        try:
            from src.visualization import plot_training_loss, plot_eval_metrics
            plots_dir = experiment.dir / "plots"
            plot_training_loss(experiment._training_log, plots_dir / "training_loss.png")
            plot_eval_metrics(experiment._training_log, plots_dir / "eval_metrics.png")
        except ImportError:
            logger.warning("matplotlib not available — skipping plot generation.")

    # ---- Log summary ----
    log_training_summary(
        total_steps=training_cfg["max_steps"],
        elapsed_seconds=elapsed,
        final_loss=final_loss,
        best_wer=experiment.best_wer if experiment else float("inf"),
        best_step=experiment.best_step if experiment else -1,
    )

    # Save final checkpoint
    output_dir = Path(training_cfg["output_dir"])
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    logger.info("Final model saved to %s", final_dir)

    # ---- Register experiment ----
    if experiment:
        experiment.register()

    log_gpu_memory_bar()


if __name__ == "__main__":
    main()
