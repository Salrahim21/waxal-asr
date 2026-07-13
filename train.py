#!/usr/bin/env python3
"""WAXAL ASR — Training Script.

Fine-tunes a Gemma 3n model on the WaxalNLP dataset using LoRA (PEFT) and
the TRL SFTTrainer.

Usage::

    python train.py                                    # default config
    python train.py --config configs/sna.yaml          # Shona override
    python train.py --config configs/multilingual.yaml  # all languages

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.dataset import load_datasets_from_config, log_dataset_statistics
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
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    setup_logging()
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

    set_seed(seed)
    log_environment()
    log_gpu_info()

    # ---- Load data ----
    logger.info("Loading datasets...")
    splits = load_datasets_from_config(config)
    train_ds = splits["train"]
    val_ds = splits["validation"]

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
    log_memory_usage()

    # ---- Build trainer and train ----
    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=shuffled_train,
        eval_dataset=val_ds_fixed,
        config=config,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning complete.")

    # Save final checkpoint
    output_dir = Path(training_cfg["output_dir"])
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    logger.info("Final model saved to %s", final_dir)
    log_memory_usage()


if __name__ == "__main__":
    main()
