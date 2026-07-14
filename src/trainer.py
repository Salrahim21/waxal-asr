"""Training orchestration for WAXAL ASR.

Wraps PEFT LoRA configuration and TRL SFTTrainer setup.  The public entry
point is :func:`build_trainer`, which returns a fully-configured Trainer
ready for ``.train()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import datasets
import peft
import torch
import transformers
import trl

from src.collator import build_collator

logger = logging.getLogger(__name__)


class _SFTTrainer(trl.SFTTrainer):
    """SFTTrainer subclass with a no-op ``create_model_card`` override.

    The default implementation can raise ``importlib`` errors in some
    environments, so we skip model-card generation entirely.
    """

    def create_model_card(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        pass


def build_lora_config(config: dict[str, Any]) -> peft.LoraConfig:
    """Construct a LoRA configuration from the YAML config.

    Args:
        config: Full configuration dictionary.

    Returns:
        A ``peft.LoraConfig`` instance.
    """
    lora_cfg = config["lora"]
    lora_config = peft.LoraConfig(
        task_type=lora_cfg["task_type"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=list(lora_cfg["target_modules"]),
        bias=lora_cfg["bias"],
        use_rslora=lora_cfg.get("use_rslora", False),
        use_dora=lora_cfg.get("use_dora", False),
    )
    logger.info(
        "LoRA config: r=%d, alpha=%d, targets=%s",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.target_modules,
    )
    return lora_config


def build_training_args(config: dict[str, Any]) -> trl.SFTConfig:
    """Construct SFT training arguments from the YAML config.

    Supports optional W&B and TensorBoard integration via
    ``config["training"]["report_to"]``.

    Args:
        config: Full configuration dictionary.

    Returns:
        A ``trl.SFTConfig`` instance.
    """
    t = config["training"]
    seed = config["seed"]
    languages = config["dataset"]["languages"]
    run_name = f"gemma3n-asr-{'-'.join(languages)}"

    report_to = t.get("report_to", "none")

    # Resolve TensorBoard log dir
    tb_cfg = config.get("tensorboard", {})
    logging_dir = tb_cfg.get("log_dir") or (str(Path(t["output_dir"]) / "runs") if report_to == "tensorboard" else None)

    args = trl.SFTConfig(
        output_dir=t["output_dir"],
        max_steps=t["max_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=t["learning_rate"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=(not torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        report_to=report_to,
        run_name=run_name,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        max_length=t["max_seq_length"],
        packing=t.get("packing", False),
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        seed=seed,
        **({"logging_dir": logging_dir} if logging_dir else {}),
    )

    # Configure W&B if selected
    if report_to == "wandb":
        import os
        wandb_cfg = config.get("wandb", {})
        if wandb_cfg.get("project"):
            os.environ.setdefault("WANDB_PROJECT", wandb_cfg["project"])
        if wandb_cfg.get("entity"):
            os.environ.setdefault("WANDB_ENTITY", wandb_cfg["entity"])

    logger.info(
        "Training args: max_steps=%d, lr=%.1e, batch=%d, grad_accum=%d, report_to=%s",
        args.max_steps,
        args.learning_rate,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        report_to,
    )
    return args


def build_trainer(
    model: transformers.Gemma3nForConditionalGeneration,
    processor: transformers.AutoProcessor,
    train_dataset: datasets.IterableDataset,
    eval_dataset: datasets.IterableDataset,
    config: dict[str, Any],
) -> _SFTTrainer:
    """Build a fully-configured SFTTrainer with LoRA.

    Args:
        model: The pre-trained Gemma model.
        processor: The model processor.
        train_dataset: Formatted training dataset (already shuffled/repeated).
        eval_dataset: Formatted validation dataset.
        config: Full configuration dictionary.

    Returns:
        An ``_SFTTrainer`` instance ready for ``.train()``.
    """
    peft_config = build_lora_config(config)
    training_args = build_training_args(config)
    collator = build_collator(processor)

    trainer = _SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    logger.info("Trainer built successfully.")
    return trainer
