"""Whisper training orchestration for WAXAL ASR.

Builds a HuggingFace Seq2SeqTrainer configured for Whisper fine-tuning
with optional LoRA and WER/CER metric computation during evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import peft
import torch
import transformers

logger = logging.getLogger(__name__)


def build_whisper_lora_config(config: dict[str, Any]) -> peft.LoraConfig | None:
    """Construct a LoRA config for Whisper, or None if LoRA is disabled.

    Args:
        config: Full configuration dictionary.

    Returns:
        A ``peft.LoraConfig`` or ``None``.
    """
    lora_cfg = config.get("lora", {})
    if not lora_cfg or lora_cfg.get("r", 0) == 0:
        return None

    # Whisper-specific target modules
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=list(target_modules),
        bias=lora_cfg.get("bias", "none"),
    )
    logger.info("Whisper LoRA: r=%d, alpha=%d, targets=%s",
                lora_config.r, lora_config.lora_alpha, lora_config.target_modules)
    return lora_config


def build_whisper_training_args(config: dict[str, Any]) -> transformers.Seq2SeqTrainingArguments:
    """Build Seq2SeqTrainingArguments from the YAML config.

    Args:
        config: Full configuration dictionary.

    Returns:
        A ``Seq2SeqTrainingArguments`` instance.
    """
    t = config["training"]
    seed = config["seed"]
    languages = config["dataset"]["languages"]
    run_name = f"whisper-asr-{'-'.join(languages)}"
    report_to = t.get("report_to", "none")

    args = transformers.Seq2SeqTrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t.get("num_train_epochs", 1),
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        learning_rate=t["learning_rate"],
        logging_steps=t["logging_steps"],
        save_steps=t.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 500),
        predict_with_generate=True,
        generation_max_length=t.get("max_seq_length", 225),
        fp16=t.get("fp16", True),
        bf16=t.get("bf16", False),
        report_to=report_to,
        run_name=run_name,
        seed=seed,
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    logger.info(
        "Whisper training args: epochs=%s, max_steps=%d, lr=%.1e, batch=%d, grad_accum=%d",
        t.get("num_train_epochs", 1),
        t.get("max_steps", -1),
        t["learning_rate"],
        t["per_device_train_batch_size"],
        t["gradient_accumulation_steps"],
    )
    return args


def build_whisper_compute_metrics(
    processor: transformers.WhisperProcessor,
) -> Any:
    """Build a compute_metrics function for Seq2SeqTrainer.

    Args:
        processor: The Whisper processor.

    Returns:
        A callable that computes WER and CER from predictions.
    """
    import jiwer

    def compute_metrics(pred: transformers.EvalPrediction) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 in labels (padding) with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_lower = [p.lower().strip() for p in pred_str]
        label_lower = [l.lower().strip() for l in label_str]

        # Filter out empty references
        pairs = [(r, p) for r, p in zip(label_lower, pred_lower) if r]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}

        refs, preds = zip(*pairs)
        wer = jiwer.wer(list(refs), list(preds))
        cer = jiwer.cer(list(refs), list(preds))

        return {"wer": wer, "cer": cer}

    return compute_metrics


def build_whisper_trainer(
    model: transformers.WhisperForConditionalGeneration,
    processor: transformers.WhisperProcessor,
    train_dataset: datasets.IterableDataset,
    eval_dataset: datasets.IterableDataset,
    data_collator: Any,
    config: dict[str, Any],
) -> transformers.Seq2SeqTrainer:
    """Build a fully-configured Seq2SeqTrainer for Whisper.

    Args:
        model: The Whisper model (optionally with LoRA applied).
        processor: The Whisper processor.
        train_dataset: Preprocessed training dataset.
        eval_dataset: Preprocessed validation dataset.
        data_collator: Whisper data collator.
        config: Full configuration dictionary.

    Returns:
        A ``Seq2SeqTrainer`` instance.
    """
    training_args = build_whisper_training_args(config)
    compute_metrics = build_whisper_compute_metrics(processor)

    # Apply LoRA if configured
    lora_config = build_whisper_lora_config(config)
    if lora_config is not None:
        model = peft.get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    logger.info("Whisper trainer built successfully.")
    return trainer
