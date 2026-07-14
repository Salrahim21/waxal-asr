"""Whisper-specific dataset loading and preprocessing for WAXAL ASR.

Handles loading WaxalNLP data and preparing it for Whisper fine-tuning
(log-mel spectrogram features + tokenized labels).
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import datasets
import numpy as np
import torch
import transformers

logger = logging.getLogger(__name__)


def prepare_whisper_example(
    example: dict[str, Any],
    processor: transformers.WhisperProcessor,
    sample_rate: int = 16_000,
) -> dict[str, Any]:
    """Convert an audio example into Whisper input features and labels.

    Args:
        example: Dict with ``audio`` (decoded) and ``transcription`` keys.
        processor: The Whisper processor.
        sample_rate: Target sample rate.

    Returns:
        Dict with ``input_features`` and ``labels`` keys.
    """
    audio = example["audio"]
    audio_array = np.asarray(audio["array"], dtype=np.float32)

    # Extract log-mel spectrogram features
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="np",
    ).input_features[0]

    # Tokenize the transcription
    labels = processor.tokenizer(
        example["transcription"],
        return_tensors="np",
    ).input_ids[0]

    return {
        "input_features": input_features,
        "labels": labels,
    }


def prepare_whisper_batch(
    batch: dict[str, list[Any]],
    processor: transformers.WhisperProcessor,
    sample_rate: int = 16_000,
) -> dict[str, list[Any]]:
    """Batch version of :func:`prepare_whisper_example` for ``datasets.map``.

    Args:
        batch: Batched dict from HuggingFace datasets.
        processor: The Whisper processor.
        sample_rate: Target sample rate.

    Returns:
        Batched dict with ``input_features`` and ``labels``.
    """
    num = len(batch["transcription"])
    all_features = []
    all_labels = []

    for i in range(num):
        audio_array = np.asarray(batch["audio"][i]["array"], dtype=np.float32)

        features = processor.feature_extractor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="np",
        ).input_features[0]

        labels = processor.tokenizer(
            batch["transcription"][i],
            return_tensors="np",
        ).input_ids[0]

        all_features.append(features)
        all_labels.append(labels)

    return {
        "input_features": all_features,
        "labels": all_labels,
        "transcription": batch["transcription"],
    }


class WhisperDataCollator:
    """Data collator for Whisper fine-tuning.

    Pads input features and labels to the same length within a batch,
    replacing padding in labels with -100 for loss masking.
    """

    def __init__(self, processor: transformers.WhisperProcessor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of examples into a batch.

        Args:
            features: List of dicts with ``input_features`` and ``labels``.

        Returns:
            Batch dict with padded tensors.
        """
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels and mask padding with -100
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if Whisper adds one automatically during generation
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_whisper_datasets(
    config: dict[str, Any],
    processor: transformers.WhisperProcessor,
) -> dict[str, datasets.IterableDataset]:
    """Load and preprocess WaxalNLP datasets for Whisper fine-tuning.

    Args:
        config: Full configuration dictionary.
        processor: The Whisper processor.

    Returns:
        Dict with ``"train"``, ``"validation"``, ``"test"`` datasets.
    """
    dataset_cfg = config["dataset"]
    languages = dataset_cfg["languages"]
    sample_rate = dataset_cfg["sample_rate"]
    streaming = dataset_cfg.get("streaming", True)

    _prep_fn = functools.partial(
        prepare_whisper_batch,
        processor=processor,
        sample_rate=sample_rate,
    )

    result: dict[str, Any] = {}

    for split in ("train", "validation", "test"):
        split_datasets = []
        for lang in languages:
            logger.info("Loading WaxalNLP — language=%s, split=%s (Whisper)", lang, split)
            ds = datasets.load_dataset(
                dataset_cfg["dataset_id"],
                name=f"{lang}_asr",
                split=split,
                streaming=streaming,
            )
            ds = ds.cast_column("audio", datasets.Audio(sampling_rate=sample_rate))
            ds = ds.map(_prep_fn, batched=True, batch_size=16,
                        remove_columns=["audio", "id", "language", "original_split"])
            split_datasets.append(ds)

        if len(split_datasets) == 1:
            result[split] = split_datasets[0]
        else:
            result[split] = datasets.interleave_datasets(split_datasets)

    logger.info("Whisper datasets loaded for languages=%s", languages)
    return result
