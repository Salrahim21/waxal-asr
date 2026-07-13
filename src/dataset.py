"""Dataset loading and preparation for WAXAL ASR.

Wraps HuggingFace ``datasets`` streaming to load the WaxalNLP corpus,
resample audio, and apply chat formatting.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import datasets
import numpy as np

from src.preprocessing import format_batch

logger = logging.getLogger(__name__)


def load_language_split(
    dataset_id: str,
    language: str,
    split: str,
    sample_rate: int,
    system_message: str,
    user_message: str,
    streaming: bool = True,
    format_batch_size: int = 32,
) -> datasets.IterableDataset | datasets.Dataset:
    """Load a single language/split from WaxalNLP and apply chat formatting.

    Args:
        dataset_id: HuggingFace dataset identifier.
        language: ISO 639-3 language code (e.g. ``"lug"``).
        split: Dataset split name (``"train"``, ``"validation"``, ``"test"``).
        sample_rate: Target audio sample rate in Hz.
        system_message: System prompt for chat formatting.
        user_message: User prompt for chat formatting.
        streaming: Whether to use streaming mode.
        format_batch_size: Batch size for the formatting map operation.

    Returns:
        A formatted (Iterable)Dataset with ``messages`` column added.
    """
    logger.info("Loading WaxalNLP — language=%s, split=%s", language, split)

    ds = datasets.load_dataset(
        dataset_id,
        name=f"{language}_asr",
        split=split,
        streaming=streaming,
    )
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=sample_rate))

    _format_fn = functools.partial(
        format_batch,
        system_message=system_message,
        user_message=user_message,
    )
    ds = ds.map(_format_fn, batched=True, batch_size=format_batch_size)
    return ds


def load_datasets_from_config(
    config: dict[str, Any],
) -> dict[str, datasets.IterableDataset | datasets.Dataset]:
    """Load train, validation, and test splits for all configured languages.

    Args:
        config: Full configuration dictionary.

    Returns:
        Dict with keys ``"train"``, ``"validation"``, ``"test"``, each
        mapping to a dataset (potentially interleaved across languages).
    """
    dataset_cfg = config["dataset"]
    prompt_cfg = config["prompts"]
    languages = dataset_cfg["languages"]

    result: dict[str, Any] = {}

    for split in ("train", "validation", "test"):
        split_datasets = []
        for lang in languages:
            ds = load_language_split(
                dataset_id=dataset_cfg["dataset_id"],
                language=lang,
                split=split,
                sample_rate=dataset_cfg["sample_rate"],
                system_message=prompt_cfg["system_message"],
                user_message=prompt_cfg["user_message"],
                streaming=dataset_cfg.get("streaming", True),
                format_batch_size=dataset_cfg.get("format_batch_size", 32),
            )
            split_datasets.append(ds)

        if len(split_datasets) == 1:
            result[split] = split_datasets[0]
        else:
            result[split] = datasets.interleave_datasets(split_datasets)

    logger.info(
        "Loaded datasets for languages=%s, splits=%s",
        languages,
        list(result.keys()),
    )
    return result


def log_dataset_statistics(dataset: datasets.IterableDataset, name: str, num_peek: int = 100) -> None:
    """Log basic statistics from the first *num_peek* examples.

    Args:
        dataset: The dataset to inspect.
        name: A label for logging (e.g. ``"train"``).
        num_peek: Number of examples to peek at for statistics.
    """
    durations: list[float] = []
    transcription_lengths: list[int] = []

    for i, example in enumerate(dataset):
        if i >= num_peek:
            break
        audio = example.get("audio", {})
        array = audio.get("array")
        sr = audio.get("sampling_rate", 16000)
        if array is not None:
            durations.append(len(np.asarray(array)) / sr)
        transcription = example.get("transcription", "")
        transcription_lengths.append(len(transcription))

    if durations:
        logger.info(
            "[%s] Peeked %d examples — audio duration: min=%.1fs, max=%.1fs, mean=%.1fs",
            name,
            len(durations),
            min(durations),
            max(durations),
            np.mean(durations),
        )
    if transcription_lengths:
        logger.info(
            "[%s] Transcription length (chars): min=%d, max=%d, mean=%.0f",
            name,
            min(transcription_lengths),
            max(transcription_lengths),
            np.mean(transcription_lengths),
        )
