"""Data collator for WAXAL ASR.

Tokenises chat-formatted examples on-the-fly and masks padding / special
tokens in the labels so the cross-entropy loss is only computed on real
transcription tokens.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import transformers

logger = logging.getLogger(__name__)


def _mask_labels(
    labels: torch.Tensor,
    processor: transformers.AutoProcessor,
) -> torch.Tensor:
    """Set padding and special-token positions to -100 (ignored by CE loss).

    Args:
        labels: Label tensor of shape ``(batch, seq_len)``.
        processor: The model processor (must expose a ``tokenizer``).

    Returns:
        A cloned label tensor with special positions masked to ``-100``.
    """
    labels = labels.clone()
    tokenizer = processor.tokenizer

    special_attrs = [
        "pad_token_id",
        "image_token_id",
        "audio_token_id",
        "boi_token_id",
        "eoi_token_id",
    ]
    mask_ids = [
        getattr(tokenizer, attr)
        for attr in special_attrs
        if getattr(tokenizer, attr, None) is not None
    ]
    if mask_ids:
        labels[
            torch.isin(labels, torch.tensor(mask_ids, device=labels.device))
        ] = -100
    return labels


def collate_fn(
    examples: Sequence[Mapping[str, Any]],
    processor: transformers.AutoProcessor,
) -> dict[str, Any]:
    """Collate a list of chat-formatted examples into a training batch.

    Args:
        examples: List of dicts, each with ``messages`` and ``audio`` keys.
        processor: The model processor.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and ``labels`` tensors.
    """
    texts = [
        processor.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        for ex in examples
    ]
    audios = [np.asarray(ex["audio"]["array"]).flatten() for ex in examples]

    batch = processor(
        text=texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    batch = {
        k: v.detach().clone() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch["labels"] = _mask_labels(batch["input_ids"], processor)
    return batch


def build_collator(
    processor: transformers.AutoProcessor,
) -> functools.partial[dict[str, Any]]:
    """Return a collator partial bound to *processor*.

    Args:
        processor: The model processor.

    Returns:
        A callable suitable for ``data_collator`` in the Trainer.
    """
    return functools.partial(collate_fn, processor=processor)
