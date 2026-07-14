"""Whisper model loading for WAXAL ASR.

Loads an OpenAI Whisper model and its processor/tokenizer from the
HuggingFace Hub for fine-tuning on African language ASR.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import transformers

logger = logging.getLogger(__name__)

# Mapping from config strings to torch dtypes
_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_whisper_model_and_processor(
    config: dict[str, Any],
) -> tuple[transformers.WhisperForConditionalGeneration, transformers.WhisperProcessor]:
    """Load a Whisper model and processor from the HuggingFace Hub.

    Args:
        config: Full configuration dictionary. Uses ``config["model"]`` keys.

    Returns:
        A ``(model, processor)`` tuple ready for training or inference.
    """
    model_cfg = config["model"]
    model_id: str = model_cfg["model_id"]
    dtype_str: str = model_cfg.get("torch_dtype", "float16")
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.float16)

    logger.info("Loading Whisper processor from %s", model_id)
    processor = transformers.WhisperProcessor.from_pretrained(model_id)

    logger.info("Loading Whisper model from %s (dtype=%s)", model_id, dtype_str)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    # Disable token forcing for fine-tuning on non-English languages
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    logger.info("Whisper model loaded — dtype=%s, params=%.1fM",
                model.dtype, sum(p.numel() for p in model.parameters()) / 1e6)
    return model, processor
