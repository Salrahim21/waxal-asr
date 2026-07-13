"""Model loading for WAXAL ASR.

Encapsulates loading a Gemma 3n (or 4n) model and its processor from the
HuggingFace Hub, with proper dtype and device-map configuration.
"""

from __future__ import annotations

import logging
from typing import Any

import timm  # noqa: F401 — must be imported before transformers for Gemma3n
import torch
import transformers

logger = logging.getLogger(__name__)

# Mapping from config strings to torch dtypes
_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model_and_processor(
    config: dict[str, Any],
) -> tuple[transformers.Gemma3nForConditionalGeneration, transformers.AutoProcessor]:
    """Load a Gemma3n/4n model and processor from the HuggingFace Hub.

    Args:
        config: Full configuration dictionary.  Uses ``config["model"]`` keys.

    Returns:
        A ``(model, processor)`` tuple ready for training or inference.
    """
    model_cfg = config["model"]
    model_id: str = model_cfg["model_id"]
    dtype_str: str = model_cfg.get("torch_dtype", "bfloat16")
    device_map: str = model_cfg.get("device_map", "auto")
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.bfloat16)

    logger.info("Loading processor from %s", model_id)
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    logger.info("Loading model from %s (dtype=%s, device_map=%s)", model_id, dtype_str, device_map)
    model = transformers.Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    logger.info("Model loaded — device=%s, dtype=%s", model.device, model.dtype)
    return model, processor
