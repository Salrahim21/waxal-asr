"""Inference utilities for WAXAL ASR.

Provides batch and single-file transcription functions that wrap the Gemma 3n
generate pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import transformers

logger = logging.getLogger(__name__)


def transcribe_batch(
    batch: dict[str, list[Any]],
    model: transformers.Gemma3nForConditionalGeneration,
    processor: transformers.AutoProcessor,
    device: torch.device,
    max_new_tokens: int = 128,
) -> list[str]:
    """Run inference on a batch of chat-formatted examples.

    Args:
        batch: Batched dict with ``messages`` and ``audio`` keys.
        model: The fine-tuned Gemma model.
        processor: The model processor.
        device: Target torch device.
        max_new_tokens: Maximum new tokens to generate per example.

    Returns:
        List of decoded transcription strings.
    """
    # Remove assistant turn (ground-truth) to create generation prompts
    messages_list = [msgs[:-1] for msgs in batch["messages"]]
    audio_list = [np.asarray(a["array"]).flatten() for a in batch["audio"]]

    text_prompts = processor.tokenizer.apply_chat_template(
        messages_list, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text_prompts,
        audio=audio_list,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    decoded = processor.tokenizer.batch_decode(
        outputs[:, input_len:], skip_special_tokens=True
    )
    return [t.strip() for t in decoded]


def transcribe_audio_file(
    audio_path: str | Path,
    model: transformers.Gemma3nForConditionalGeneration,
    processor: transformers.AutoProcessor,
    device: torch.device,
    sample_rate: int = 16_000,
    system_message: str = "You are an assistant that transcribes speech accurately.",
    user_message: str = "Please transcribe this audio.",
    max_new_tokens: int = 128,
) -> str:
    """Transcribe a single WAV file.

    Args:
        audio_path: Path to an audio file (WAV, FLAC, MP3, etc.).
        model: The fine-tuned Gemma model.
        processor: The model processor.
        device: Target torch device.
        sample_rate: Target sample rate for resampling.
        system_message: System prompt for chat formatting.
        user_message: User prompt for chat formatting.
        max_new_tokens: Maximum new tokens to generate.

    Returns:
        The transcribed text.
    """
    audio_path = Path(audio_path)
    logger.info("Loading audio from %s", audio_path)

    audio_array, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": user_message},
            ],
        },
    ]

    text_prompt = processor.tokenizer.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text_prompt,
        audio=[audio_array],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    decoded = processor.tokenizer.decode(
        outputs[0, input_len:], skip_special_tokens=True
    )
    transcript = decoded.strip()
    logger.info("Transcription: %s", transcript)
    return transcript
