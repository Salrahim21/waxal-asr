"""Whisper inference utilities for WAXAL ASR.

Provides batch transcription for Whisper models on HuggingFace datasets
and single-file transcription.
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


def whisper_transcribe_batch(
    batch: dict[str, list[Any]],
    model: transformers.WhisperForConditionalGeneration,
    processor: transformers.WhisperProcessor,
    device: torch.device,
    max_new_tokens: int = 225,
    num_beams: int = 1,
    length_penalty: float = 1.0,
) -> list[str]:
    """Run Whisper inference on a batch of preprocessed examples.

    Args:
        batch: Batched dict with ``input_features`` key.
        model: The fine-tuned Whisper model.
        processor: The Whisper processor.
        device: Target torch device.
        max_new_tokens: Max tokens to generate.
        num_beams: Beam search width (1 = greedy).
        length_penalty: Length penalty for beam search.

    Returns:
        List of decoded transcription strings.
    """
    input_features = torch.tensor(
        np.array(batch["input_features"]), dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    transcriptions = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )
    return [t.strip() for t in transcriptions]


def whisper_transcribe_file(
    audio_path: str | Path,
    model: transformers.WhisperForConditionalGeneration,
    processor: transformers.WhisperProcessor,
    device: torch.device,
    sample_rate: int = 16_000,
    max_new_tokens: int = 225,
    num_beams: int = 1,
) -> str:
    """Transcribe a single audio file using Whisper.

    Args:
        audio_path: Path to the audio file.
        model: The fine-tuned Whisper model.
        processor: The Whisper processor.
        device: Torch device.
        sample_rate: Target sample rate.
        max_new_tokens: Max tokens to generate.
        num_beams: Beam search width.

    Returns:
        Transcribed text.
    """
    audio_path = Path(audio_path)
    logger.info("Loading audio from %s", audio_path)

    audio_array, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    input_features = processor.feature_extractor(
        audio_array, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    transcript = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
    logger.info("Transcription: %s", transcript)
    return transcript
