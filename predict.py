#!/usr/bin/env python3
"""WAXAL ASR — Single-File Prediction Script.

Transcribes one or more audio files using a fine-tuned Gemma 3n model.

Usage::

    python predict.py audio.wav
    python predict.py file1.wav file2.wav --config configs/sna.yaml

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.inference import transcribe_audio_file
from src.model import load_model_and_processor
from src.utils import get_device, log_gpu_info, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using a fine-tuned Gemma ASR model.",
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        type=str,
        help="Path(s) to audio file(s) to transcribe.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML override config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max new tokens for generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for prediction."""
    setup_logging()
    args = parse_args()

    config = load_config(override_path=args.config)
    set_seed(config["seed"])
    log_gpu_info()

    inference_cfg = config["inference"]
    max_new_tokens = args.max_new_tokens or inference_cfg["max_new_tokens"]
    sample_rate = config["dataset"]["sample_rate"]
    system_message = config["prompts"]["system_message"]
    user_message = config["prompts"]["user_message"]

    # ---- Load model ----
    model, processor = load_model_and_processor(config)
    device = get_device()

    # ---- Transcribe each file ----
    for audio_path in args.audio_files:
        path = Path(audio_path)
        if not path.exists():
            logger.error("File not found: %s", path)
            continue

        transcript = transcribe_audio_file(
            audio_path=path,
            model=model,
            processor=processor,
            device=device,
            sample_rate=sample_rate,
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=max_new_tokens,
        )
        print(f"[{path.name}] {transcript}")


if __name__ == "__main__":
    main()
