#!/usr/bin/env python3
"""WAXAL ASR — Submission Generation Script.

Generates a ``submission.csv`` file in the format expected by the Zindi
competition platform.  The Test.csv file contains IDs with language prefixes
(e.g. ``lug_12345``), which are used to load the corresponding audio from
the WaxalNLP HuggingFace dataset and generate transcriptions.

Usage::

    python submit.py                              # default config
    python submit.py --config configs/sna.yaml    # Shona-only model
    python submit.py --output submission_v2.csv   # custom output path

"""

from __future__ import annotations

import argparse
import csv
import functools
import logging
import sys
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.inference import transcribe_batch
from src.model import load_model_and_processor
from src.preprocessing import format_batch
from src.utils import get_device, log_gpu_info, log_memory_usage, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Zindi submission CSV for the WAXAL ASR challenge.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML override config.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to Test.csv (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for submission CSV (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    return parser.parse_args()


def load_test_ids(test_csv_path: str | Path) -> list[dict[str, str]]:
    """Load test IDs and parse language codes from the ID prefix.

    Each row in Test.csv has an ``ID`` column like ``lug_12345``.  The prefix
    before the underscore is the language code.

    Args:
        test_csv_path: Path to the competition Test.csv file.

    Returns:
        List of dicts with ``"id"`` and ``"language"`` keys.
    """
    entries: list[dict[str, str]] = []
    with open(test_csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            test_id = row["ID"]
            language = test_id.rsplit("_", 1)[0]
            entries.append({"id": test_id, "language": language})
    logger.info("Loaded %d test IDs from %s", len(entries), test_csv_path)
    return entries


def validate_submission(
    submission_path: Path,
    sample_submission_path: str | Path,
) -> bool:
    """Validate that the submission CSV matches the expected format.

    Args:
        submission_path: Path to the generated submission.
        sample_submission_path: Path to the sample submission for reference.

    Returns:
        ``True`` if validation passes, ``False`` otherwise.
    """
    with open(sample_submission_path, "r", encoding="utf-8") as fh:
        expected_ids = {row["ID"] for row in csv.DictReader(fh)}

    with open(submission_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["ID", "Target"]:
            logger.error(
                "Column mismatch: expected ['ID', 'Target'], got %s",
                reader.fieldnames,
            )
            return False

        submitted_ids: set[str] = set()
        empty_count = 0
        for row in reader:
            submitted_ids.add(row["ID"])
            if not row["Target"].strip():
                empty_count += 1

    missing = expected_ids - submitted_ids
    extra = submitted_ids - expected_ids

    if missing:
        logger.error("Missing %d IDs in submission.", len(missing))
        return False
    if extra:
        logger.error("Found %d unexpected IDs in submission.", len(extra))
        return False
    if empty_count > 0:
        logger.warning("%d rows have empty predictions.", empty_count)

    logger.info(
        "Submission validated: %d rows, %d non-empty.",
        len(submitted_ids),
        len(submitted_ids) - empty_count,
    )
    return True


def main() -> None:
    """Entry point for submission generation."""
    setup_logging()
    args = parse_args()

    config = load_config(override_path=args.config)
    set_seed(config["seed"])
    log_gpu_info()

    sub_cfg = config["submission"]
    test_csv_path = Path(args.test_csv or sub_cfg["test_csv"])
    output_path = Path(args.output or sub_cfg["output_path"])
    sample_csv_path = Path(sub_cfg["sample_submission_csv"])
    batch_size = args.batch_size
    max_new_tokens = config["inference"]["max_new_tokens"]
    sample_rate = config["dataset"]["sample_rate"]
    prompt_cfg = config["prompts"]

    # ---- Load model ----
    model, processor = load_model_and_processor(config)
    device = get_device()
    model.eval()
    log_memory_usage()

    # ---- Load test IDs and group by language ----
    test_entries = load_test_ids(test_csv_path)
    languages_in_test = sorted({e["language"] for e in test_entries})
    logger.info("Languages in test set: %s", languages_in_test)

    # ---- Generate predictions per language ----
    predictions_map: dict[str, str] = {}

    for language in languages_in_test:
        lang_ids = {e["id"] for e in test_entries if e["language"] == language}
        logger.info("Processing language=%s (%d examples)", language, len(lang_ids))

        # Load test split for this language
        ds = datasets.load_dataset(
            config["dataset"]["dataset_id"],
            name=f"{language}_asr",
            split="test",
            streaming=True,
        )
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=sample_rate))

        _format_fn = functools.partial(
            format_batch,
            system_message=prompt_cfg["system_message"],
            user_message=prompt_cfg["user_message"],
        )
        ds = ds.map(_format_fn, batched=True, batch_size=32)

        for batch_data in tqdm(
            ds.batch(batch_size=batch_size),
            desc=f"Transcribing {language}",
        ):
            preds = transcribe_batch(
                batch_data, model, processor, device, max_new_tokens
            )
            # Map predictions to IDs
            batch_ids = batch_data.get("id", batch_data.get("ID", []))
            for sample_id, pred in zip(batch_ids, preds):
                full_id = f"{language}_{sample_id}" if not str(sample_id).startswith(language) else str(sample_id)
                if full_id in lang_ids:
                    predictions_map[full_id] = pred

    # ---- Write submission CSV ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ID", "Target"])
        for entry in test_entries:
            test_id = entry["id"]
            prediction = predictions_map.get(test_id, "")
            writer.writerow([test_id, prediction])

    logger.info("Submission written to %s (%d rows)", output_path, len(test_entries))

    # ---- Validate ----
    if sample_csv_path.exists():
        is_valid = validate_submission(output_path, sample_csv_path)
        if is_valid:
            logger.info("Submission validation PASSED.")
        else:
            logger.error("Submission validation FAILED.")
    else:
        logger.warning("Sample submission not found at %s — skipping validation.", sample_csv_path)

    log_memory_usage()


if __name__ == "__main__":
    main()
