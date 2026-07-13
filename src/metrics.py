"""ASR evaluation metrics for WAXAL ASR.

Computes Word Error Rate (WER) and Character Error Rate (CER) using the
``jiwer`` library, with optional per-example breakdowns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jiwer

logger = logging.getLogger(__name__)


def compute_wer(references: list[str], predictions: list[str]) -> float:
    """Compute Word Error Rate between reference and predicted transcriptions.

    Both inputs are lowercased before comparison.

    Args:
        references: Ground-truth transcriptions.
        predictions: Model-predicted transcriptions.

    Returns:
        WER as a float in [0, +inf).
    """
    refs = [r.lower() for r in references]
    preds = [p.lower() for p in predictions]
    return float(jiwer.wer(refs, preds))


def compute_cer(references: list[str], predictions: list[str]) -> float:
    """Compute Character Error Rate between reference and predicted transcriptions.

    Both inputs are lowercased before comparison.

    Args:
        references: Ground-truth transcriptions.
        predictions: Model-predicted transcriptions.

    Returns:
        CER as a float in [0, +inf).
    """
    refs = [r.lower() for r in references]
    preds = [p.lower() for p in predictions]
    return float(jiwer.cer(refs, preds))


def compute_all_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    """Compute WER and CER and return them in a dict.

    Args:
        references: Ground-truth transcriptions.
        predictions: Model-predicted transcriptions.

    Returns:
        Dict with ``"wer"`` and ``"cer"`` keys.
    """
    return {
        "wer": compute_wer(references, predictions),
        "cer": compute_cer(references, predictions),
    }


def save_metrics(
    metrics: dict[str, float],
    output_path: str | Path,
) -> None:
    """Persist metrics to a JSON file.

    Args:
        metrics: Dict of metric name to value.
        output_path: File path for the JSON output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", output_path)


def save_prediction_examples(
    references: list[str],
    predictions: list[str],
    output_path: str | Path,
    num_examples: int = 20,
) -> None:
    """Save side-by-side prediction examples to a JSON file.

    Args:
        references: Ground-truth transcriptions.
        predictions: Model-predicted transcriptions.
        output_path: File path for the JSON output.
        num_examples: Maximum number of examples to save.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for i, (ref, pred) in enumerate(zip(references, predictions)):
        if i >= num_examples:
            break
        examples.append({
            "index": i,
            "reference": ref,
            "prediction": pred,
            "wer": float(jiwer.wer(ref.lower(), pred.lower())),
        })
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d prediction examples to %s", len(examples), output_path)
