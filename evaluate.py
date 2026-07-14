#!/usr/bin/env python3
"""WAXAL ASR — Evaluation Script.

Runs inference on a test or validation split, computes WER / CER,
performs error analysis, generates visualizations, and produces
an HTML error analysis report.

Usage::

    python evaluate.py                             # default config
    python evaluate.py --config configs/sna.yaml   # Shona
    python evaluate.py --split validation           # eval on val split
    python evaluate.py --experiment experiments/20250714_120000_sna

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets
import torch

from src.config import load_config
from src.dataset import load_datasets_from_config
from src.error_analysis import (
    build_error_analysis,
    generate_html_report,
    save_error_analysis_json,
)
from src.experiment import Experiment
from src.inference import transcribe_batch
from src.logging_utils import log_gpu_memory_bar, setup_colored_logging
from src.metrics import compute_all_metrics, save_metrics, save_prediction_examples
from src.model import load_model_and_processor
from src.utils import (
    get_device,
    log_environment,
    log_gpu_info,
    log_memory_usage,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Gemma ASR model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML override config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Which split to evaluate on.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for metrics and prediction outputs.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Path to an existing experiment directory to save results into.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (creates a new experiment if --experiment is not set).",
    )
    return parser.parse_args()


def run_evaluation(
    eval_dataset: datasets.IterableDataset,
    model: Any,
    processor: Any,
    device: torch.device,
    num_samples: int = 200,
    batch_size: int = 4,
    max_new_tokens: int = 128,
) -> tuple[list[str], list[str]]:
    """Run inference on *eval_dataset* and return (references, predictions).

    Args:
        eval_dataset: Chat-formatted evaluation dataset.
        model: The fine-tuned model.
        processor: The model processor.
        device: Torch device.
        num_samples: Max examples to evaluate.
        batch_size: Inference batch size.
        max_new_tokens: Max tokens to generate per example.

    Returns:
        Tuple of (references, predictions) string lists.
    """
    model.eval()

    references: list[str] = []
    predictions: list[str] = []

    ds = eval_dataset.take(num_samples)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch in tqdm(ds.batch(batch_size=batch_size), total=num_batches, desc="Evaluating"):
        preds = transcribe_batch(batch, model, processor, device, max_new_tokens)
        references.extend(batch["transcription"])
        predictions.extend(preds)

    return references, predictions


def main() -> None:
    """Entry point for evaluation."""
    args = parse_args()

    config = load_config(override_path=args.config)
    logging_cfg = config.get("logging", {})
    experiment_cfg = config.get("experiment", {})

    if logging_cfg.get("colored", True):
        setup_colored_logging()
    else:
        setup_logging()

    set_seed(config["seed"])
    log_environment()
    log_gpu_info()

    eval_cfg = config["evaluation"]
    num_samples = args.num_samples or eval_cfg["num_samples"]
    batch_size = args.batch_size or eval_cfg["batch_size"]
    max_new_tokens = eval_cfg["max_new_tokens"]

    # Determine output directory
    if args.experiment:
        output_dir = Path(args.experiment)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["training"]["output_dir"]) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create or reuse experiment
    experiment: Experiment | None = None
    if experiment_cfg.get("enabled", True) and not args.experiment:
        experiment = Experiment.create(config=config, name=args.name)
        output_dir = experiment.dir

    # ---- Load data ----
    splits = load_datasets_from_config(config)
    eval_ds = splits[args.split]

    # ---- Load model ----
    model, processor = load_model_and_processor(config)
    device = get_device()
    log_gpu_memory_bar()

    # ---- Run evaluation ----
    logger.info("Evaluating on %s split (%d samples)...", args.split, num_samples)
    references, predictions = run_evaluation(
        eval_dataset=eval_ds,
        model=model,
        processor=processor,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # ---- Compute and save metrics ----
    metrics = compute_all_metrics(references, predictions)
    languages = config["dataset"]["languages"]
    lang_str = "-".join(languages)

    logger.info("Results on %s %s set:", lang_str, args.split)
    logger.info("  WER : %.2f%%", metrics["wer"] * 100)
    logger.info("  CER : %.2f%%", metrics["cer"] * 100)

    save_metrics(metrics, output_dir / "metrics.json")
    save_prediction_examples(references, predictions, output_dir / "predictions.json")

    # ---- Error analysis ----
    logger.info("Running error analysis...")
    analysis = build_error_analysis(references, predictions)
    save_error_analysis_json(analysis, output_dir / "error_analysis.json")

    # ---- HTML report ----
    if experiment_cfg.get("generate_html_report", True):
        report_title = f"WAXAL ASR Error Analysis — {lang_str} ({args.split})"
        generate_html_report(analysis, output_dir / "error_report.html", title=report_title)
        logger.info("HTML report: %s", output_dir / "error_report.html")

    # ---- Visualization ----
    if experiment_cfg.get("generate_plots", True):
        try:
            from src.visualization import plot_prediction_table
            per_wer = analysis["per_example_wer"]
            plot_prediction_table(
                references, predictions, per_wer,
                output_dir / "prediction_comparison.png",
                title=f"Worst Predictions — {lang_str}",
            )
        except ImportError:
            logger.warning("matplotlib not available — skipping prediction table plot.")

    # ---- Register experiment ----
    if experiment:
        experiment.log_metrics(metrics, tag=args.split)
        experiment.save_predictions(references, predictions, tag=args.split)
        experiment.register(metrics)

    log_gpu_memory_bar()
    logger.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
