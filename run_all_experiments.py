#!/usr/bin/env python3
"""WAXAL ASR — Run All Experiments.

Executes three baseline experiments sequentially, generates submissions,
and produces a final day-1 report.

Pipeline per experiment:
    1. Load YAML config
    2. Train the model
    3. Evaluate on validation
    4. Compute WER / CER
    5. Save checkpoint, metrics, predictions, training log, config
    6. Generate submission CSV
    7. Validate submission against SampleSubmission.csv
    8. Append to experiment registry

Usage::

    python run_all_experiments.py

If any experiment crashes, the pipeline continues with the remaining ones.
"""

from __future__ import annotations

import csv
import functools
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets
import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.error_analysis import build_error_analysis, generate_html_report, save_error_analysis_json
from src.experiment import Experiment, collect_environment_info, get_git_commit_hash
from src.logging_utils import setup_colored_logging, log_gpu_memory_bar, log_training_summary
from src.metrics import compute_all_metrics, save_metrics
from src.utils import set_seed, log_environment, log_gpu_info, get_device
from src.visualization import plot_training_loss, plot_eval_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "baseline_v1",
        "config_path": "configs/baseline_v1.yaml",
        "submission_file": "submissions/submission_exp001.csv",
    },
    {
        "name": "baseline_v2",
        "config_path": "configs/baseline_v2.yaml",
        "submission_file": "submissions/submission_exp002.csv",
    },
    {
        "name": "baseline_v3",
        "config_path": "configs/baseline_v3.yaml",
        "submission_file": "submissions/submission_exp003.csv",
    },
]


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    exp_def: dict[str, str],
) -> dict[str, Any]:
    """Run a single experiment: train, evaluate, generate submission.

    Args:
        exp_def: Dict with ``name``, ``config_path``, ``submission_file``.

    Returns:
        Dict with results: metrics, paths, timing, or error info.
    """
    exp_name = exp_def["name"]
    config_path = exp_def["config_path"]
    submission_path = Path(exp_def["submission_file"])

    logger.info("=" * 60)
    logger.info("  EXPERIMENT: %s", exp_name)
    logger.info("  Config    : %s", config_path)
    logger.info("=" * 60)

    start_time = time.monotonic()
    config = load_config(override_path=config_path)
    seed = config["seed"]
    set_seed(seed)

    training_cfg = config["training"]
    eval_cfg = config["evaluation"]
    inference_cfg = config.get("inference", {})
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    architecture = model_cfg.get("architecture", "gemma")

    # Create experiment directory
    experiment = Experiment.create(config=config, name=exp_name, notes=config.get("experiment", {}).get("notes", ""))
    exp_dir = experiment.dir

    # ---- Load model and data ----
    if architecture == "whisper":
        from src.whisper_model import load_whisper_model_and_processor
        from src.whisper_dataset import load_whisper_datasets, WhisperDataCollator
        from src.whisper_trainer import build_whisper_trainer
        from src.whisper_inference import whisper_transcribe_batch

        model, processor = load_whisper_model_and_processor(config)
        device = get_device()
        model.to(device)

        splits = load_whisper_datasets(config, processor)
        train_ds = splits["train"]
        val_ds = splits["validation"]
        test_ds = splits["test"]

        data_collator = WhisperDataCollator(processor)

        # Prepare datasets for trainer
        num_val = training_cfg.get("num_validation_examples", 200)
        val_ds_fixed = val_ds.take(num_val)

        # Shuffle and repeat training data
        shuffled_train = train_ds.shuffle(
            buffer_size=training_cfg.get("shuffle_buffer_size", 1000),
            seed=seed,
        )

        # ---- Train ----
        logger.info("Training %s...", exp_name)
        trainer = build_whisper_trainer(
            model=model,
            processor=processor,
            train_dataset=shuffled_train,
            eval_dataset=val_ds_fixed,
            data_collator=data_collator,
            config=config,
        )
        train_result = trainer.train()

        # Save training log
        if hasattr(trainer, "state") and trainer.state.log_history:
            for entry in trainer.state.log_history:
                step = entry.get("step", 0)
                experiment.log_step(step, entry)
            experiment.save_training_log()

        # Save checkpoint
        checkpoint_dir = exp_dir / "checkpoints" / "final"
        trainer.save_model(str(checkpoint_dir))
        logger.info("Checkpoint saved to %s", checkpoint_dir)

        # ---- Evaluate on validation ----
        logger.info("Evaluating %s on validation...", exp_name)
        num_beams = inference_cfg.get("num_beams", 1)
        length_penalty = inference_cfg.get("length_penalty", 1.0)
        max_new_tokens = inference_cfg.get("max_new_tokens", 225)

        # Re-load validation with transcriptions for WER
        val_for_eval = datasets.IterableDataset
        references = []
        predictions = []
        num_eval = eval_cfg.get("num_samples", 200)

        # Need raw val data with transcriptions for evaluation
        for lang in dataset_cfg["languages"]:
            raw_val = datasets.load_dataset(
                dataset_cfg["dataset_id"],
                name=f"{lang}_asr",
                split="validation",
                streaming=True,
            )
            raw_val = raw_val.cast_column("audio", datasets.Audio(sampling_rate=dataset_cfg["sample_rate"]))

            count = 0
            for example in raw_val:
                if count >= num_eval // len(dataset_cfg["languages"]):
                    break
                audio_array = np.asarray(example["audio"]["array"], dtype=np.float32)
                input_features = processor.feature_extractor(
                    audio_array,
                    sampling_rate=dataset_cfg["sample_rate"],
                    return_tensors="pt",
                ).input_features.to(device)

                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                    )

                transcript = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
                references.append(example["transcription"])
                predictions.append(transcript)
                count += 1

    else:
        # Gemma path (existing pipeline)
        from src.model import load_model_and_processor
        from src.dataset import load_datasets_from_config
        from src.trainer import build_trainer
        from src.inference import transcribe_batch

        model, processor = load_model_and_processor(config)
        device = get_device()

        splits = load_datasets_from_config(config)
        train_ds = splits["train"]
        val_ds = splits["validation"]

        shuffled_train = train_ds.shuffle(
            buffer_size=training_cfg.get("shuffle_buffer_size", 1000),
            seed=seed,
        ).repeat(None)

        num_val = training_cfg.get("num_validation_examples", 200)
        val_ds_fixed = val_ds.take(num_val)

        trainer = build_trainer(
            model=model, processor=processor,
            train_dataset=shuffled_train, eval_dataset=val_ds_fixed,
            config=config,
        )

        logger.info("Training %s...", exp_name)
        trainer.train()

        if hasattr(trainer, "state") and trainer.state.log_history:
            for entry in trainer.state.log_history:
                experiment.log_step(entry.get("step", 0), entry)
            experiment.save_training_log()

        checkpoint_dir = exp_dir / "checkpoints" / "final"
        trainer.save_model(str(checkpoint_dir))

        # Evaluate
        logger.info("Evaluating %s...", exp_name)
        references = []
        predictions = []
        eval_ds = splits["validation"]
        num_eval = eval_cfg.get("num_samples", 200)
        batch_size = eval_cfg.get("batch_size", 4)
        model.eval()

        for batch in tqdm(eval_ds.take(num_eval).batch(batch_size=batch_size), desc="Evaluating"):
            preds = transcribe_batch(batch, model, processor, device, eval_cfg.get("max_new_tokens", 128))
            references.extend(batch["transcription"])
            predictions.extend(preds)

    # ---- Compute metrics ----
    metrics = compute_all_metrics(references, predictions)
    elapsed = time.monotonic() - start_time

    logger.info("[%s] WER: %.2f%% | CER: %.2f%%", exp_name, metrics["wer"] * 100, metrics["cer"] * 100)

    # Save metrics
    experiment.log_metrics(metrics, tag="validation")
    save_metrics(metrics, exp_dir / "metrics.json")

    # Save predictions
    experiment.save_predictions(references, predictions, tag="validation")

    # Save predictions CSV
    predictions_csv = exp_dir / "predictions.csv"
    with open(predictions_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reference", "prediction"])
        for ref, pred in zip(references, predictions):
            writer.writerow([ref, pred])

    # ---- Error analysis ----
    analysis = build_error_analysis(references, predictions)
    save_error_analysis_json(analysis, exp_dir / "error_analysis.json")
    generate_html_report(analysis, exp_dir / "error_report.html",
                         title=f"Error Analysis — {exp_name}")

    # ---- Plots ----
    try:
        plots_dir = exp_dir / "plots"
        plot_training_loss(experiment._training_log, plots_dir / "training_loss.png")
        plot_eval_metrics(experiment._training_log, plots_dir / "eval_metrics.png")
    except Exception:
        logger.warning("Plot generation failed — continuing.")

    # ---- Generate submission ----
    logger.info("Generating submission for %s...", exp_name)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    test_csv = Path(config.get("submission", {}).get("test_csv", "./Test.csv"))
    sample_csv = Path(config.get("submission", {}).get("sample_submission_csv", "./SampleSubmission.csv"))

    if test_csv.exists():
        _generate_submission(
            config=config,
            model=model,
            processor=processor,
            device=device,
            architecture=architecture,
            test_csv_path=test_csv,
            sample_csv_path=sample_csv,
            output_path=submission_path,
        )
    else:
        logger.warning("Test.csv not found at %s — skipping submission generation.", test_csv)

    # ---- Register ----
    experiment._best_checkpoint_path = str(checkpoint_dir)
    experiment.register(metrics)

    log_gpu_memory_bar()

    return {
        "name": exp_name,
        "status": "completed",
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(checkpoint_dir),
        "submission_file": str(submission_path),
        "experiment_dir": str(exp_dir),
    }


def _generate_submission(
    config: dict[str, Any],
    model: Any,
    processor: Any,
    device: torch.device,
    architecture: str,
    test_csv_path: Path,
    sample_csv_path: Path,
    output_path: Path,
) -> None:
    """Generate a submission CSV from the test set.

    Args:
        config: Full configuration dictionary.
        model: Trained model.
        processor: Model processor.
        device: Torch device.
        architecture: ``"whisper"`` or ``"gemma"``.
        test_csv_path: Path to Test.csv.
        sample_csv_path: Path to SampleSubmission.csv.
        output_path: Where to write the submission CSV.
    """
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})
    max_new_tokens = inference_cfg.get("max_new_tokens", 225)
    num_beams = inference_cfg.get("num_beams", 1)
    length_penalty = inference_cfg.get("length_penalty", 1.0)

    # Load test IDs
    test_entries: list[dict[str, str]] = []
    with open(test_csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            test_id = row["ID"]
            language = test_id.rsplit("_", 1)[0]
            test_entries.append({"id": test_id, "language": language})

    languages_in_test = sorted({e["language"] for e in test_entries})
    predictions_map: dict[str, str] = {}

    model.eval()

    for language in languages_in_test:
        lang_ids = {e["id"] for e in test_entries if e["language"] == language}
        logger.info("Generating predictions for %s (%d examples)...", language, len(lang_ids))

        ds = datasets.load_dataset(
            dataset_cfg["dataset_id"],
            name=f"{language}_asr",
            split="test",
            streaming=True,
        )
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=dataset_cfg["sample_rate"]))

        for example in tqdm(ds, desc=f"Predict {language}"):
            audio_array = np.asarray(example["audio"]["array"], dtype=np.float32)

            if architecture == "whisper":
                input_features = processor.feature_extractor(
                    audio_array, sampling_rate=dataset_cfg["sample_rate"],
                    return_tensors="pt",
                ).input_features.to(device)

                with torch.no_grad():
                    pred_ids = model.generate(
                        input_features,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                    )
                transcript = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
            else:
                # Gemma path — use the chat formatting
                from src.preprocessing import format_for_chat
                prompt_cfg = config["prompts"]
                formatted = format_for_chat(example, prompt_cfg["system_message"], prompt_cfg["user_message"])
                messages = formatted["messages"][:-1]  # Remove assistant turn

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
                transcript = processor.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            # Map to test ID
            sample_id = example.get("id", "")
            full_id = f"{language}_{sample_id}" if not str(sample_id).startswith(language) else str(sample_id)
            if full_id in lang_ids:
                predictions_map[full_id] = transcript

    # Write submission
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ID", "Target"])
        for entry in test_entries:
            writer.writerow([entry["id"], predictions_map.get(entry["id"], "")])

    logger.info("Submission written to %s (%d rows)", output_path, len(test_entries))

    # Validate
    if sample_csv_path.exists():
        with open(sample_csv_path, "r", encoding="utf-8") as fh:
            expected_ids = {row["ID"] for row in csv.DictReader(fh)}
        with open(output_path, "r", encoding="utf-8") as fh:
            submitted_ids = {row["ID"] for row in csv.DictReader(fh)}
        missing = expected_ids - submitted_ids
        if missing:
            logger.error("Submission missing %d IDs!", len(missing))
        else:
            logger.info("Submission validation PASSED.")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_day1_report(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate the day-1 markdown report.

    Args:
        results: List of per-experiment result dicts.
        output_path: Path for the markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    env_info = collect_environment_info()

    # Find best experiment
    completed = [r for r in results if r["status"] == "completed"]
    best = min(completed, key=lambda r: r["wer"]) if completed else None

    lines = [
        "# WAXAL ASR — Day 1 Report",
        "",
        f"**Generated:** {now}",
        f"**Git commit:** `{get_git_commit_hash()}`",
        "",
        "---",
        "",
        "## Environment",
        "",
        f"- **Python:** {env_info.get('python', 'N/A').split()[0]}",
        f"- **PyTorch:** {env_info.get('pytorch', 'N/A')}",
        f"- **CUDA:** {env_info.get('cuda_version', 'N/A')}",
        f"- **GPU:** {env_info.get('gpu_0', 'N/A')}",
        "",
        "## Experiments Summary",
        "",
        "| Experiment | Status | WER | CER | Training Time | Submission |",
        "|------------|--------|-----|-----|---------------|------------|",
    ]

    for r in results:
        status = r["status"]
        if status == "completed":
            wer_str = f"{r['wer']:.2%}"
            cer_str = f"{r['cer']:.2%}"
            elapsed = r["elapsed_seconds"]
            h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
            sub_str = f"`{r['submission_file']}`"
        else:
            wer_str = "FAILED"
            cer_str = "FAILED"
            time_str = "N/A"
            sub_str = "N/A"

        lines.append(f"| {r['name']} | {status} | {wer_str} | {cer_str} | {time_str} | {sub_str} |")

    lines.extend([
        "",
        "## Best Experiment",
        "",
    ])

    if best:
        lines.extend([
            f"**{best['name']}** achieved the lowest WER of **{best['wer']:.2%}** "
            f"(CER: {best['cer']:.2%}).",
            "",
            f"- Checkpoint: `{best['checkpoint_path']}`",
            f"- Submission: `{best['submission_file']}`",
        ])
    else:
        lines.append("No experiments completed successfully.")

    lines.extend([
        "",
        "## Leaderboard-Ready Submissions",
        "",
    ])

    for r in completed:
        lines.append(f"- `{r['submission_file']}`")

    lines.extend([
        "",
        "## Recommendations for Tomorrow",
        "",
        "1. **Increase training epochs** to 5-10 for the best-performing config",
        "2. **Try LoRA fine-tuning** with r=16 to reduce overfitting risk",
        "3. **Add language-specific models** — train separate models per language",
        "4. **Experiment with Whisper Medium** for higher capacity",
        "5. **Implement data augmentation** — speed perturbation, noise injection",
        "6. **Try learning rate scheduling** — cosine annealing with warmup",
        "7. **Ensemble predictions** from multiple checkpoints / models",
        "",
        "---",
        "",
        f"*Report generated automatically by `run_all_experiments.py`*",
    ])

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Day 1 report saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all experiments sequentially with crash recovery."""
    setup_colored_logging()
    log_environment()
    log_gpu_info()

    logger.info("=" * 60)
    logger.info("  WAXAL ASR — Running %d Experiments", len(EXPERIMENTS))
    logger.info("=" * 60)

    results: list[dict[str, Any]] = []

    for i, exp_def in enumerate(EXPERIMENTS, 1):
        logger.info("")
        logger.info(">>> Starting experiment %d/%d: %s", i, len(EXPERIMENTS), exp_def["name"])
        logger.info("")

        try:
            result = run_single_experiment(exp_def)
            results.append(result)
            print(f"\n  Experiment {i} Complete: {exp_def['name']}\n")
        except Exception as e:
            logger.error("Experiment %s FAILED: %s", exp_def["name"], e)
            logger.error(traceback.format_exc())
            results.append({
                "name": exp_def["name"],
                "status": "failed",
                "error": str(e),
                "wer": float("inf"),
                "cer": float("inf"),
                "elapsed_seconds": 0,
                "checkpoint_path": "",
                "submission_file": "",
                "experiment_dir": "",
            })
            print(f"\n  Experiment {i} FAILED: {exp_def['name']} ({e})\n")
            # Continue with remaining experiments
            continue

        # Clear GPU memory between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Generate final report ----
    report_path = Path("reports") / "day1_report.md"
    generate_day1_report(results, report_path)

    # ---- Print summary ----
    print("\n" + "=" * 50)
    completed_count = sum(1 for r in results if r["status"] == "completed")
    submission_count = sum(1 for r in results if r["status"] == "completed" and Path(r["submission_file"]).exists())

    for r in results:
        if r["status"] == "completed":
            print(f"  Experiment {r['name']} Complete (WER: {r['wer']:.2%})")
        else:
            print(f"  Experiment {r['name']} FAILED")

    print(f"\n  {completed_count} experiments completed")
    print(f"  {submission_count} submission files generated")
    if submission_count > 0:
        print("  Ready for Zindi upload")
    print("=" * 50)


if __name__ == "__main__":
    main()
