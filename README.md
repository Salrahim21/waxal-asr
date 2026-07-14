# WAXAL ASR — Gemma 3n Fine-Tuning for African Language Speech Recognition

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffd21e?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Competition](https://img.shields.io/badge/Zindi-WAXAL%20Challenge-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Fine-tune Google's [Gemma 3n](https://huggingface.co/google/gemma-3n-E2B-it) multimodal model for Automatic Speech Recognition (ASR) on African languages, built for the [Google Research WAXAL Challenge](https://zindi.africa/) on the Zindi platform.

---

## Motivation

Sub-Saharan Africa is home to over 2,000 languages, most of which remain underserved by commercial ASR systems. The WaxalNLP dataset represents a significant step toward closing this gap, providing transcribed speech data for 27 African languages. This repository provides a modular, reproducible pipeline to fine-tune state-of-the-art multimodal models on this data.

## Competition Overview

The WAXAL African Language ASR Challenge asks participants to build speech recognition models for three target languages:

| Language | ISO Code | Training Examples | Test Examples |
|----------|----------|-------------------|---------------|
| Luganda  | `lug`    | 2,602             | 638           |
| Lingala  | `lin`    | 9,937             | 1,866         |
| Shona    | `sna`    | 10,489            | 1,749         |

**Evaluation metric:** Word Error Rate (WER) — lower is better.

**Submission format:** CSV with columns `ID` and `Target` (predicted transcription).

## Architecture

```
                         ┌─────────────────────────────┐
                         │      YAML Configuration      │
                         │   configs/default.yaml + override │
                         └──────────────┬──────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
            │  WaxalNLP    │   │   Gemma 3n   │   │  Experiment  │
            │  Dataset     │   │   + LoRA     │   │  Tracker     │
            │  (Streaming) │   │  (bfloat16)  │   │              │
            └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                   │                  │                   │
                   ▼                  ▼                   ▼
            ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
            │ Chat Format  │   │  SFTTrainer  │   │   Registry   │
            │ + Collator   │──▶│  + LoRA      │──▶│   + Plots    │
            │              │   │              │   │   + Reports  │
            └──────────────┘   └──────────────┘   └──────┬───────┘
                                                         │
                               ┌─────────────────────────┼────────┐
                               ▼                         ▼        ▼
                        ┌────────────┐           ┌────────────┐ ┌──────┐
                        │ Evaluation │           │   Error    │ │ HTML │
                        │ WER / CER  │           │  Analysis  │ │Report│
                        └─────┬──────┘           └────────────┘ └──────┘
                              │
                              ▼
                       ┌─────────────┐
                       │ submission  │
                       │    .csv     │
                       └─────────────┘
```

## Experiment Workflow

```
python train.py --config configs/sna.yaml --name "exp-001"
       │
       ├──▶ experiments/exp-001/
       │       ├── config.yaml          # Frozen config snapshot
       │       ├── environment.json     # Software + GPU versions
       │       ├── logs/
       │       │   └── training_log.json
       │       ├── plots/
       │       │   ├── training_loss.png
       │       │   └── eval_metrics.png
       │       ├── predictions/
       │       │   └── predictions_eval.json
       │       ├── checkpoints/
       │       └── metrics_eval.json
       │
       └──▶ experiments/experiment_registry.csv   # Auto-appended
```

## Benchmark Results

| Experiment | Model | Languages | Steps | LR | LoRA r | WER | CER |
|------------|-------|-----------|-------|----|--------|-----|-----|
| Baseline (starter) | gemma-3n-E2B-it | sna | 500 | 1e-3 | 8 | 52.36% | 14.38% |

> Results will be updated as experiments are run.

## Dataset

The [WaxalNLP dataset](https://huggingface.co/datasets/google/WaxalNLP) provides:
- ~1,846 hours of transcribed ASR data across 27 languages
- ~565 hours of TTS recordings
- CC-BY-4.0 license

Each example contains a raw audio waveform and its corresponding transcription. Audio is loaded via HuggingFace `datasets` in streaming mode and resampled to 16 kHz.

## Approach

This pipeline uses **LoRA** (Low-Rank Adaptation) to fine-tune Gemma 3n, which natively accepts audio input alongside text. The model processes speech end-to-end without a separate audio encoder.

Key design decisions:
- **LoRA** over full fine-tuning: reduces trainable parameters from ~2B to ~1M, enabling single-GPU training
- **Streaming datasets**: avoids downloading hundreds of GB upfront
- **Chat formatting**: wraps audio in the instruction-tuning format Gemma expects
- **bfloat16 precision**: maintains numerical stability with half the memory of float32

## Repository Structure

```
waxal-asr/
├── configs/
│   ├── default.yaml          # Base configuration (all parameters)
│   ├── lug.yaml              # Luganda-specific overrides
│   ├── lin.yaml              # Lingala-specific overrides
│   ├── sna.yaml              # Shona-specific overrides
│   └── multilingual.yaml     # Train on all three languages
├── src/
│   ├── __init__.py
│   ├── config.py             # YAML loading and deep-merge logic
│   ├── utils.py              # Seed, logging, GPU info, memory reporting
│   ├── logging_utils.py      # Rich colored logging, GPU bars, timers
│   ├── preprocessing.py      # Chat-message formatting for Gemma
│   ├── dataset.py            # HuggingFace dataset loading and statistics
│   ├── model.py              # Model and processor loading
│   ├── collator.py           # Tokenisation and label masking
│   ├── trainer.py            # LoRA config, training args, SFTTrainer setup
│   ├── inference.py          # Batch and single-file transcription
│   ├── metrics.py            # WER, CER computation and persistence
│   ├── experiment.py         # Experiment tracking and registry
│   ├── visualization.py      # Training curves, distributions, tables
│   └── error_analysis.py     # Error breakdown and HTML report generation
├── notebooks/
│   ├── waxal_asr_train.ipynb                 # Refactored training notebook
│   └── Waxal_Challenge_Starter_Code.ipynb    # Original starter notebook
├── experiments/              # Auto-managed experiment directories
│   └── experiment_registry.csv
├── models/                   # Saved model checkpoints (gitignored)
├── outputs/                  # Training outputs and logs (gitignored)
├── reports/                  # Generated evaluation reports
├── train.py                  # CLI training entry point
├── evaluate.py               # CLI evaluation with error analysis
├── predict.py                # CLI single-file transcription
├── submit.py                 # CLI submission CSV generation
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment specification
├── .gitignore
├── LICENSE                   # Apache 2.0
└── README.md
```

## Installation

### Option A: pip

```bash
git clone https://github.com/Salrahim21/waxal-asr.git
cd waxal-asr
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate waxal-asr
```

### HuggingFace Authentication

Gemma models require accepting the [model license](https://huggingface.co/google/gemma-3n-E2B-it). After accepting:

```bash
huggingface-cli login
```

### Optional: W&B or TensorBoard

```bash
pip install wandb       # For Weights & Biases
pip install tensorboard  # For TensorBoard
```

Set `training.report_to: "wandb"` or `"tensorboard"` in your config YAML.

## Quick Start

### Train on Shona (single language)

```bash
python train.py --config configs/sna.yaml
```

### Train with experiment tracking

```bash
python train.py --config configs/sna.yaml --name "exp-sna-lr5e4" --notes "Testing lower LR"
```

### Train on all competition languages

```bash
python train.py --config configs/multilingual.yaml
```

### Evaluate with error analysis

```bash
python evaluate.py --config configs/sna.yaml --split test
```

### Transcribe a single audio file

```bash
python predict.py recording.wav --config configs/sna.yaml
```

### Generate submission CSV

```bash
python submit.py --config configs/default.yaml
```

## Training

All training hyperparameters are defined in `configs/default.yaml` and can be overridden per-language or via CLI flags.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.max_steps` | 500 | Training iterations (increase to ~3000 for competition) |
| `training.learning_rate` | 1e-3 | Peak learning rate |
| `training.per_device_train_batch_size` | 2 | Batch size per GPU |
| `training.gradient_accumulation_steps` | 8 | Effective batch size = 2 * 8 = 16 |
| `training.report_to` | `"none"` | Set to `"wandb"` or `"tensorboard"` |
| `lora.r` | 8 | LoRA rank |
| `lora.alpha` | 16 | LoRA scaling factor |
| `lora.target_modules` | `[v_proj, o_proj]` | Adapted attention projections |

### CLI overrides

```bash
python train.py --config configs/sna.yaml --max-steps 3000 --learning-rate 5e-4
```

## Experiment Tracking

Every training run automatically creates an experiment directory under `experiments/` containing:

| Artifact | Description |
|----------|-------------|
| `config.yaml` | Frozen copy of the full merged config |
| `environment.json` | Python, PyTorch, CUDA, GPU info, git commit hash |
| `logs/training_log.json` | Per-step metrics (loss, eval_loss, etc.) |
| `plots/training_loss.png` | Loss curve with EMA smoothing |
| `plots/eval_metrics.png` | WER/CER validation curves |
| `metrics_eval.json` | Final evaluation metrics |
| `predictions/` | Reference vs. prediction JSON files |

The global registry at `experiments/experiment_registry.csv` is auto-appended after each run for cross-experiment comparison.

## Error Analysis

After evaluation, an HTML report is generated with:
- Summary statistics (mean/median/std WER)
- Highest-WER samples (worst predictions)
- Lowest-WER samples (best predictions)
- Longest and shortest transcript analysis
- Most common substitutions, insertions, and deletions

```bash
python evaluate.py --config configs/sna.yaml
# Open: outputs/eval/error_report.html
```

## Inference

The `predict.py` script accepts any audio file format supported by `librosa` (WAV, FLAC, MP3, OGG):

```bash
python predict.py audio1.wav audio2.flac --config configs/sna.yaml
```

Output:
```
[audio1.wav] Mudhuri mikuru mirefu yekuturikidzanwa...
[audio2.flac] Amaato abali gali ku mazzi...
```

## Roadmap

- [x] Experiment tracking with auto-registry
- [x] Rich colored console logging with GPU memory bars
- [x] Training loss and validation metric visualization
- [x] Error analysis with HTML reports
- [x] W&B and TensorBoard support (optional)
- [x] Reproducibility snapshots (git hash, env info, frozen configs)
- [ ] Learning rate scheduling (cosine annealing with warmup)
- [ ] Audio data augmentation (speed perturbation, noise injection)
- [ ] Support Gemma 4n (`google/gemma-4n-E4B-it`) for higher capacity
- [ ] Multi-GPU training via DeepSpeed / FSDP
- [ ] Quantised inference (4-bit / 8-bit) for deployment
- [ ] Ensemble decoding across language-specific and multilingual models

## Future Improvements

- **Larger LoRA rank**: `r=16` or `r=32` with additional target modules (`q_proj`, `k_proj`) for more adapter capacity
- **Longer training**: 3,000–10,000 steps with proper learning rate scheduling
- **Multilingual pretraining**: train a shared model, then fine-tune per-language
- **Text normalisation**: language-specific text cleaning for evaluation consistency
- **Beam search decoding**: use `num_beams > 1` at inference time

## Acknowledgements

- [Google Research](https://research.google/) for the WaxalNLP dataset and Gemma model family
- [Zindi](https://zindi.africa/) for hosting the competition
- [HuggingFace](https://huggingface.co/) for the Transformers, PEFT, TRL, and Datasets libraries
- The original [starter notebook](https://huggingface.co/datasets/google/WaxalNLP) provided by the competition organisers

## References

1. Gemma Team. *Gemma 3 Technical Report*. Google DeepMind, 2025.
2. Hu, E.J., et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
3. Morris, A.C., et al. *From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition*. INTERSPEECH 2004.
4. WaxalNLP Dataset: [huggingface.co/datasets/google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The WaxalNLP dataset is licensed under CC-BY-4.0. The Gemma model is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
