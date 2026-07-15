# WAXAL ASR — Whisper for African Language Speech Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffd21e?logo=huggingface&logoColor=black)
![Competition](https://img.shields.io/badge/Zindi-WAXAL%20Challenge-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Speech recognition pipeline for the [Google Research WAXAL African Language ASR Challenge](https://zindi.africa/) on Zindi. Uses OpenAI's **Whisper Large-V3** with beam search for zero-shot inference, heuristic ensembling, and optional LoRA fine-tuning on three African languages.

---

## Competition Overview

The WAXAL challenge asks participants to transcribe speech in three African languages:

| Language | ISO Code | Train Examples | Test Examples |
|----------|----------|----------------|---------------|
| Luganda  | `lug`    | 2,602          | 638           |
| Lingala  | `lin`    | 9,937          | 1,832         |
| Shona    | `sna`    | 10,489         | 1,596         |

**Evaluation metric:** Word Error Rate (WER) — lower is better.

**Submission format:** CSV with columns `ID` and `Target` (predicted transcription), 4,253 test samples total.

## Dataset

The [WaxalNLP dataset](https://huggingface.co/datasets/google/WaxalNLP) is hosted on HuggingFace. The Zindi competition CSVs (`Train.csv`, `Test.csv`) contain metadata only — audio must be downloaded from HuggingFace.

**Test split parquet files (for zero-shot inference):**

| Language | File | Size |
|----------|------|------|
| Luganda | [`lug-test-00000.parquet`](https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/ASR/lug/lug-test-00000.parquet) | ~216 MB |
| Lingala | [`lin-test-00000.parquet`](https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/ASR/lin/lin-test-00000.parquet) | ~494 MB |
| Shona | [`sna-test-00000.parquet`](https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/ASR/sna/sna-test-00000.parquet) | ~552 MB |

Total test data: **~1.3 GB**

**Additional test shards (for 100% coverage):**

| Language | File | Size |
|----------|------|------|
| Lingala | [`lin-test-00001.parquet`](https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/ASR/lin/lin-test-00001.parquet) | ~8.5 MB |
| Shona | [`sna-test-00001.parquet`](https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/ASR/sna/sna-test-00001.parquet) | ~47 MB |

## Approaches

This repo implements **4 tiers** of increasing complexity:

### Tier 1+3: Whisper Large-V3 Zero-Shot + Beam Search (Recommended Start)

The fastest path to a competitive submission — no training data needed.

1. Clone and install:
```bash
git clone https://github.com/Salrahim21/waxal-asr.git
cd waxal-asr
pip install -r requirements.txt
```

2. Download test parquet files (links above) into `data/`

3. Create `.env` with your HuggingFace token:
```
HF_TOKEN=your_token_here
```

4. Open `notebooks/whisper_train_submit.ipynb` and run all cells:
   - Loads `whisper-large-v3` (1.55B params, ~3GB VRAM in float16)
   - Beam search decoding (`num_beams=5`, `no_repeat_ngram_size=3`)
   - Language-specific transcription (Swahili for Luganda, Lingala, Shona)
   - Outputs `submissions/submission_large_v3.csv`

### Tier 2: Ensemble (Small + Large-V3)

Combines predictions from Whisper Small and Large-V3 using quality heuristics.

1. Generate both submissions first (Tier 1 + original zero-shot)
2. Open `notebooks/whisper_ensemble.ipynb` and run all cells
3. Per-sample selection based on: garbled script detection, repetition analysis, emptiness, length
4. Outputs `submissions/submission_ensemble.csv`

### Tier 4: LoRA Fine-Tuning

Fine-tunes Whisper Large-V3 with LoRA on the full training set (~38K examples).

1. Download training + validation parquet files into `data/` (~6.9 GB total)
2. Open `notebooks/whisper_finetune.ipynb` and run all cells
3. LoRA config: `r=32`, `alpha=64`, targets `q_proj, v_proj, k_proj, o_proj`
4. Fits in 8GB VRAM with gradient checkpointing + batch_size=2
5. 500 steps default, evaluates WER every 100 steps
6. Outputs `submissions/submission_finetuned.csv`

### Legacy: Script-Based Experiments

```bash
python run_all_experiments.py
```

Runs 3 baseline configs (`baseline_v1`, `v2`, `v3`) with Whisper Small. Requires full training data download.

## Repository Structure

```
waxal-asr/
├── configs/
│   ├── default.yaml              # Base configuration
│   ├── baseline_v1.yaml          # 1 epoch, lr=1e-5, greedy
│   ├── baseline_v2.yaml          # 3 epochs, lr=5e-6, greedy
│   └── baseline_v3.yaml          # 3 epochs, lr=5e-6, beam search
├── src/
│   ├── competition.py            # Competition compliance guards
│   ├── config.py                 # YAML config loading
│   ├── utils.py                  # Seed, logging, GPU info
│   ├── logging_utils.py          # Colored logging
│   ├── metrics.py                # WER, CER computation
│   ├── whisper_model.py          # Whisper model loading
│   ├── whisper_dataset.py        # Dataset loading and preprocessing
│   ├── whisper_trainer.py        # Seq2Seq trainer setup
│   ├── experiment.py             # Experiment tracking
│   ├── visualization.py          # Training curves and plots
│   └── error_analysis.py         # Error breakdown reports
├── notebooks/
│   ├── whisper_train_submit.ipynb # Large-V3 zero-shot + beam search
│   ├── whisper_ensemble.ipynb     # Ensemble small + large-v3
│   ├── whisper_finetune.ipynb     # LoRA fine-tuning on training data
│   └── waxal_asr_train.ipynb     # Legacy fine-tuning notebook
├── data/                         # Local parquet files (gitignored)
├── submissions/                  # Generated submission CSVs (gitignored)
├── experiments/                  # Experiment outputs (gitignored)
├── run_all_experiments.py        # Automated 3-experiment runner
├── Train.csv                     # Zindi training metadata
├── Test.csv                      # Zindi test IDs (4,253 samples)
├── SampleSubmission.csv          # Submission template
├── requirements.txt
├── reports/
│   ├── zero_shot_performance.md  # Analysis of zero-shot results
│   └── day1_report.md            # Day 1 experiment report
├── .env                          # HF token (gitignored)
└── README.md
```

## Technical Details

- **Models:** `openai/whisper-small` (244M) and `openai/whisper-large-v3` (1.55B)
- **Precision:** float16 (small: ~500 MB, large-v3: ~3 GB VRAM)
- **Inference:** Beam search (5 beams), `no_repeat_ngram_size=3`, language-specific transcription
- **Fine-tuning:** LoRA (r=32, alpha=64) with gradient checkpointing, fits 8GB VRAM
- **GPU tested:** NVIDIA RTX 4060 Laptop (8 GB VRAM)
- **Dependencies:** PyTorch 2.6, Transformers 5.x, datasets 3.2.0, PEFT

## Competition Compliance

Built-in guards in `src/competition.py` ensure:
- Only `google/WaxalNLP` dataset is used
- Only competition languages (`lug`, `lin`, `sna`) are loaded
- Only valid splits (`train`, `validation`, `test`) are accepted

Set `COMPETITION_MODE = False` in `src/competition.py` to use the pipeline outside the competition.

## Acknowledgements

- [Google Research](https://research.google/) for the WaxalNLP dataset
- [OpenAI](https://openai.com/) for the Whisper model family
- [Zindi](https://zindi.africa/) for hosting the competition
- [HuggingFace](https://huggingface.co/) for the Transformers and Datasets libraries

## References

1. Radford, A., et al. *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI, 2022.
2. WaxalNLP Dataset: [huggingface.co/datasets/google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The WaxalNLP dataset is licensed under CC-BY-4.0.
