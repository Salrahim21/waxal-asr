# WAXAL ASR — Whisper for African Language Speech Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffd21e?logo=huggingface&logoColor=black)
![Competition](https://img.shields.io/badge/Zindi-WAXAL%20Challenge-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Speech recognition pipeline for the [Google Research WAXAL African Language ASR Challenge](https://zindi.africa/) on Zindi. Uses OpenAI's **Whisper Small** for zero-shot inference and optional fine-tuning on three African languages.

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

## Quick Start (Zero-Shot Submission)

The fastest path to a submission file — no fine-tuning required.

### 1. Clone and install

```bash
git clone https://github.com/Salrahim21/waxal-asr.git
cd waxal-asr
pip install -r requirements.txt
```

### 2. Download test data

Download the 3 parquet files from the links above and place them in `data/`:

```
data/
  lug-test-00000.parquet
  lin-test-00000.parquet
  sna-test-00000.parquet
```

### 3. Set up HuggingFace token

Create a `.env` file in the project root:

```
HF_TOKEN=your_token_here
```

### 4. Run the notebook

Open `notebooks/whisper_train_submit.ipynb` and run all cells. The notebook will:

1. Load test audio from local parquet files
2. Load Whisper Small (float16, ~500 MB GPU memory)
3. Transcribe all 4,253 test samples zero-shot
4. Write `submissions/submission_zero_shot.csv`
5. Validate the submission against `SampleSubmission.csv`

## Fine-Tuning (Optional)

For better results, fine-tune Whisper on the training data using the automated experiment runner:

```bash
python run_all_experiments.py
```

This runs 3 experiment configurations:

| Config | Epochs | LR | Decoding |
|--------|--------|----|----------|
| `baseline_v1` | 1 | 1e-5 | Greedy |
| `baseline_v2` | 3 | 5e-6 | Greedy |
| `baseline_v3` | 3 | 5e-6 | Beam search (5 beams) |

Fine-tuning requires downloading the full training data (~5 GB).

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
│   ├── whisper_train_submit.ipynb # Zero-shot inference notebook
│   └── waxal_asr_train.ipynb     # Fine-tuning notebook
├── data/                         # Local parquet files (gitignored)
├── submissions/                  # Generated submission CSVs (gitignored)
├── experiments/                  # Experiment outputs (gitignored)
├── run_all_experiments.py        # Automated 3-experiment runner
├── Train.csv                     # Zindi training metadata
├── Test.csv                      # Zindi test IDs (4,253 samples)
├── SampleSubmission.csv          # Submission template
├── requirements.txt
├── .env                          # HF token (gitignored)
└── README.md
```

## Technical Details

- **Model:** `openai/whisper-small` (241.7M parameters)
- **Precision:** float16 (~500 MB GPU memory)
- **Inference:** Greedy decoding, max 225 new tokens
- **GPU tested:** NVIDIA RTX 4060 Laptop (8 GB VRAM)
- **Dependencies:** PyTorch 2.6, Transformers 5.x, datasets 3.2.0

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
