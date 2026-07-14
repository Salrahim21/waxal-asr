# WAXAL ASR — Day 1 Report

**Generated:** 2026-07-14 14:22:14 UTC
**Git commit:** `81c6c5e`

---

## Environment

- **Python:** 3.10.11
- **PyTorch:** 2.6.0+cu126
- **CUDA:** 12.6
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)

## Experiments Summary

| Experiment | Status | WER | CER | Training Time | Submission |
|------------|--------|-----|-----|---------------|------------|
| baseline_v1 | failed | FAILED | FAILED | N/A | N/A |
| baseline_v2 | failed | FAILED | FAILED | N/A | N/A |
| baseline_v3 | failed | FAILED | FAILED | N/A | N/A |

## Best Experiment

No experiments completed successfully.

## Leaderboard-Ready Submissions


## Recommendations for Tomorrow

1. **Increase training epochs** to 5-10 for the best-performing config
2. **Try LoRA fine-tuning** with r=16 to reduce overfitting risk
3. **Add language-specific models** — train separate models per language
4. **Experiment with Whisper Medium** for higher capacity
5. **Implement data augmentation** — speed perturbation, noise injection
6. **Try learning rate scheduling** — cosine annealing with warmup
7. **Ensemble predictions** from multiple checkpoints / models

---

*Report generated automatically by `run_all_experiments.py`*